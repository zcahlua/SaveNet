from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from data_molecule3d import load_molecule3d_datasets
from model.savenet import SaVeNetWrapper


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_target_stats(dataset):
    ys = [dataset[i].y.view(-1).float() for i in range(len(dataset))]
    y = torch.cat(ys, dim=0)
    mean = y.mean()
    std = y.std(unbiased=False) + 1e-8
    return mean, std


def evaluate(model, loader, device):
    model.eval()
    loss_sum = abs_err = sq_err = n = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)["y"]
            target = batch.y
            pred = pred.view_as(target)
            loss = F.mse_loss(pred, target)
            err = pred - target
            bsz = target.numel()
            loss_sum += loss.item() * bsz
            abs_err += err.abs().sum().item()
            sq_err += (err ** 2).sum().item()
            n += bsz
    n = max(n, 1.0)
    return {"loss": loss_sum / n, "mae": abs_err / n, "rmse": (sq_err / n) ** 0.5}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--target-id", type=int, default=5)
    p.add_argument("--split-mode", choices=["random", "scaffold"], default="random")
    p.add_argument("--process-dir-base", default="processed_downstream")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-encoder", type=int, default=8)
    p.add_argument("--num-rbf", type=int, default=32)
    p.add_argument("--cutoff", type=float, default=5.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subset-train", type=int)
    p.add_argument("--subset-val", type=int)
    p.add_argument("--subset-test", type=int)
    p.add_argument("--no-center-positions", dest="center_positions", action="store_false")
    p.set_defaults(center_positions=True)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--out-dir", default="runs/molecule3d_savenet")
    p.add_argument("--eval-only")
    p.add_argument("--save-every", type=int)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    train_ds, val_ds, test_ds = load_molecule3d_datasets(
        root=args.data_root, target_id=args.target_id, split_mode=args.split_mode,
        center_positions=args.center_positions, process_dir_base=args.process_dir_base,
        subset_train=args.subset_train, subset_val=args.subset_val, subset_test=args.subset_test,
    )
    mean, std = compute_target_stats(train_ds)

    model = SaVeNetWrapper(hidden_dim=args.hidden_dim, num_encoder=args.num_encoder, num_rbf=args.num_rbf,
                           cutoff=args.cutoff, target_mean=mean, target_std=std).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and "cuda" in args.device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    if args.eval_only:
        model.load_state_dict(torch.load(args.eval_only, map_location=args.device)["model"])
        print("eval", evaluate(model, test_loader, args.device))
        return

    best_mae = float("inf")
    metrics_path = out_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_mae", "val_rmse"])
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss_sum = n = 0.0
            for batch in tqdm(train_loader, desc=f"epoch {epoch}"):
                batch = batch.to(args.device)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=args.amp and "cuda" in args.device):
                    pred = model(batch)["y"]
                    target = batch.y
                    pred = pred.view_as(target)
                    loss = F.mse_loss(pred, target)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()
                bsz = target.numel()
                train_loss_sum += loss.item() * bsz
                n += bsz

            train_loss = train_loss_sum / max(n, 1.0)
            val_metrics = evaluate(model, val_loader, args.device)
            print(f"epoch={epoch} train_loss={train_loss:.6f} val_mae={val_metrics['mae']:.6f} val_rmse={val_metrics['rmse']:.6f}")
            writer.writerow({"epoch": epoch, "train_loss": train_loss, "val_loss": val_metrics["loss"], "val_mae": val_metrics["mae"], "val_rmse": val_metrics["rmse"]})
            f.flush()

            ckpt = {"model": model.state_dict(), "epoch": epoch, "val": val_metrics}
            torch.save(ckpt, out_dir / "last.pt")
            if val_metrics["mae"] < best_mae:
                best_mae = val_metrics["mae"]
                torch.save(ckpt, out_dir / "best.pt")
            if args.save_every and epoch % args.save_every == 0:
                torch.save(ckpt, out_dir / f"epoch_{epoch}.pt")

    best_path = out_dir / "best.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=args.device)["model"])
    print("test", evaluate(model, test_loader, args.device))


if __name__ == "__main__":
    main()
