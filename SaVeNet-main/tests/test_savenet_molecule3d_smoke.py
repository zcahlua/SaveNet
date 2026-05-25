import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from data_molecule3d import Molecule3DToSaVeNet
from model.savenet import SaVeNetWrapper


def test_savenet_molecule3d_smoke():
    t = Molecule3DToSaVeNet(target_id=5)
    d1 = Data(z=torch.tensor([6, 1, 1]), xyz=torch.randn(3, 3), props=torch.randn(7))
    d2 = Data(z=torch.tensor([8, 1]), xyz=torch.randn(2, 3), props=torch.randn(7))
    batch = Batch.from_data_list([t(d1), t(d2)])

    model = SaVeNetWrapper(hidden_dim=16, num_encoder=2, num_rbf=8, cutoff=5.0)
    out = model(batch)
    assert "y" in out
    assert out["y"].shape[0] == 2

    target = batch.y.view_as(out["y"])
    loss = F.mse_loss(out["y"], target)
    loss.backward()
    assert torch.isfinite(loss)
