import torch

from aml_fraud.models.deep import Autoencoder, GANomaly, HybridAnomalyNet, VariationalAutoencoder


def test_deep_models_forward_shapes():
    x = torch.randn(4, 12)
    assert Autoencoder(12)(x).shape == x.shape
    assert VariationalAutoencoder(12)(x)[0].shape == x.shape
    assert GANomaly(12)(x)[0].shape == x.shape
    reconstruction, logits = HybridAnomalyNet(12)(x)
    assert reconstruction.shape == x.shape
    assert logits.shape == (4,)
