import torch
from forge.models._dit import FlagDiT


def test_dit_forward_pass():
    input_dim = 1280
    hidden_dim = 1280
    mlp_ratio = 2.0
    num_heads = 4
    num_layers = 2
    conditioning_dropout = 0.1

    model = FlagDiT(
        input_dim, hidden_dim, mlp_ratio, num_heads, num_layers, conditioning_dropout
    )

    batch_size = 2
    seq_len = 50
    x = torch.randn(batch_size, seq_len, input_dim)  # (bs, L, raygun_dim)
    t = torch.rand(batch_size)  # scalar timesteps
    c = torch.randn(batch_size, seq_len, input_dim)  # conditioning embeddings

    out = model(x, t, c)

    assert out.shape == (batch_size, seq_len, input_dim)

    out.sum().backward()

    grad_params = [
        p for p in model.parameters() if p.requires_grad and p.grad is not None
    ]
    assert grad_params
