"""CPU-only smoke tests for hnet_model. Run: `python test_hnet.py`."""
from __future__ import annotations

import torch

from hnet_model import DynamicChunking, HNet, upsample_with_ema


def test_dc_shapes() -> None:
    dc = DynamicChunking(dim=16, target_ratio=1.0 / 6.0)
    x = torch.randn(2, 8, 16)
    compressed, mask, b_hard, p, p_compressed, ratio_loss = dc(x)
    assert compressed.shape[0] == 2 and compressed.shape[2] == 16
    assert mask.shape == compressed.shape[:2] and mask.dtype == torch.bool
    assert b_hard.shape == (2, 8) and p.shape == (2, 8)
    assert p_compressed.shape == compressed.shape[:2]
    assert ratio_loss.dim() == 0
    print(
        f"[dc_shapes] compressed={tuple(compressed.shape)} "
        f"ratio_loss={ratio_loss.item():.4f}"
    )


def test_dc_gradients() -> None:
    dc = DynamicChunking(dim=16, target_ratio=1.0 / 6.0)
    x = torch.randn(2, 8, 16, requires_grad=True)
    compressed, _, _, _, _, ratio_loss = dc(x)
    (compressed.sum() + 0.1 * ratio_loss).backward()
    assert x.grad is not None and x.grad.abs().sum().item() > 0
    assert dc.W_q.weight.grad is not None and dc.W_q.weight.grad.abs().sum().item() > 0
    print("[dc_gradients] gradients flow through routing + downsample")


def test_upsample_chunk_ema_close_to_gather() -> None:
    # With p≈1 at boundaries (clipped to 1-1e-4 internally), EMA at chunk level
    # carries almost all the chunk's own value: z_bar[c] ≈ z[c].
    # 3 chunks: positions [0,1] -> 0, [2,3,4] -> 1, [5,6,7] -> 2
    b_hard = torch.tensor([[1, 0, 1, 0, 0, 1, 0, 0]], dtype=torch.float32)
    p = b_hard.clone()
    p_compressed = torch.tensor([[1.0, 1.0, 1.0]])
    z = torch.tensor([[[1.0], [2.0], [3.0]]])
    out = upsample_with_ema(z, b_hard, p, p_compressed).flatten().tolist()
    expected = [1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0]
    for got, exp in zip(out, expected):
        assert abs(got - exp) < 1e-2, f"got {out}"
    print(f"[upsample_chunk_ema_hard] {[round(o, 3) for o in out]}")


def test_upsample_soft_boundaries_smooth() -> None:
    # Soft chunk-prob blends successive chunks; values are CONSTANT inside a chunk.
    # 2 chunks: [0,1] -> 0, [2,3] -> 1.
    b_hard = torch.tensor([[1, 0, 1, 0]], dtype=torch.float32)
    p = torch.tensor([[1.0, 0.1, 0.5, 0.1]])
    p_compressed = torch.tensor([[1.0, 0.5]])  # probs at boundary positions 0, 2
    z = torch.tensor([[[10.0], [20.0]]])
    out = upsample_with_ema(z, b_hard, p, p_compressed).flatten().tolist()
    # Chunk-level EMA (p clipped to ~1 and 0.5):
    #   z_bar[0] = z[0]                       = 10
    #   z_bar[1] = 0.5*z[1] + 0.5*z_bar[0]    = 15
    # gather to fine: [10, 10, 15, 15]
    # c_ste forward = 1, so output ≈ [10, 10, 15, 15]
    assert abs(out[0] - 10.0) < 1e-2
    assert abs(out[1] - 10.0) < 1e-2
    assert abs(out[2] - 15.0) < 1e-2
    assert abs(out[3] - 15.0) < 1e-2
    print(f"[upsample_soft] chunk-EMA blend: {[round(o, 3) for o in out]}")


def test_hnet_forward_backward() -> None:
    model = HNet(vocab_size=260, d_enc=64, d_main=128, n_enc=2, n_main=2, n_dec=2, n_heads=4)
    byte_ids = torch.randint(0, 260, (2, 32))
    targets = torch.randint(0, 260, (2, 32))
    ar_loss, ratio_loss = model(byte_ids, targets)
    assert ar_loss.dim() == 0 and ratio_loss.dim() == 0
    (ar_loss + 0.03 * ratio_loss).backward()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[hnet] ar={ar_loss.item():.4f} ratio={ratio_loss.item():.4f} params={n_params:,}")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_dc_shapes()
    test_dc_gradients()
    test_upsample_chunk_ema_close_to_gather()
    test_upsample_soft_boundaries_smooth()
    test_hnet_forward_backward()
    print("\nAll tests pass.")
