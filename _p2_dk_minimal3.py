"""Dump q[s=3] vs q[s=0] and find what's structurally different."""
import os
import sys
import torch

os.environ["TILELANG_ASCEND_MODE"] = "Developer"
sys.path.insert(0, "/home/z00637938/workspace/tilelang-mlir-ascend")

torch.npu.set_device(0)
S, SKV, H, D = 4, 16, 8, 32

torch.manual_seed(0)
q_full = torch.randn(S, 1, H, D, dtype=torch.bfloat16, device="npu") * 0.1

q0 = q_full[0, 0]  # [H, D]
q3 = q_full[3, 0]  # [H, D]

print(f"q0 min: {q0.min().item():.6f}, max: {q0.max().item():.6f}, abs mean: {q0.abs().mean().item():.6f}")
print(f"q3 min: {q3.min().item():.6f}, max: {q3.max().item():.6f}, abs mean: {q3.abs().mean().item():.6f}")
print(f"q0 nan: {torch.isnan(q0).sum().item()}, inf: {torch.isinf(q0).sum().item()}")
print(f"q3 nan: {torch.isnan(q3).sum().item()}, inf: {torch.isinf(q3).sum().item()}")
print(f"q3 first row: {q3[0, :8].cpu().tolist()}")
print(f"q3 elements close to zero (< 1e-3): {(q3.abs() < 1e-3).sum().item()} / {q3.numel()}")
print(f"q3 contains denormals: {((q3.abs() < 1.18e-38) & (q3.abs() > 0)).sum().item()}")

# What if it's bf16 precision boundary?
print(f"\nq3 unique bf16 representable values check:")
q3_fp32 = q3.float()
# Spot any odd patterns
print(f"q3 has any negative element close to -0?  min abs of negatives: {q3[q3<0].abs().min().item():.6e}")
