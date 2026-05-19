"""Reproduce the shim's exact dk-garbage repro standalone."""
import os
import sys
import torch

os.environ["TILELANG_ASCEND_MODE"] = "Developer"
sys.path.insert(0, "/home/z00637938/workspace/tilelang-mlir-ascend")
from examples.deepseek_v4.example_lighting_indexer_bwd_kernel import lighting_indexer_bwd

torch.npu.set_device(0)
S, SKV, H, D = 4, 16, 8, 32  # match shim test
topk_K = 4
BI = 4

# Compile kernel once
kernel = lighting_indexer_bwd(
    seq_len=1, seq_len_kv=SKV, heads=H, index_dim=D, topk=topk_K, block_I=BI
)

# Run twice with SAME inputs — second call should give same outputs as first
torch.manual_seed(0)
q = torch.randn(1, H, D, dtype=torch.bfloat16, device="npu") * 0.1
k = torch.randn(SKV, D, dtype=torch.bfloat16, device="npu") * 0.1
w = torch.randn(1, H, dtype=torch.float32, device="npu") * 0.5
topk_idx = torch.tensor([[7, 1, 5, 3]], dtype=torch.int32, device="npu")  # non-monotonic
grad_scores = torch.randn(1, topk_K, dtype=torch.float32, device="npu") * 0.1

# Call 1
dq1 = torch.zeros_like(q)
dw1 = torch.zeros_like(w)
dk1 = torch.zeros(SKV, D, dtype=torch.float32, device="npu")
kernel(q, k, w, topk_idx, grad_scores, dq1, dw1, dk1)
print(f"Call 1: dk1[0,:4] = {dk1[0,:4].cpu().tolist()}")
print(f"        dk1 max abs: {dk1.abs().max().item():.6f}")

# Call 2 — fresh buffers
dq2 = torch.zeros_like(q)
dw2 = torch.zeros_like(w)
dk2 = torch.zeros(SKV, D, dtype=torch.float32, device="npu")
kernel(q, k, w, topk_idx, grad_scores, dq2, dw2, dk2)
print(f"Call 2: dk2[0,:4] = {dk2[0,:4].cpu().tolist()}")
print(f"        dk2 max abs: {dk2.abs().max().item():.6f}")

# Are they identical?
diff = (dk1.cpu() - dk2.cpu()).abs().max().item()
print(f"Call1 vs Call2 max abs diff: {diff:.6f}")
