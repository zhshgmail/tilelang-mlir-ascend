"""Reproduce the shim's per-seq-position loop exactly."""
import os
import sys
import torch

os.environ["TILELANG_ASCEND_MODE"] = "Developer"
sys.path.insert(0, "/home/z00637938/workspace/tilelang-mlir-ascend")
from examples.deepseek_v4.example_lighting_indexer_bwd_kernel import lighting_indexer_bwd

torch.npu.set_device(0)
S, SKV, H, D = 4, 16, 8, 32
topk_K = 4
BI = 4

# Build inputs LIKE the shim — sliced from full tensors
torch.manual_seed(0)
q_full = torch.randn(S, 1, H, D, dtype=torch.bfloat16, device="npu") * 0.1
k_full = torch.randn(SKV, 1, D, dtype=torch.bfloat16, device="npu") * 0.1
w_full = torch.randn(S, 1, H, dtype=torch.float32, device="npu") * 0.5
topk_idx_full = torch.zeros(1, S, topk_K, dtype=torch.int32, device="npu")
for s in range(S):
    topk_idx_full[0, s] = torch.arange(topk_K, dtype=torch.int32)
grad_scores_full = torch.randn(1, S, topk_K, dtype=torch.float32, device="npu") * 0.1

# Compile once
kernel = lighting_indexer_bwd(
    seq_len=1, seq_len_kv=SKV, heads=H, index_dim=D, topk=topk_K, block_I=BI
)

# Per-batch (B=1) per-seq loop matching shim
b = 0
q_b = q_full[:, b, :, :].contiguous()  # [S, H, D]
k_b = k_full[:, b, :].contiguous()      # [SKV, D]
w_b = w_full[:, b, :].contiguous()      # [S, H]
topk_idx_b = topk_idx_full[b, :, :].contiguous()
grad_scores_b = grad_scores_full[b, :, :].contiguous()

# Call kernel 6 times with the SAME inputs (s=0 fixed). See if 4th+ explodes.
for s in range(6):
    q_bs = q_b[0:1].contiguous()
    w_bs = w_b[0:1].contiguous()
    topk_idx_bs = topk_idx_b[0:1].contiguous()
    grad_scores_bs = grad_scores_b[0:1].contiguous()
    dq_bs = torch.zeros_like(q_bs)
    dw_bs = torch.zeros_like(w_bs)
    dk_call = torch.zeros(SKV, D, dtype=torch.float32, device="npu")
    kernel(q_bs, k_b, w_bs, topk_idx_bs, grad_scores_bs, dq_bs, dw_bs, dk_call)
    print(f"  call={s}: dk_call max abs = {dk_call.abs().max().item():.6f}")
dk_acc = torch.zeros(SKV, D, dtype=torch.float32, device="npu")

print(f"\nFinal: dk_acc max abs = {dk_acc.abs().max().item():.6f}")
print(f"       dk_acc[0,:4] = {dk_acc[0,:4].cpu().tolist()}")
