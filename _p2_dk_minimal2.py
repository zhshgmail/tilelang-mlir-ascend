"""Dump s=0 vs s=3 input values to find what differs structurally."""
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

torch.manual_seed(0)
q_full = torch.randn(S, 1, H, D, dtype=torch.bfloat16, device="npu") * 0.1
k_full = torch.randn(SKV, 1, D, dtype=torch.bfloat16, device="npu") * 0.1
w_full = torch.randn(S, 1, H, dtype=torch.float32, device="npu") * 0.5
topk_idx_full = torch.zeros(1, S, topk_K, dtype=torch.int32, device="npu")
for s in range(S):
    topk_idx_full[0, s] = torch.arange(topk_K, dtype=torch.int32)
grad_scores_full = torch.randn(1, S, topk_K, dtype=torch.float32, device="npu") * 0.1


def _run_once(kernel, k_b, q_b, w_b, topk_idx_b, grad_scores_b):
    dq = torch.zeros_like(q_b)
    dw = torch.zeros_like(w_b)
    dk = torch.zeros(SKV, D, dtype=torch.float32, device="npu")
    kernel(q_b, k_b, w_b, topk_idx_b, grad_scores_b, dq, dw, dk)
    return dk.abs().max().item()


k_b = k_full[:, 0, :].contiguous()
kernel = lighting_indexer_bwd(seq_len=1, seq_len_kv=SKV, heads=H, index_dim=D, topk=topk_K, block_I=BI)


def get_s(s):
    q_b = q_full[:, 0, :, :].contiguous()[s : s + 1].contiguous()
    w_b = w_full[:, 0, :].contiguous()[s : s + 1].contiguous()
    topk_idx_b = topk_idx_full[0][s : s + 1].contiguous()
    grad_scores_b = grad_scores_full[0][s : s + 1].contiguous()
    return q_b, w_b, topk_idx_b, grad_scores_b


print("=== Investigate by replacing each input from s=3 with s=0's value ===\n")

# Baseline: s=3 explodes
q3, w3, idx3, g3 = get_s(3)
print(f"baseline s=3 all: max abs = {_run_once(kernel, k_b, q3, w3, idx3, g3):.6f}")

q0, w0, idx0, g0 = get_s(0)
print(f"baseline s=0 all: max abs = {_run_once(kernel, k_b, q0, w0, idx0, g0):.6f}")

# Replace one input at a time
print(f"\ns=3 except q=s0:   max abs = {_run_once(kernel, k_b, q0, w3, idx3, g3):.6f}")
print(f"s=3 except w=s0:   max abs = {_run_once(kernel, k_b, q3, w0, idx3, g3):.6f}")
print(f"s=3 except idx=s0: max abs = {_run_once(kernel, k_b, q3, w3, idx0, g3):.6f}")
print(f"s=3 except g=s0:   max abs = {_run_once(kernel, k_b, q3, w3, idx3, g0):.6f}")
