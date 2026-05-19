"""Isolate the s=3 inputs to a single standalone call. Goal: show whether
the bug is (a) input-driven (specific tensor values trigger garbage) or
(b) stateful (any 4th call after 3 others, regardless of inputs).

Strategy:
  Test A: call kernel with s=3's inputs FIRST (no prior calls). If garbage,
          it's input-driven.
  Test B: call kernel 3 times with s=0's inputs (sane), then 4th call with
          s=3's inputs. If garbage, it's not pure input-driven — needs
          history.
"""
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


def _make_inputs(s):
    q_b = q_full[:, 0, :, :].contiguous()[s : s + 1].contiguous()
    w_b = w_full[:, 0, :].contiguous()[s : s + 1].contiguous()
    topk_idx_b = topk_idx_full[0][s : s + 1].contiguous()
    grad_scores_b = grad_scores_full[0][s : s + 1].contiguous()
    return q_b, w_b, topk_idx_b, grad_scores_b


def _run_once(kernel, k_b, s):
    q_b, w_b, topk_idx_b, grad_scores_b = _make_inputs(s)
    dq = torch.zeros_like(q_b)
    dw = torch.zeros_like(w_b)
    dk = torch.zeros(SKV, D, dtype=torch.float32, device="npu")
    kernel(q_b, k_b, w_b, topk_idx_b, grad_scores_b, dq, dw, dk)
    return dk.abs().max().item()


k_b = k_full[:, 0, :].contiguous()
kernel = lighting_indexer_bwd(seq_len=1, seq_len_kv=SKV, heads=H, index_dim=D, topk=topk_K, block_I=BI)

print("=== Test A: call s=3 inputs FIRST (no prior history) ===")
m_a = _run_once(kernel, k_b, s=3)
print(f"s=3 first-call: max abs = {m_a:.6f}")
if m_a > 1e10:
    print("⇒ INPUT-DRIVEN: s=3 inputs trigger garbage regardless of history")
else:
    print("⇒ STATEFUL: garbage requires prior calls. Continuing to Test B.")
    print()
    print("=== Test B: 3 calls with s=0 inputs, then s=3 ===")
    for i in range(3):
        m = _run_once(kernel, k_b, s=0)
        print(f"  call {i} (s=0): max abs = {m:.6f}")
    m_b = _run_once(kernel, k_b, s=3)
    print(f"  call 3 (s=3): max abs = {m_b:.6f}")
    if m_b > 1e10:
        print("⇒ STATEFUL+INPUT: prior history of distinct inputs primes kernel; specific s=3 inputs explode")
    else:
        print("⇒ Bug not reproduced — needs different sequence")
