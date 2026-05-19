"""Try various q tensors to find the trigger pattern."""
import os
import sys
import torch

os.environ["TILELANG_ASCEND_MODE"] = "Developer"
sys.path.insert(0, "/home/z00637938/workspace/tilelang-mlir-ascend")
from examples.deepseek_v4.example_lighting_indexer_bwd_kernel import lighting_indexer_bwd

torch.npu.set_device(0)
SKV, H, D = 16, 8, 32
K = 4
BI = 4

torch.manual_seed(0)
k_b = torch.randn(SKV, D, dtype=torch.bfloat16, device="npu") * 0.1
w_b = torch.randn(1, H, dtype=torch.float32, device="npu") * 0.5
idx = torch.arange(K, dtype=torch.int32, device="npu").unsqueeze(0)
gs = torch.randn(1, K, dtype=torch.float32, device="npu") * 0.1

kernel = lighting_indexer_bwd(seq_len=1, seq_len_kv=SKV, heads=H, index_dim=D, topk=K, block_I=BI)


def _run(q_b, label):
    dq = torch.zeros_like(q_b)
    dw = torch.zeros_like(w_b)
    dk = torch.zeros(SKV, D, dtype=torch.float32, device="npu")
    kernel(q_b, k_b, w_b, idx, gs, dq, dw, dk)
    m = dk.abs().max().item()
    print(f"  {label}: max abs = {m:.6f}  {'EXPLODES' if m > 1e10 else 'ok'}")
    return m


# 1) All zeros
q = torch.zeros(1, H, D, dtype=torch.bfloat16, device="npu")
_run(q, "all zeros")

# 2) All ones
q = torch.ones(1, H, D, dtype=torch.bfloat16, device="npu") * 0.1
_run(q, "all 0.1")

# 3) Negative-only
q = -torch.ones(1, H, D, dtype=torch.bfloat16, device="npu") * 0.1
_run(q, "all -0.1")

# 4) Random seed 1
torch.manual_seed(1)
q = torch.randn(1, H, D, dtype=torch.bfloat16, device="npu") * 0.1
_run(q, "randn seed 1")

# 5) Random seed 2
torch.manual_seed(2)
q = torch.randn(1, H, D, dtype=torch.bfloat16, device="npu") * 0.1
_run(q, "randn seed 2")

# 6) Random seed 3
torch.manual_seed(3)
q = torch.randn(1, H, D, dtype=torch.bfloat16, device="npu") * 0.1
_run(q, "randn seed 3")

# 7) Different magnitudes
torch.manual_seed(0)
for scale in [1.0, 10.0, 100.0, 0.01]:
    q = torch.randn(1, H, D, dtype=torch.bfloat16, device="npu") * scale
    _run(q, f"randn seed 0 scale {scale}")
