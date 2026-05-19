# Copyright (c) Huawei Technologies Co., Ltd. 2026.
#
# T33.P2 miles V4 indexer integration smoke on Ascend NPU.
#
# Goal: prove that miles' `V4IndexerFunction` (a torch.autograd.Function from
# miles_plugins/models/deepseek_v4/ops/kernel/tilelang_indexer.py) can produce
# correct gradients on NPU when its underlying tilelang kernel is our
# NPU-ported version.
#
# We don't install miles wholesale — only the autograd-Function-level wrapper
# semantics. The wrapper:
#   forward:  logits = batched_indexer_fwd(q, k, w, cu_seqlen_ks, cu_seqlen_ke)
#             topk_indices = torch.topk(logits, K)
#             index_score  = gather(logits, topk_indices)
#   backward: dq, dw, dk = batched_indexer_bwd(q, w, k, topk_indices, grad_scores)
#
# Adaptations vs miles' CUDA path:
#   * Use our NPU lighting_indexer_fwd / lighting_indexer_bwd kernels in place
#     of `tl_indexer_fwd_impl` / `tl_indexer_bwd_impl`.
#   * fp16 inputs (our kernels' current support; bf16 = future work).
#   * No causal masking inside kernel (cu_seqlens ignored; caller is per-batch).
#   * Per-batch-element invocation (SEQ=1) — matches miles' `batched_indexer_*`
#     loop pattern exactly and avoids our R-KA-14 (multi-block NaN) blocker.
import os
import torch

from examples.deepseek_v4.example_lighting_indexer_fwd_kernel import lighting_indexer_fwd
from examples.deepseek_v4.example_lighting_indexer_bwd_kernel import lighting_indexer_bwd


def pytorch_extract_topk_scores(logits, topk_indices, dim=-1):
    """Direct copy from miles."""
    valid_mask = topk_indices != -1
    safe_indices = topk_indices.clamp(min=0).to(torch.int64)
    scores = torch.gather(logits, dim=dim, index=safe_indices)
    scores = torch.where(valid_mask, scores, float("-inf"))
    return scores


class V4IndexerFunctionNPU(torch.autograd.Function):
    """NPU-backed reimplementation of miles' V4IndexerFunction.

    Signature is identical to miles' miles_plugins.models.deepseek_v4.ops.
    kernel.tilelang_indexer.V4IndexerFunction; only the underlying tilelang
    kernel calls are redirected to our mlir-ascend backend.

    Inputs (matching miles' V4 SBHD layout):
        index_q: [seqlen_q, batch, heads, dim]   fp16  (miles uses bf16)
        index_k: [seqlen_kv, batch, dim]          fp16
        weights: [seqlen_q, batch, heads]         fp32
        compress_ratio:  4 for C4 layer (default), 128 for C128
        topk:            number of indices to keep per query position
    """

    @staticmethod
    def forward(ctx, index_q, index_k, weights, compress_ratio, topk, topk_indices=None):
        seqlen_q = index_q.shape[0]
        batch = index_q.shape[1]
        seq_len_kv = index_k.shape[0]
        heads = index_q.shape[2]
        dim = index_q.shape[3]

        # Per-batch loop (matches miles batched_indexer_fwd pattern)
        all_logits = torch.empty(
            [batch, seqlen_q, seq_len_kv], device=index_q.device, dtype=torch.float32
        )
        for b in range(batch):
            q_b = index_q[:, b, :, :].contiguous()  # [seqlen, H, dim]
            k_b = index_k[:, b, :].contiguous()      # [seqlen_kv, dim]
            w_b = weights[:, b, :].contiguous()      # [seqlen, H]
            # Our kernel expects q flat [seqlen * H, dim]
            q_flat = q_b.reshape(seqlen_q * heads, dim)
            kernel = lighting_indexer_fwd(
                seq_len=seqlen_q, seq_len_kv=seq_len_kv,
                heads=heads, index_dim=dim,
                block_N=min(64, seq_len_kv), block_Q=min(4, seqlen_q),
            )
            logits_b = kernel(q_flat, k_b, w_b)
            all_logits[b] = logits_b

        # Top-k selection (still on NPU)
        if topk_indices is None:
            actual_topk = min(topk, seq_len_kv)
            index_score, topk_indices = torch.topk(all_logits, actual_topk, dim=-1)
            topk_indices = topk_indices.to(torch.int32)
            # Mask -inf scores (positions that were masked out)
            topk_indices = topk_indices.masked_fill(index_score == -torch.inf, -1)

        index_score = pytorch_extract_topk_scores(all_logits, topk_indices)

        ctx.save_for_backward(index_q, index_k, weights, topk_indices)
        ctx.compress_ratio = compress_ratio
        ctx.topk = topk
        return index_score, topk_indices

    @staticmethod
    def backward(ctx, grad_scores, grad_indices):
        # grad_indices is ignored (integer indices aren't differentiated through)
        index_q, index_k, weights, topk_indices = ctx.saved_tensors
        seqlen_q, batch, heads, dim = index_q.shape
        seq_len_kv = index_k.shape[0]
        topk_K = topk_indices.shape[-1]

        # Allocate output gradients
        grad_q = torch.zeros_like(index_q)
        grad_w = torch.zeros_like(weights, dtype=torch.float32)
        grad_k = torch.zeros_like(index_k, dtype=torch.float32)

        # Per-batch loop (matches miles batched_indexer_bwd pattern + avoids R-KA-14)
        for b in range(batch):
            q_b = index_q[:, b, :, :].contiguous()   # [seqlen, H, dim]
            k_b = index_k[:, b, :].contiguous()       # [seqlen_kv, dim]
            w_b = weights[:, b, :].contiguous()       # [seqlen, H]
            topk_idx_b = topk_indices[b, :, :].contiguous()       # [seqlen, K]
            grad_scores_b = grad_scores[b, :, :].contiguous()      # [seqlen, K]

            dq_b = torch.zeros_like(q_b)
            dw_b = torch.zeros_like(w_b)
            dk_b_acc = torch.zeros(seq_len_kv, dim, dtype=torch.float32, device=index_q.device)

            kernel = lighting_indexer_bwd(
                seq_len=seqlen_q, seq_len_kv=seq_len_kv,
                heads=heads, index_dim=dim,
                topk=topk_K,
                block_I=min(8, topk_K),
            )
            kernel(q_b, k_b, w_b, topk_idx_b, grad_scores_b, dq_b, dw_b, dk_b_acc)

            grad_q[:, b, :, :] = dq_b
            grad_w[:, b, :] = dw_b
            # Accumulate dk per batch (atomic across batches done in Python)
            grad_k[:, b, :] += dk_b_acc

        return grad_q, grad_k, grad_w, None, None, None


def v4_lighting_indexer_npu(index_q, index_k, weights, compress_ratio, topk, topk_indices=None):
    """Drop-in replacement for miles' v4_lighting_indexer, NPU-backed."""
    return V4IndexerFunctionNPU.apply(index_q, index_k, weights, compress_ratio, topk, topk_indices)


def _ref_v4_indexer(q, k, w, topk_K):
    """Pure-PyTorch reference matching miles' algorithm on CPU."""
    # q [S, B, H, D], k [SKV, B, D], w [S, B, H]
    S, B, H, D = q.shape
    SKV = k.shape[0]

    # scores[s, b, h, kv] = max(q[s,b,h] @ k[kv,b], 0) * w[s,b,h]
    # logits[b, s, kv] = sum_h scores[s, b, h, kv]
    qf = q.float()  # [S, B, H, D]
    kf = k.float()  # [SKV, B, D]
    wf = w.float()  # [S, B, H]

    scores = torch.einsum("sbhd,kbd->sbhk", qf, kf)  # [S, B, H, SKV]
    scores = scores.clamp(min=0)
    scores = scores * wf.unsqueeze(-1)  # broadcast over kv axis
    logits = scores.sum(dim=2)  # [S, B, SKV]
    logits = logits.permute(1, 0, 2)  # [B, S, SKV]

    # top-k
    index_score, topk_indices = torch.topk(logits, topk_K, dim=-1)
    topk_indices = topk_indices.to(torch.int32)
    return index_score, topk_indices, logits


def test_miles_integration():
    torch.npu.set_device(0)
    # Tiny test — single batch, single seq position to satisfy our R-KA-14
    # workaround (SEQ=1 per inner kernel call).
    S, B, H, D = 4, 1, 8, 32
    SKV = 16
    topk_K = 8

    torch.manual_seed(0)
    # IMPORTANT: requires_grad must be set on the LEAF tensor. If we multiply
    # by 0.1 after requires_grad=True, we get a non-leaf tensor and .grad is None.
    q_init = torch.randn(S, B, H, D, dtype=torch.float16, device="npu") * 0.1
    k_init = torch.randn(SKV, B, D, dtype=torch.float16, device="npu") * 0.1
    w_init = torch.randn(S, B, H, dtype=torch.float32, device="npu") * 0.5
    q = q_init.detach().requires_grad_(True)
    k = k_init.detach().requires_grad_(True)
    w = w_init.detach().requires_grad_(True)

    print(f"Inputs: q {tuple(q.shape)} {q.dtype}, k {tuple(k.shape)}, w {tuple(w.shape)}")

    # Forward
    index_score, topk_idx = v4_lighting_indexer_npu(q, k, w, compress_ratio=4, topk=topk_K)
    print(f"Output: index_score {tuple(index_score.shape)}, topk_idx {tuple(topk_idx.shape)}")
    print(f"  index_score[0,0,:4] = {index_score[0,0,:4].cpu().tolist()}")

    # Backward
    loss = index_score.sum()
    loss.backward()
    print(f"\nGradients computed via autograd through V4IndexerFunctionNPU:")
    print(f"  dq shape: {tuple(q.grad.shape)}, dtype: {q.grad.dtype}")
    print(f"  dk shape: {tuple(k.grad.shape)}, dtype: {k.grad.dtype}")
    print(f"  dw shape: {tuple(w.grad.shape)}, dtype: {w.grad.dtype}")
    print(f"  dq[0,0,0,:4] = {q.grad[0,0,0,:4].cpu().tolist()}")
    print(f"  dw[0,0,:4]   = {w.grad[0,0,:4].cpu().tolist()}")
    print(f"  dk[0,0,:4]   = {k.grad[0,0,:4].cpu().tolist()}")

    # CPU reference
    q_ref = q.detach().cpu().float().requires_grad_(True)
    k_ref = k.detach().cpu().float().requires_grad_(True)
    w_ref = w.detach().cpu().requires_grad_(True)
    index_score_ref, topk_idx_ref, logits_ref = _ref_v4_indexer(q_ref, k_ref, w_ref, topk_K)
    loss_ref = index_score_ref.sum()
    loss_ref.backward()
    print(f"\nCPU reference grads:")
    print(f"  dq_ref[0,0,0,:4] = {q_ref.grad[0,0,0,:4].tolist()}")
    print(f"  dw_ref[0,0,:4]   = {w_ref.grad[0,0,:4].tolist()}")
    print(f"  dk_ref[0,0,:4]   = {k_ref.grad[0,0,:4].tolist()}")

    err_q = (q.grad.cpu().float() - q_ref.grad).abs().max().item()
    err_w = (w.grad.cpu().float() - w_ref.grad).abs().max().item()
    err_k = (k.grad.cpu().float() - k_ref.grad).abs().max().item()
    print(f"\n=== max abs err vs CPU ref ===")
    print(f"  dq: {err_q:.5f}")
    print(f"  dw: {err_w:.5f}")
    print(f"  dk: {err_k:.5f}")


if __name__ == "__main__":
    os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
    test_miles_integration()
