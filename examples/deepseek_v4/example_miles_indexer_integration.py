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
            # R-KA-14 workaround for fwd path too: invoke per seq position.
            # Our kernel's fwd has T.Kernel(NQ); with seqlen=1 this is 1 block.
            kernel_s1 = lighting_indexer_fwd(
                seq_len=1, seq_len_kv=seq_len_kv,
                heads=heads, index_dim=dim,
                block_N=min(64, seq_len_kv), block_Q=1,
            )
            for s in range(seqlen_q):
                q_bs = q_b[s : s + 1].contiguous()         # [1, H, D]
                w_bs = w_b[s : s + 1].contiguous()         # [1, H]
                q_flat = q_bs.reshape(heads, dim)
                logits_bs = kernel_s1(q_flat, k_b, w_bs)   # [1, SKV]
                all_logits[b, s] = logits_bs[0]

        # Apply V4 causal mask via cu_seqlens (matches miles' _make_causal_cu_seqlens):
        # For query position p, valid compressed kv range = [0, (p+1) // compress_ratio).
        positions = torch.arange(seqlen_q, device=index_q.device, dtype=torch.int32)
        valid_end = (positions + 1) // compress_ratio  # [seqlen_q]
        kv_positions = torch.arange(seq_len_kv, device=index_q.device, dtype=torch.int32)
        # mask[s, kv] = True if kv < valid_end[s]
        mask = kv_positions.unsqueeze(0) < valid_end.unsqueeze(1)  # [seqlen_q, seq_len_kv]
        all_logits = all_logits.masked_fill(~mask.unsqueeze(0), float("-inf"))

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
            # Sanitize: kernel reads IndexK[cur_idx]; cur_idx=-1 (masked-out
            # position from V4 causal logic) would produce NaN via OOB read.
            # Replace -1 with 0 (safe) and zero its grad_score so the atomic
            # contribution at dKV[0,...] is no-op.
            invalid_mask = topk_idx_b == -1
            topk_idx_b = torch.where(invalid_mask, torch.zeros_like(topk_idx_b), topk_idx_b)
            grad_scores_b = torch.where(invalid_mask, torch.zeros_like(grad_scores_b), grad_scores_b)

            dq_b = torch.zeros_like(q_b)
            dw_b = torch.zeros_like(w_b)
            dk_b_acc = torch.zeros(seq_len_kv, dim, dtype=torch.float32, device=index_q.device)

            # R-KA-14 workaround: invoke kernel ONE seq position at a time.
            # Use a per-call dk-shadow buffer and accumulate in Python to
            # avoid any kernel-side accumulator-state leakage.
            kernel_s1 = lighting_indexer_bwd(
                seq_len=1, seq_len_kv=seq_len_kv,
                heads=heads, index_dim=dim,
                topk=topk_K,
                block_I=min(8, topk_K),
            )
            for s in range(seqlen_q):
                q_bs = q_b[s : s + 1].contiguous()        # [1, H, D]
                w_bs = w_b[s : s + 1].contiguous()        # [1, H]
                topk_idx_bs = topk_idx_b[s : s + 1].contiguous()  # [1, K]
                grad_scores_bs = grad_scores_b[s : s + 1].contiguous()  # [1, K]
                dq_bs = torch.zeros_like(q_bs)
                dw_bs = torch.zeros_like(w_bs)
                # Per-call dk buffer (DON'T share — kernel may treat input dk
                # as having undefined initial state per-call on NPU)
                dk_call = torch.zeros(seq_len_kv, dim, dtype=torch.float32, device=index_q.device)
                kernel_s1(q_bs, k_b, w_bs, topk_idx_bs, grad_scores_bs, dq_bs, dw_bs, dk_call)
                dq_b[s : s + 1] = dq_bs
                dw_b[s : s + 1] = dw_bs
                dk_b_acc += dk_call  # accumulate in Python (safe)

            grad_q[:, b, :, :] = dq_b
            grad_w[:, b, :] = dw_b
            # Accumulate dk per batch (atomic across batches done in Python)
            grad_k[:, b, :] += dk_b_acc

        return grad_q, grad_k, grad_w, None, None, None


def v4_lighting_indexer_npu(index_q, index_k, weights, compress_ratio, topk, topk_indices=None):
    """Drop-in replacement for miles' v4_lighting_indexer, NPU-backed."""
    return V4IndexerFunctionNPU.apply(index_q, index_k, weights, compress_ratio, topk, topk_indices)


def _ref_v4_indexer(q, k, w, topk_K, compress_ratio=4):
    """Pure-PyTorch reference matching miles' algorithm on CPU."""
    # q [S, B, H, D], k [SKV, B, D], w [S, B, H]
    S, B, H, D = q.shape
    SKV = k.shape[0]

    qf = q.float()
    kf = k.float()
    wf = w.float()

    scores = torch.einsum("sbhd,kbd->sbhk", qf, kf)
    scores = scores.clamp(min=0)
    scores = scores * wf.unsqueeze(-1)
    logits = scores.sum(dim=2)
    logits = logits.permute(1, 0, 2)  # [B, S, SKV]

    # V4 causal mask via cu_seqlens (matches _make_causal_cu_seqlens)
    positions = torch.arange(S, dtype=torch.int32)
    valid_end = (positions + 1) // compress_ratio
    kv_positions = torch.arange(SKV, dtype=torch.int32)
    mask = kv_positions.unsqueeze(0) < valid_end.unsqueeze(1)  # [S, SKV]
    logits = logits.masked_fill(~mask.unsqueeze(0), float("-inf"))

    # top-k after mask
    index_score, topk_indices = torch.topk(logits, topk_K, dim=-1)
    topk_indices = topk_indices.to(torch.int32)
    return index_score, topk_indices, logits


def test_miles_integration():
    torch.npu.set_device(0)
    # Very small test: just one query that has full kv range
    # compress_ratio=4, query position p: valid_end = (p+1)//4
    # query p=3: valid_end=1, p=7: valid_end=2, ...
    # Use S=8, SKV=2, compress_ratio=4: queries 4..7 have valid_end=1,2,2,2.
    # Force all queries to skip topk altogether — use compress_ratio=1 (no mask)
    S, B, H, D = 4, 1, 8, 32
    SKV = 16
    topk_K = 4

    torch.manual_seed(0)
    # IMPORTANT: requires_grad must be set on the LEAF tensor. If we multiply
    # by 0.1 after requires_grad=True, we get a non-leaf tensor and .grad is None.
    # bf16 matches miles' V4IndexerFunction spec
    q_init = torch.randn(S, B, H, D, dtype=torch.bfloat16, device="npu") * 0.1
    k_init = torch.randn(SKV, B, D, dtype=torch.bfloat16, device="npu") * 0.1
    w_init = torch.randn(S, B, H, dtype=torch.float32, device="npu") * 0.5
    q = q_init.detach().requires_grad_(True)
    k = k_init.detach().requires_grad_(True)
    w = w_init.detach().requires_grad_(True)

    print(f"Inputs: q {tuple(q.shape)} {q.dtype}, k {tuple(k.shape)}, w {tuple(w.shape)}")

    # Forward (use compress_ratio=1 to disable causal mask for this test)
    compress_ratio = 1
    index_score, topk_idx = v4_lighting_indexer_npu(q, k, w, compress_ratio=compress_ratio, topk=topk_K)
    print(f"Output: index_score {tuple(index_score.shape)}, topk_idx {tuple(topk_idx.shape)}")
    print(f"  index_score[0,0,:4] = {index_score[0,0,:4].cpu().tolist()}")

    # Backward — replace -inf with 0 BEFORE loss (0 * grad = 0 has clean autograd)
    valid_score_mask = torch.isfinite(index_score)
    safe_score = torch.where(valid_score_mask, index_score, torch.zeros_like(index_score))
    loss = safe_score.sum()
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
    index_score_ref, topk_idx_ref, logits_ref = _ref_v4_indexer(q_ref, k_ref, w_ref, topk_K, compress_ratio=compress_ratio)
    valid_ref_mask = torch.isfinite(index_score_ref)
    safe_score_ref = torch.where(valid_ref_mask, index_score_ref, torch.zeros_like(index_score_ref))
    loss_ref = safe_score_ref.sum()
    loss_ref.backward()
    print(f"\nCPU reference grads:")
    print(f"  dq_ref[0,0,0,:4] = {q_ref.grad[0,0,0,:4].tolist()}")
    print(f"  dw_ref[0,0,:4]   = {w_ref.grad[0,0,:4].tolist()}")
    print(f"  dk_ref[0,0,:4]   = {k_ref.grad[0,0,:4].tolist()}")

    err_q = (q.grad.cpu().float() - q_ref.grad).abs().max().item()
    err_w = (w.grad.cpu().float() - w_ref.grad).abs().max().item()
    # dk diagnostic
    dk_k = k.grad.cpu().float()
    dk_r = k_ref.grad
    print(f"  k.grad kernel finite count: {torch.isfinite(dk_k).sum().item()}/{dk_k.numel()}")
    print(f"  k.grad kernel max abs (finite): {dk_k[torch.isfinite(dk_k)].abs().max().item():.4f}")
    print(f"  k_ref.grad max abs:             {dk_r.abs().max().item():.4f}")
    print(f"  k.grad kernel min:  {dk_k.min().item()}")
    print(f"  k.grad kernel max:  {dk_k.max().item()}")
    print(f"  k.grad has {torch.isnan(dk_k).sum().item()} nans, {torch.isinf(dk_k).sum().item()} infs")
    err_k_diff = (dk_k - dk_r).abs()
    finite = torch.isfinite(err_k_diff)
    err_k = err_k_diff[finite].max().item() if finite.any() else float('nan')
    print(f"\n=== max abs err vs CPU ref ===")
    print(f"  dq: {err_q:.5f}")
    print(f"  dw: {err_w:.5f}")
    print(f"  dk: {err_k:.5f}")


if __name__ == "__main__":
    os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
    test_miles_integration()
