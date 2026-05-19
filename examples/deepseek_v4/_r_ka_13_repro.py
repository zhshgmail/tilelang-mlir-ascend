# R-KA-13 minimal reproducer: bishengir-compile silently zeros the result of
# T.vsub(big_BHxBS, small_BHx1, big) at certain points in a pipelined kernel.
#
# Setup: a kernel that does
#   1) gemm A @ B^T -> C        (gives C ~ 3.2)
#   2) gemm dO @ B^T -> D       (gives D ~ 3.2)
#   3) vsub(C, c1, C)           (works — produces C - c1, magnitudes match)
#   4) vsub(D, d1, D)           (broken — produces 0)
#
# Both vsubs operate on identically-shaped buffers ([16,8] - [16,1]). The first
# one works. The second one zeros out the result.
#
# Used to bisect P1.4 sparse_mla_bwd dQ=0 issue in T33.
import os
import torch
import tilelang
import tilelang.language as T

os.environ["TILELANG_ASCEND_MODE"] = "Developer"


@tilelang.jit(
    target="npuir",
    pass_configs={"npuir.enable_auto_multi_buffer": False},
)
def repro_kernel():
    BH = 16
    BS = 8
    D = 64
    dtype = "float16"
    accum = "float32"

    @T.prim_func
    def main(
        A: T.Tensor([BH, D], dtype),
        B: T.Tensor([BS, D], dtype),
        c1_bias: T.Tensor([BH, 1], accum),
        d1_bias: T.Tensor([BH, 1], accum),
        C_out: T.Tensor([BH, BS], accum),
        D_out: T.Tensor([BH, BS], accum),
    ):
        with T.Kernel(1, is_npu=True) as (_, _unused):
            A_shared = T.alloc_shared([BH, D], dtype)
            B_shared = T.alloc_shared([BS, D], dtype)
            C_frag = T.alloc_fragment([BH, BS], accum)
            C_cast = T.alloc_fragment([BH, BS], dtype)
            D_frag = T.alloc_fragment([BH, BS], accum)
            c1_frag = T.alloc_fragment([BH, 1], accum)
            d1_frag = T.alloc_fragment([BH, 1], accum)
            c1_shared = T.alloc_shared([BH, 1], accum)
            d1_shared = T.alloc_shared([BH, 1], accum)
            C_shared = T.alloc_shared([BH, BS], accum)
            D_shared = T.alloc_shared([BH, BS], accum)
            C_cast_shared = T.alloc_shared([BH, BS], dtype)

            # Load all inputs
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.copy(c1_bias, c1_shared)
            T.copy(d1_bias, d1_shared)
            T.copy(c1_shared, c1_frag)
            T.copy(d1_shared, d1_frag)

            # Add the pipelined loop wrapper matching real P1.4 sparse_mla_bwd
            for k in T.Pipelined(1, num_stages=1):
                # Step 1: C = A @ B^T  (non-zero)
                T.gemm(A_shared, B_shared, C_frag, initC=True, b_transpose=True)
                # Step 2: vsub(C, c1) -- "lse-position" vsub in real kernel
                T.vsub(C_frag, c1_frag, C_frag)
                # Real kernel adds vexp + vcast here before second gemm
                T.vexp(C_frag, C_frag)
                T.vcast(C_frag, C_cast, round_mode="rint")
                T.copy(C_cast, C_cast_shared)
                T.copy(C_frag, C_shared)
                T.copy(C_shared, C_out)

                # Step 3: D = A @ B^T  (non-zero, same gemm pattern, after vexp)
                T.gemm(A_shared, B_shared, D_frag, initC=True, b_transpose=True)
                # Step 4: vsub(D, d1) -- "delta-position" vsub in real kernel
                T.vsub(D_frag, d1_frag, D_frag)
                T.copy(D_frag, D_shared)
                T.copy(D_shared, D_out)

    return main


def main():
    torch.npu.set_device(0)
    BH, BS, D = 16, 8, 64
    torch.manual_seed(0)
    A = torch.randn(BH, D, dtype=torch.float16, device="npu") * 0.5
    B = torch.randn(BS, D, dtype=torch.float16, device="npu") * 0.5
    c1 = torch.full((BH, 1), 0.3, dtype=torch.float32, device="npu")
    d1 = torch.full((BH, 1), 0.3, dtype=torch.float32, device="npu")

    print("compile repro ...")
    k = repro_kernel()
    print("compile OK; running ...")
    C_out = torch.zeros(BH, BS, dtype=torch.float32, device="npu")
    D_out = torch.zeros(BH, BS, dtype=torch.float32, device="npu")
    k(A, B, c1, d1, C_out, D_out)

    print(f"C_out[0,:4] = {C_out[0,:4].cpu().tolist()}")
    print(f"D_out[0,:4] = {D_out[0,:4].cpu().tolist()}")

    # Reference
    AB = (A.float() @ B.float().T).cpu()
    ref_C = (AB - 0.3).exp()
    ref_D = AB - 0.3
    print(f"ref_C[0,:4] = {ref_C[0,:4].tolist()}")
    print(f"ref_D[0,:4] = {ref_D[0,:4].tolist()}")

    C_finite = torch.isfinite(C_out).all().item()
    D_finite = torch.isfinite(D_out).all().item()
    C_nonzero = (C_out.abs() > 1e-6).any().item()
    D_nonzero = (D_out.abs() > 1e-6).any().item()
    print(f"C_out: finite={C_finite}  non-zero={C_nonzero}")
    print(f"D_out: finite={D_finite}  non-zero={D_nonzero}")
    if D_nonzero and not C_nonzero:
        print("\nUNEXPECTED: first vsub broken, second works (opposite of P1.4)")
    elif C_nonzero and not D_nonzero:
        print("\n!!! R-KA-13 REPRODUCED: second vsub zeros out the result")
    elif C_nonzero and D_nonzero:
        print("\nBoth vsubs work in this minimal kernel — bug needs more pipeline context")
    else:
        print("\nNeither works — different bug class")


if __name__ == "__main__":
    main()
