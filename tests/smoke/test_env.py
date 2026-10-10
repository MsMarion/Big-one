"""GPU / ROCm sanity check:  python -m tests.smoke.test_env"""
import time

import torch
import torch.nn.functional as F


def main():
    assert torch.cuda.is_available(), "no GPU visible to torch"
    p = torch.cuda.get_device_properties(0)
    print(f"torch {torch.__version__}  hip {torch.version.hip}  device {p.name} "
          f"({getattr(p, 'gcnArchName', '?')}, {p.total_memory / 1e9:.1f} GB)")
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    a @ a
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(20):
        c = a @ a
    torch.cuda.synchronize()
    print(f"bf16 matmul: {20 * 2 * 4096**3 / (time.perf_counter() - t) / 1e12:.1f} TFLOPS")
    q = torch.randn(8, 6, 257, 64, device="cuda", dtype=torch.bfloat16)
    o = F.scaled_dot_product_attention(q, q, q)
    assert torch.isfinite(c).all() and torch.isfinite(o).all()
    print("ENV OK")


if __name__ == "__main__":
    main()
