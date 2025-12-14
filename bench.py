import torch
from torch.utils.cpp_extension import load
import timeit
import os
import argparse

VARIANTS = 10
NUM_BENCH = 15

# Use PyTorch's OpenMP library to avoid conflicts
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')

my_ext = load(name="my_ext",
              sources = ["./csrc/torch_interface.cpp"] +
              [f"./csrc/matmul{i}.cpp" for i in range(VARIANTS)],
              extra_cflags=["-O3", "-Xclang", "-fopenmp", f"-I{torch_lib_path}"],
              extra_ldflags=[f"-L{torch_lib_path}", "-lomp"],
              verbose=True
              )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MOE Benchmark Script')
    parser.add_argument('--sizes',
                        type=int,
                        nargs='+',
                        default=[32, 128, 256, 512],
                        help='Batch sizes to benchmark')
    args = parser.parse_args()
    sizes = args.sizes

    for s in sizes:
        M, N, K = (s,s,s)
        a = torch.randn((M, K), dtype=torch.float32)
        b = torch.randn((K, N), dtype=torch.float32)

        out = a@b
        torch_time = timeit.timeit(lambda: a@b, number=NUM_BENCH) / NUM_BENCH
        print("\nsize =", s)
        for variant in range(VARIANTS):
            c = torch.zeros((M, N), dtype=torch.float32)
            my_ext.matmul(a, b, c, variant)

            if not torch.allclose(c, out, atol=1e-4, rtol=1e-6):
                print(c)
                print(out)

            assert torch.allclose(c, out, atol=1e-4, rtol=1e-6)

            flops = 2*M*K*N
            matmul_time = timeit.timeit(lambda: my_ext.matmul(a, b, c, variant), number=NUM_BENCH) / NUM_BENCH
            print(f"torch_time={torch_time*1000:.4f}ms({flops/(torch_time*1e9):.2f}GFLOP/s), matmul_time={matmul_time*1000:.4f}ms({flops/(matmul_time*1e9):.2f}GFLOP/s), {variant=} performance = {(torch_time/matmul_time)*100:.2f}%")
