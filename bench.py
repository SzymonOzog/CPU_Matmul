import torch
from torch.utils.cpp_extension import load
import timeit

VARIANTS = 8
NUM_BENCH = 5

my_ext = load(name="my_ext", 
              sources = ["./csrc/torch_interface.cpp"] +
              [f"./csrc/matmul{i}.cpp" for i in range(VARIANTS)],
              extra_cflags=["-O3"],
              verbose=True
              )

if __name__ == "__main__":
    sizes = [32, 128, 256, 512]
    for s in sizes:
        M, N, K = (s,s,s)
        a = torch.randn((M, K), dtype=torch.float32)
        b = torch.randn((K, N), dtype=torch.float32)

        out = a@b
        torch_time = timeit.timeit(lambda: a@b, number=NUM_BENCH)
        print("\nsize =", s)
        for variant in range(VARIANTS):
            c = torch.zeros((M, N), dtype=torch.float32)
            my_ext.matmul(a, b, c, variant)

            assert torch.allclose(c, out, atol=1e-4, rtol=1e-6)

            matmul_time = timeit.timeit(lambda: my_ext.matmul(a, b, c, variant), number=NUM_BENCH)
            print(f"{torch_time=}, {matmul_time=}, {variant=} performance = {(torch_time/matmul_time)*100:.2f}%")
