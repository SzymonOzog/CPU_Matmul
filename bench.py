import torch
from torch.utils.cpp_extension import load
import timeit

M, N, K = (512, 256, 1024)
M, N, K = (1024, 1024, 1024)
VARIANTS = 2

my_ext = load(name="my_ext", sources = ["./csrc/torch_interface.cpp",
                                        "./csrc/matmul0.cpp",
                                        "./csrc/matmul1.cpp",
                                        ])

if __name__ == "__main__":

    a = torch.randn((M, K), dtype=torch.float32)
    b = torch.randn((K, N), dtype=torch.float32)

    out = a@b
    for variant in range(VARIANTS):
        c = torch.zeros((M, N), dtype=torch.float32)
        my_ext.matmul(a, b, c, variant)

        assert torch.allclose(c, out)

        torch_time = timeit.timeit(lambda: a@b, number=10)
        matmul_time = timeit.timeit(lambda: my_ext.matmul(a, b, c, variant), number=10)
        print(f"{torch_time=}, {matmul_time=}, {variant=} performance = {(torch_time/matmul_time)*100:.2f}%")
