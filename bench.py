import torch
from torch.utils.cpp_extension import load

M, N, K = (512, 256, 1024)

my_ext = load(name="my_ext", sources = ["./csrc/torch_interface.cpp",
                                        "./csrc/matmul0.cpp",
                                        ])

if __name__ == "__main__":

    a = torch.randn((M, K), dtype=torch.float32)
    b = torch.randn((K, N), dtype=torch.float32)
    c = torch.zeros((M, N), dtype=torch.float32)

    out = a@b
    my_ext.matmul(a, b, c, 0)

    assert torch.allclose(c, out)
