#include <torch/all.h>
#include <torch/library.h>
#include <torch/extension.h>


#define SIG float*a, float* b, float* c, \
    const int M, const int N, const int K

#define CALL static_cast<float*>(a.data_ptr()), \
            static_cast<float*>(b.data_ptr()), \
            static_cast<float*>(c.data_ptr()), \
            a.size(0), \
            b.size(1), \
            a.size(1)

void matmul0(SIG);

// CUDA implementation
void matmul(
        const torch::Tensor& a,
        const torch::Tensor& b,
        const torch::Tensor& c,
        int variant
) 
{
    switch (variant)
    {
        case 0:
            matmul0(CALL);
            break;
    
    }
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul);
}

