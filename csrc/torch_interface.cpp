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
void matmul1(SIG);
void matmul2(SIG);
void matmul3(SIG);
void matmul4(SIG);
void matmul5(SIG);
void matmul6(SIG);
void matmul7(SIG);
void matmul8(SIG);
void matmul9(SIG);
void matmul10(SIG);

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
        case 1:
            matmul1(CALL);
            break;
        case 2:
            matmul2(CALL);
            break;
        case 3:
            matmul3(CALL);
            break;
        case 4:
            matmul4(CALL);
            break;
        case 5:
            matmul5(CALL);
            break;
        case 6:
            matmul6(CALL);
            break;
        case 7:
            matmul7(CALL);
            break;
        case 8:
            matmul8(CALL);
            break;
        case 9:
            matmul9(CALL);
            break;
        case 10:
            matmul10(CALL);
            break;
    
    }
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul);
}

