#include <algorithm>
#include <immintrin.h>

namespace mm9 {
void inner(float* a, float* b, float* c, 
        const int M, const int N, const int K,
        const int SM, const int SN, const int SK
        );

}
using namespace mm9;
void matmul9(float* a, float* b, float* c, const int M, const int N, const int K)
{
    constexpr int BK = 128;
    constexpr int BN = 128;
    constexpr int BM = 128;
    #pragma omp parallel for collapse(2) schedule(static)
    for(int m = 0; m<M; m+=BM)
    {
        for(int n = 0; n<N; n+=BN)
        {
            for(int k = 0; k<K; k+=BK)
            {
                float* b_a = a + m*K + k;
                float* b_b = b + k*N + n;
                float* b_c = c + m*N + n;
                int bm = std::min(M-m, BM);
                int bn = std::min(N-n, BN);
                int bk = std::min(K-k, BK);
                inner(b_a, b_b, b_c, bm, bn, bk, M, N, K);
            }
        }
    }
}

namespace mm9 {
    void inner(float* a, float* b, float* c, 
            const int M, const int N, const int K,
            const int SM, const int SN, const int SK
            )
    {
        for(int m = 0; m<M; m+=4)
        {
            for(int n = 0; n<N; n+=8)
            {
                __m256 c_reg[4] = {
                    _mm256_setzero_ps(),
                    _mm256_setzero_ps(),
                    _mm256_setzero_ps(),
                    _mm256_setzero_ps()
                };
                for(int k = 0; k<K; k++)
                {
                    float a_reg[4];
                    a_reg[0] = a[(m+0)*SK + k];
                    a_reg[1] = a[(m+1)*SK + k];
                    a_reg[2] = a[(m+2)*SK + k];
                    a_reg[3] = a[(m+3)*SK + k];
                    float* b_ptr = &b[k*SN + n];

                    __m256 vec_b = _mm256_loadu_ps(b_ptr);
                    b_ptr += 8;
                    float* b_reg = reinterpret_cast<float*>(&vec_b);

                    c_reg[0] = _mm256_fmadd_ps(vec_b, _mm256_set1_ps(a_reg[0]), c_reg[0]);
                    c_reg[1] = _mm256_fmadd_ps(vec_b, _mm256_set1_ps(a_reg[1]), c_reg[1]);
                    c_reg[2] = _mm256_fmadd_ps(vec_b, _mm256_set1_ps(a_reg[2]), c_reg[2]);
                    c_reg[3] = _mm256_fmadd_ps(vec_b, _mm256_set1_ps(a_reg[3]), c_reg[3]);
                }
                _mm256_storeu_ps(&c[(m + 0) * SN + n],
                     _mm256_add_ps(_mm256_loadu_ps(&c[(m + 0) * SN + n]), c_reg[0]));
                _mm256_storeu_ps(&c[(m + 1) * SN + n],
                        _mm256_add_ps(_mm256_loadu_ps(&c[(m + 1) * SN + n]), c_reg[1]));
                _mm256_storeu_ps(&c[(m + 2) * SN + n],
                        _mm256_add_ps(_mm256_loadu_ps(&c[(m + 2) * SN + n]), c_reg[2]));
                _mm256_storeu_ps(&c[(m + 3) * SN + n],
                        _mm256_add_ps(_mm256_loadu_ps(&c[(m + 3) * SN + n]), c_reg[3]));
            }
        }
    }
}

