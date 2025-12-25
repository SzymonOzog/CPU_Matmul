#include <algorithm>
#include <immintrin.h>

namespace mm11 {
void inner(float* a, float* b, float* c, 
        const int M, const int N, const int K,
        const int SM, const int SN, const int SK
        );

}
using namespace mm11;
void matmul11(float* a, float* b, float* c, const int M, const int N, const int K)
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
                alignas(64) float pack_a[BM*BK];
                float* b_a = a + m*K + k;
                float* b_b = b + k*N + n;
                float* b_c = c + m*N + n;
                int bm = std::min(M-m, BM);
                int bn = std::min(N-n, BN);
                int bk = std::min(K-k, BK);

                float* pa = pack_a;
                for(int pm = 0; pm<bm; pm+=4)
                {
                    for(int pk = 0; pk<bk; pk++)
                    {
                       *pa++ = b_a[(0+pm)*K + pk];
                       *pa++ = b_a[(1+pm)*K + pk];
                       *pa++ = b_a[(2+pm)*K + pk];
                       *pa++ = b_a[(3+pm)*K + pk];
                    }
                }

                inner(pack_a, b_b, b_c, bm, bn, bk, M, N, K);
            }
        }
    }
}

namespace mm11 {
    void inner(float* __restrict__ a, float* __restrict__ b, float* __restrict__ c, 
            const int M, const int N, const int K,
            const int SM, const int SN, const int SK
            )
    {
        for(int m = 0; m<M; m+=4)
        {
            for(int n = 0; n<N; n+=8)
            {
                float* a_ptr = a + m*K;
                __m256 c_reg[4] = {
                    _mm256_loadu_ps(&c[(m+0)*SN + n+0]),
                    _mm256_loadu_ps(&c[(m+1)*SN + n+0]),
                    _mm256_loadu_ps(&c[(m+2)*SN + n+0]),
                    _mm256_loadu_ps(&c[(m+3)*SN + n+0])
                };
                for(int k = 0; k<K; k++)
                {
                    float* b_ptr = &b[k*SN + n];

                    __m256 vec_b = _mm256_loadu_ps(b_ptr);
                    b_ptr += 8;
                    c_reg[0] = _mm256_fmadd_ps(vec_b, _mm256_set1_ps(a_ptr[0]), c_reg[0]);
                    c_reg[1] = _mm256_fmadd_ps(vec_b, _mm256_set1_ps(a_ptr[1]), c_reg[1]);
                    c_reg[2] = _mm256_fmadd_ps(vec_b, _mm256_set1_ps(a_ptr[2]), c_reg[2]);
                    c_reg[3] = _mm256_fmadd_ps(vec_b, _mm256_set1_ps(a_ptr[3]), c_reg[3]);
                    a_ptr += 4;
                }
                _mm256_storeu_ps(&c[(m+0)*SN + n+0], c_reg[0]);
                _mm256_storeu_ps(&c[(m+1)*SN + n+0], c_reg[1]);
                _mm256_storeu_ps(&c[(m+2)*SN + n+0], c_reg[2]);
                _mm256_storeu_ps(&c[(m+3)*SN + n+0], c_reg[3]);

            }
        }
    }
}

