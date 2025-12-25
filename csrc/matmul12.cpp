#include <algorithm>
#include <immintrin.h>
namespace mm12 {
void inner(float* a, float* b, float* c, 
        const int M, const int N, const int K,
        const int SM, const int SN, const int SK
        );
}
using namespace mm12;
void matmul12(float* a, float* b, float* c, const int M, const int N, const int K)
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
                alignas(64) float pack_b[BN*BK];
                float* b_a = a + m*K + k;
                float* b_b = b + k*N + n;
                float* b_c = c + m*N + n;
                int bm = std::min(M-m, BM);
                int bn = std::min(N-n, BN);
                int bk = std::min(K-k, BK);
                float* pa = pack_a;
                for(int pm = 0; pm<bm; pm+=8)
                {
                    for(int pk = 0; pk<bk; pk++)
                    {
                       *pa++ = b_a[(0+pm)*K + pk];
                       *pa++ = b_a[(1+pm)*K + pk];
                       *pa++ = b_a[(2+pm)*K + pk];
                       *pa++ = b_a[(3+pm)*K + pk];
                       *pa++ = b_a[(4+pm)*K + pk];
                       *pa++ = b_a[(5+pm)*K + pk];
                       *pa++ = b_a[(6+pm)*K + pk];
                       *pa++ = b_a[(7+pm)*K + pk];
                    }
                }
                float* pb = pack_b;
                for(int pn = 0; pn<bn; pn+=8)
                {
                    for(int pk = 0; pk<bk; pk++)
                    {
                       *pb++ = b_b[pk*N + (0+pn)];
                       *pb++ = b_b[pk*N + (1+pn)];
                       *pb++ = b_b[pk*N + (2+pn)];
                       *pb++ = b_b[pk*N + (3+pn)];
                       *pb++ = b_b[pk*N + (4+pn)];
                       *pb++ = b_b[pk*N + (5+pn)];
                       *pb++ = b_b[pk*N + (6+pn)];
                       *pb++ = b_b[pk*N + (7+pn)];
                    }
                }
                inner(pack_a, pack_b, b_c, bm, bn, bk, M, N, K);
            }
        }
    }
}
namespace mm12 {
    void inner(float* __restrict__ a, float* __restrict__ b, float* __restrict__ c, 
            const int M, const int N, const int K,
            const int SM, const int SN, const int SK
            )
    {
        for(int m = 0; m<M; m+=8)
        {
            float* b_ptr = b;
            for(int n = 0; n<N; n+=8)
            {
                float* a_ptr = a + m*K;
                __m256 c_reg[8] = {
                    _mm256_loadu_ps(&c[(m+0)*SN + n+0]),
                    _mm256_loadu_ps(&c[(m+1)*SN + n+0]),
                    _mm256_loadu_ps(&c[(m+2)*SN + n+0]),
                    _mm256_loadu_ps(&c[(m+3)*SN + n+0]),
                    _mm256_loadu_ps(&c[(m+4)*SN + n+0]),
                    _mm256_loadu_ps(&c[(m+5)*SN + n+0]),
                    _mm256_loadu_ps(&c[(m+6)*SN + n+0]),
                    _mm256_loadu_ps(&c[(m+7)*SN + n+0])
                };
                for(int k = 0; k<K; k++)
                {
                    __m256 vec_a = _mm256_loadu_ps(a_ptr);
                    a_ptr += 8;
                    __m256 vec_b = _mm256_loadu_ps(b_ptr);
                    b_ptr += 8;
                    c_reg[0] = _mm256_fmadd_ps(vec_b, _mm256_set1_ps(vec_a[0]), c_reg[0]);
                    c_reg[1] = _mm256_fmadd_ps(vec_b, _mm256_set1_ps(vec_a[1]), c_reg[1]);
                    c_reg[2] = _mm256_fmadd_ps(vec_b, _mm256_set1_ps(vec_a[2]), c_reg[2]);
                    c_reg[3] = _mm256_fmadd_ps(vec_b, _mm256_set1_ps(vec_a[3]), c_reg[3]);
                    c_reg[4] = _mm256_fmadd_ps(vec_b, _mm256_set1_ps(vec_a[4]), c_reg[4]);
                    c_reg[5] = _mm256_fmadd_ps(vec_b, _mm256_set1_ps(vec_a[5]), c_reg[5]);
                    c_reg[6] = _mm256_fmadd_ps(vec_b, _mm256_set1_ps(vec_a[6]), c_reg[6]);
                    c_reg[7] = _mm256_fmadd_ps(vec_b, _mm256_set1_ps(vec_a[7]), c_reg[7]);
                }
                _mm256_storeu_ps(&c[(m+0)*SN + n+0], c_reg[0]);
                _mm256_storeu_ps(&c[(m+1)*SN + n+0], c_reg[1]);
                _mm256_storeu_ps(&c[(m+2)*SN + n+0], c_reg[2]);
                _mm256_storeu_ps(&c[(m+3)*SN + n+0], c_reg[3]);
                _mm256_storeu_ps(&c[(m+4)*SN + n+0], c_reg[4]);
                _mm256_storeu_ps(&c[(m+5)*SN + n+0], c_reg[5]);
                _mm256_storeu_ps(&c[(m+6)*SN + n+0], c_reg[6]);
                _mm256_storeu_ps(&c[(m+7)*SN + n+0], c_reg[7]);
            }
        }
    }
}
