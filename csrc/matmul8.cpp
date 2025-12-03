#include <algorithm>
namespace mm8 {
void inner(float* a, float* b, float* c, 
        const int M, const int N, const int K,
        const int SM, const int SN, const int SK
        );

}
using namespace mm8;
void matmul8(float* a, float* b, float* c, const int M, const int N, const int K)
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

namespace mm8 {
    void inner(float* a, float* b, float* c, 
            const int M, const int N, const int K,
            const int SM, const int SN, const int SK
            )
    {
        for(int m = 0; m<M; m+=4)
        {
            for(int n = 0; n<N; n+=4)
            {
                float c_reg[16] = {0.f};
                for(int k = 0; k<K; k++)
                {
                    float a_reg[4];
                    a_reg[0] = a[(m+0)*SK + k];
                    a_reg[1] = a[(m+1)*SK + k];
                    a_reg[2] = a[(m+2)*SK + k];
                    a_reg[3] = a[(m+3)*SK + k];
                    float* b_ptr = &b[k*SN + n];

                    float b_reg[4];
                    b_reg[0] = *b_ptr++;
                    b_reg[1] = *b_ptr++;
                    b_reg[2] = *b_ptr++;
                    b_reg[3] = *b_ptr++;

                    c_reg[0] += a_reg[0] * b_reg[0];
                    c_reg[1] += a_reg[0] * b_reg[1];
                    c_reg[2] += a_reg[0] * b_reg[2];
                    c_reg[3] += a_reg[0] * b_reg[3];

                    c_reg[4] += a_reg[1] * b_reg[0];
                    c_reg[5] += a_reg[1] * b_reg[1];
                    c_reg[6] += a_reg[1] * b_reg[2];
                    c_reg[7] += a_reg[1] * b_reg[3];

                    c_reg[8] += a_reg[2] * b_reg[0];
                    c_reg[9] += a_reg[2] * b_reg[1];
                    c_reg[10] += a_reg[2] * b_reg[2];
                    c_reg[11] += a_reg[2] * b_reg[3];

                    c_reg[12] += a_reg[3] * b_reg[0];
                    c_reg[13] += a_reg[3] * b_reg[1];
                    c_reg[14] += a_reg[3] * b_reg[2];
                    c_reg[15] += a_reg[3] * b_reg[3];
                }
                c[(m+0)*SN + n+0] += c_reg[0];
                c[(m+0)*SN + n+1] += c_reg[1];
                c[(m+0)*SN + n+2] += c_reg[2];
                c[(m+0)*SN + n+3] += c_reg[3];

                c[(m+1)*SN + n+0] += c_reg[4];
                c[(m+1)*SN + n+1] += c_reg[5];
                c[(m+1)*SN + n+2] += c_reg[6];
                c[(m+1)*SN + n+3] += c_reg[7];

                c[(m+2)*SN + n+0] += c_reg[8];
                c[(m+2)*SN + n+1] += c_reg[9];
                c[(m+2)*SN + n+2] += c_reg[10];
                c[(m+2)*SN + n+3] += c_reg[11];

                c[(m+3)*SN + n+0] += c_reg[12];
                c[(m+3)*SN + n+1] += c_reg[13];
                c[(m+3)*SN + n+2] += c_reg[14];
                c[(m+3)*SN + n+3] += c_reg[15];
            }
        }
    }
}
