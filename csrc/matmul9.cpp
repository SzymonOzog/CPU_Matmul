#include <algorithm>
#include <arm_neon.h>

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
            for(int n = 0; n<N; n+=4)
            {
                float32x4_t c_reg[4] = {{0.f}};
                for(int k = 0; k<K; k++)
                {
                    float a_reg[4];
                    a_reg[0] = a[(m+0)*SK + k];
                    a_reg[1] = a[(m+1)*SK + k];
                    a_reg[2] = a[(m+2)*SK + k];
                    a_reg[3] = a[(m+3)*SK + k];
                    float* b_ptr = &b[k*SN + n];

                    float32x4_t vec_b = vld1q_f32(b_ptr); // Load 4 floats
                    b_ptr += 4;
                    float* b_reg = reinterpret_cast<float*>(&vec_b);

                    c_reg[0] = vmlaq_n_f32(c_reg[0], vec_b, a_reg[0]);
                    c_reg[1] = vmlaq_n_f32(c_reg[1], vec_b, a_reg[1]);
                    c_reg[2] = vmlaq_n_f32(c_reg[2], vec_b, a_reg[2]);
                    c_reg[3] = vmlaq_n_f32(c_reg[3], vec_b, a_reg[3]);
                }
                float32x4_t* c_vec = reinterpret_cast<float32x4_t*>(&c[(m+0)*SN + n+0]);
                vst1q_f32(&c[(m+0)*SN + n+0], 
                        vaddq_f32(*c_vec, 
                            c_reg[0]
                            )
                        );

                c_vec = reinterpret_cast<float32x4_t*>(&c[(m+1)*SN + n+0]);
                vst1q_f32(&c[(m+1)*SN + n+0], 
                        vaddq_f32(*c_vec, 
                            c_reg[1]
                            )
                        );

                c_vec = reinterpret_cast<float32x4_t*>(&c[(m+2)*SN + n+0]);
                vst1q_f32(&c[(m+2)*SN + n+0], 
                        vaddq_f32(*c_vec, 
                            c_reg[2]
                            )
                        );


                c_vec = reinterpret_cast<float32x4_t*>(&c[(m+3)*SN + n+0]);
                vst1q_f32(&c[(m+3)*SN + n+0], 
                        vaddq_f32(*c_vec, 
                            c_reg[3]
                            )
                        );
            }
        }
    }
}

