#include <algorithm>
#include <arm_neon.h>

namespace mm10 {
void inner(float* a, float* b, float* c, 
        const int M, const int N, const int K,
        const int SM, const int SN, const int SK
        );

}
using namespace mm10;
void matmul10(float* a, float* b, float* c, const int M, const int N, const int K)
{
    constexpr int BK = 128;
    constexpr int BN = 128;
    constexpr int BM = 128;
    float pack_a[BM*BK];
    #pragma omp parallel for collapse(2) schedule(static) private(pack_a)
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

namespace mm10 {
    void inner(float* a, float* b, float* c, 
            const int M, const int N, const int K,
            const int SM, const int SN, const int SK
            )
    {
        for(int m = 0; m<M; m+=4)
        {
            for(int n = 0; n<N; n+=4)
            {
                float* a_ptr = a + m*K;
                float32x4_t c_reg[4] = {{0.f}};
                for(int k = 0; k<K; k++)
                {
                    float32x4_t vec_a = vld1q_f32(a_ptr); // Load 4 floats
                    a_ptr += 4;
                    float* b_ptr = &b[k*SN + n];

                    float32x4_t vec_b = vld1q_f32(b_ptr); // Load 4 floats
                    b_ptr += 4;
                    float* b_reg = reinterpret_cast<float*>(&vec_b);

                    c_reg[0] = vmlaq_n_f32(c_reg[0], vec_b, vgetq_lane_f32(vec_a, 0));
                    c_reg[1] = vmlaq_n_f32(c_reg[1], vec_b, vgetq_lane_f32(vec_a, 1));
                    c_reg[2] = vmlaq_n_f32(c_reg[2], vec_b, vgetq_lane_f32(vec_a, 2));
                    c_reg[3] = vmlaq_n_f32(c_reg[3], vec_b, vgetq_lane_f32(vec_a, 3));
                }
                float32x4_t* c_vec = reinterpret_cast<float32x4_t*>(&c[(m+0)*SN + n+0]);
                vst1q_f32(&c[(m+0)*SN + n+0], 
                        vaddq_f32(*c_vec, c_reg[0])
                        );

                c_vec = reinterpret_cast<float32x4_t*>(&c[(m+1)*SN + n+0]);
                vst1q_f32(&c[(m+1)*SN + n+0], 
                        vaddq_f32(*c_vec, c_reg[1])
                        );

                c_vec = reinterpret_cast<float32x4_t*>(&c[(m+2)*SN + n+0]);
                vst1q_f32(&c[(m+2)*SN + n+0], 
                        vaddq_f32(*c_vec, c_reg[2])
                        );


                c_vec = reinterpret_cast<float32x4_t*>(&c[(m+3)*SN + n+0]);
                vst1q_f32(&c[(m+3)*SN + n+0], 
                        vaddq_f32(*c_vec, c_reg[3])
                        );
            }
        }
    }
}

