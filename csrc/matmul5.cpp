void matmul5(float* a, float* b, float* c, const int M, const int N, const int K)
{
    for(int k = 0; k<K; k++)
    {
        for(int m = 0; m<M; m+=4)
        {
            float a_reg[4];
            a_reg[0] = a[(m+0)*K + k];
            a_reg[1] = a[(m+1)*K + k];
            a_reg[2] = a[(m+2)*K + k];
            a_reg[3] = a[(m+3)*K + k];
            for(int n = 0; n<N; n+=4)
            {
                float* b_ptr = &b[k*N + n];
                float b_reg[4];
                b_reg[0] = *b_ptr++;
                b_reg[1] = *b_ptr++;
                b_reg[2] = *b_ptr++;
                b_reg[3] = *b_ptr;

                c[(m+0)*N + n+0] += a_reg[0] * b_reg[0];
                c[(m+0)*N + n+1] += a_reg[0] * b_reg[1];
                c[(m+0)*N + n+2] += a_reg[0] * b_reg[2];
                c[(m+0)*N + n+3] += a_reg[0] * b_reg[3];

                c[(m+1)*N + n+0] += a_reg[1] * b_reg[0];
                c[(m+1)*N + n+1] += a_reg[1] * b_reg[1];
                c[(m+1)*N + n+2] += a_reg[1] * b_reg[2];
                c[(m+1)*N + n+3] += a_reg[1] * b_reg[3];

                c[(m+2)*N + n+0] += a_reg[2] * b_reg[0];
                c[(m+2)*N + n+1] += a_reg[2] * b_reg[1];
                c[(m+2)*N + n+2] += a_reg[2] * b_reg[2];
                c[(m+2)*N + n+3] += a_reg[2] * b_reg[3];
                
                c[(m+3)*N + n+0] += a_reg[3] * b_reg[0];
                c[(m+3)*N + n+1] += a_reg[3] * b_reg[1];
                c[(m+3)*N + n+2] += a_reg[3] * b_reg[2];
                c[(m+3)*N + n+3] += a_reg[3] * b_reg[3];
            }
        }
    }
}
