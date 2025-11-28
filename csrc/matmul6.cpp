void matmul6(float* a, float* b, float* c, const int M, const int N, const int K)
{
    for(int m = 0; m<M; m+=4)
    {
        for(int n = 0; n<N; n+=4)
        {
            float c_reg[16] = {0.f};
            for(int k = 0; k<K; k++)
            {
                float a_reg[4];
                a_reg[0] = a[(m+0)*K + k];
                a_reg[1] = a[(m+1)*K + k];
                a_reg[2] = a[(m+2)*K + k];
                a_reg[3] = a[(m+3)*K + k];
                float* b_ptr = &b[k*N + n];

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
            c[(m+0)*N + n+0] = c_reg[0];
            c[(m+0)*N + n+1] = c_reg[1];
            c[(m+0)*N + n+2] = c_reg[2];
            c[(m+0)*N + n+3] = c_reg[3];

            c[(m+1)*N + n+0] = c_reg[4];
            c[(m+1)*N + n+1] = c_reg[5];
            c[(m+1)*N + n+2] = c_reg[6];
            c[(m+1)*N + n+3] = c_reg[7];

            c[(m+2)*N + n+0] = c_reg[8];
            c[(m+2)*N + n+1] = c_reg[9];
            c[(m+2)*N + n+2] = c_reg[10];
            c[(m+2)*N + n+3] = c_reg[11];

            c[(m+3)*N + n+0] = c_reg[12];
            c[(m+3)*N + n+1] = c_reg[13];
            c[(m+3)*N + n+2] = c_reg[14];
            c[(m+3)*N + n+3] = c_reg[15];
        }
    }
}
