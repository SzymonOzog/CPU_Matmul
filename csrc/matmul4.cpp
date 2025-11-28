void matmul4(float* a, float* b, float* c, const int M, const int N, const int K)
{
    for(int k = 0; k<K; k++)
    {
        for(int m = 0; m<M; m++)
        {
            float a_reg = a[m*K + k];
            for(int n = 0; n<N; n+=4)
            {
                float* b_ptr = &b[k*N + n];
                c[m*N + n+0] += a_reg * *b_ptr++;
                c[m*N + n+1] += a_reg * *b_ptr++;
                c[m*N + n+2] += a_reg * *b_ptr++;
                c[m*N + n+3] += a_reg * *b_ptr;
            }
        }
    }
}
