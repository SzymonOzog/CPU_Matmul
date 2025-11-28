
void matmul3(float* a, float* b, float* c, const int M, const int N, const int K)
{
    for(int m = 0; m<M; m++)
    {
        for(int k = 0; k<K; k++)
        {
            float a_reg = a[m*K + k];
            for(int n = 0; n<N; n+=4)
            {

                c[m*N + n+0] += a_reg * b[k*N + n+0];
                c[m*N + n+1] += a_reg * b[k*N + n+1];
                c[m*N + n+2] += a_reg * b[k*N + n+2];
                c[m*N + n+3] += a_reg * b[k*N + n+3];
            }
        }
    }
}
