void matmul2(float* a, float* b, float* c, const int M, const int N, const int K)
{
    for(int m = 0; m<M; m++)
    {
        for(int k = 0; k<K; k++)
        {
            for(int n = 0; n<N; n+=4)
            {
                c[m*N + n+0] += a[m*K + k] * b[k*N + n+0];
                c[m*N + n+1] += a[m*K + k] * b[k*N + n+1];
                c[m*N + n+2] += a[m*K + k] * b[k*N + n+2];
                c[m*N + n+3] += a[m*K + k] * b[k*N + n+3];
            }
        }
    }
}
