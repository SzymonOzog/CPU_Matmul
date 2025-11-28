void matmul1(float* a, float* b, float* c, const int M, const int N, const int K)
{
    for(int m = 0; m<M; m++)
    {
        for(int k = 0; k<K; k++)
        {
            for(int n = 0; n<N; n++)
            {
                c[m*N + n] += a[m*K + k] * b[k*N + n];
            }
        }
    }
}
