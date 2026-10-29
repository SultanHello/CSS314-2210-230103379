#include <stdint.h>
#include <stddef.h>

double stride_sum(const double *a, size_t n, size_t stride) {
    double s=0; for(size_t i=0;i<n;i+=stride) s+=a[i]; return s;
}
void matmul(const double *a,const double *b,double *c,int n,int m,int order) {
    for(int i=0;i<m;i++) for(int j=0;j<m;j++) c[(size_t)i*n+j]=0;
    if(order==0) {
        for(int i=0;i<m;i++) for(int j=0;j<m;j++) {
            double s=0; for(int k=0;k<m;k++) s+=a[(size_t)i*n+k]*b[(size_t)k*n+j];
            c[(size_t)i*n+j]=s;
        }
    } else {
        for(int i=0;i<m;i++) for(int k=0;k<m;k++) {
            double x=a[(size_t)i*n+k];
            for(int j=0;j<m;j++) c[(size_t)i*n+j]+=x*b[(size_t)k*n+j];
        }
    }
}
void count_shared(volatile int64_t *a,size_t index,int64_t k) {
    for(int64_t i=0;i<k;i++) a[index]+=1;
}
void count_local(volatile int64_t *a,size_t index,int64_t k) {
    volatile int64_t local=0;
    for(int64_t i=0;i<k;i++) local+=1;
    a[index]=local;
}
/* Repeated application to the same input: a stencil throughput benchmark,
   not a time-evolving PDE. Each pass performs 4 additions + 1 multiplication. */
void stencil(const double *a,double *b,int n,int lo,int hi,int passes) {
    for(int r=0;r<passes;r++) for(int i=lo;i<hi;i++) for(int j=1;j<n-1;j++) {
        size_t k=(size_t)i*n+j;
        b[k]=(a[k]+a[k-1]+a[k+1]+a[k-n]+a[k+n])*0.2;
    }
}
