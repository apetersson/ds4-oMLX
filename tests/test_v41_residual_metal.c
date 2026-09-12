#include "../ds4_gpu.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    enum { W=5120, R=3, N=W*4*R };
    float *input=malloc(N*4), *out=malloc(N*4), *d=calloc(W*40,4), *mean=malloc(W*4);
    assert(input && out && d && mean);
    for(int r=0;r<R;r++) for(int h=0;h<4;h++) for(int j=0;j<W;j++)
        input[(r*4+h)*W+j]=(float)(r*3+h)+(float)(j%13)*0.125f;
    d[39*W]=0.6f; d[39*W+1]=0.8f;
    assert(ds4_gpu_init());
    ds4_gpu_tensor *x=ds4_gpu_tensor_alloc(N*4), *dirs=ds4_gpu_tensor_alloc(W*40*4), *cap=ds4_gpu_tensor_alloc(W*4);
    assert(x && dirs && cap && ds4_gpu_tensor_write(dirs,0,d,W*40*4));
    const float scales[]={0,0.5f,1,-0.5f};
    for(unsigned a=0;a<sizeof(scales)/sizeof(scales[0]);a++) {
        assert(ds4_gpu_tensor_write(x,0,input,N*4));
        assert(ds4_gpu_ds41_residual_mean_equal(x,dirs,cap,39,W,R,1,scales[a]));
        assert(ds4_gpu_tensor_read(x,0,out,N*4));
        assert(ds4_gpu_tensor_read(cap,0,mean,W*4));
        if(scales[a]==0) assert(memcmp(input,out,N*4)==0);
        for(int j=0;j<W;j++) assert(mean[j]==4.5f+(float)(j%13)*0.125f);
        for(int r=0;r<R;r++) {
            double dot=(r*3+1.5)*0.6+(r*3+1.625)*0.8;
            for(int h=0;h<4;h++) for(int j=0;j<W;j++) {
                int i=(r*4+h)*W+j;
                double expected=input[i]-scales[a]*dot*d[39*W+j];
                assert(fabs(out[i]-expected)<2e-6);
                if(h) assert(fabs((out[i]-out[r*4*W+j])-h)<2e-6);
            }
        }
    }
    /* The graph rounds a nonzero residual edit before compact BF16 carry.
     * A second carry round must be byte-idempotent, with RNE tie behavior. */
    assert(ds4_gpu_tensor_write(x,0,input,N*4));
    assert(ds4_gpu_ds41_residual_mean_equal(x,dirs,NULL,39,W,R,UINT32_MAX,0.5f));
    assert(ds4_gpu_tensor_read(x,0,out,N*4));
    for(int i=0;i<N;i++) {
        uint32_t bits;
        memcpy(&bits,&out[i],4);
        bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000u;
        memcpy(&out[i],&bits,4);
    }
    assert(ds4_gpu_dsv41_quantize(x,W*4,R,DS4_V41_BF16));
    float *rounded=malloc(N*4); assert(rounded);
    assert(ds4_gpu_tensor_read(x,0,rounded,N*4));
    assert(memcmp(out,rounded,N*4)==0);
    assert(ds4_gpu_dsv41_quantize(x,W*4,R,DS4_V41_BF16));
    assert(ds4_gpu_tensor_read(x,0,rounded,N*4));
    assert(memcmp(out,rounded,N*4)==0);
    free(rounded);
    assert(ds4_gpu_ds41_residual_mean_equal(x,NULL,cap,0,W,R,2,0));
    assert(!ds4_gpu_ds41_residual_mean_equal(x,NULL,cap,0,W,R,2,1));
    assert(!ds4_gpu_ds41_residual_mean_equal(x,dirs,cap,40,W,R,2,1));
    ds4_gpu_tensor_free(x);ds4_gpu_tensor_free(dirs);ds4_gpu_tensor_free(cap);
    free(input);free(out);free(d);free(mean);
    puts("V4.1 residual Metal fixtures passed: capture, zero, half, full, negative, stream differences");
}
