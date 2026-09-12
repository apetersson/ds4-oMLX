#include "../ds4_gpu.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    assert(ds4_gpu_init());
    const unsigned widths[]={4096,4096,5120}, counts[]={43,45,40};
    const float scales[]={0.5f,1.0f,-0.5f};
    for(unsigned shape=0;shape<3;shape++) {
        unsigned width=widths[shape],layers=counts[shape],n=width*3;
        float *input=malloc(n*4),*out=malloc(n*4),*d=calloc(width*layers,4);
        assert(input && out && d);
        for(unsigned i=0;i<n;i++) input[i]=(float)(i%19)*0.113f+0.017f;
        d[(layers-1)*width]=0.6f; d[(layers-1)*width+1]=0.8f;
        ds4_gpu_tensor *x=ds4_gpu_tensor_alloc(n*4), *directions=ds4_gpu_tensor_alloc(width*layers*4);
        assert(x && directions && ds4_gpu_tensor_write(directions,0,d,width*layers*4));
        for(unsigned a=0;a<3;a++) {
            assert(ds4_gpu_tensor_write(x,0,input,n*4));
            assert(ds4_gpu_directional_steering_project_tensor(x,directions,layers-1,width,3,scales[a]));
            assert(ds4_gpu_tensor_read(x,0,out,n*4));
            for(unsigned row=0;row<3;row++) {
                double dot=(double)input[row*width]*0.6f+(double)input[row*width+1]*0.8f;
                for(unsigned j=0;j<width;j++) {
                    double expected=input[row*width+j]-scales[a]*dot*d[(layers-1)*width+j];
                    assert(fabs(out[row*width+j]-expected)<1e-6);
                    if(j>1) assert(out[row*width+j]==input[row*width+j]);
                }
            }
            /* Verify the release BF16 boundary consumes the projected F32. */
            for(unsigned j=0;j<n;j++) {
                uint32_t bits; memcpy(&bits,&out[j],4);
                bits=(bits+0x7fff+((bits>>16)&1))&0xffff0000u;
                memcpy(&out[j],&bits,4);
            }
            assert(ds4_gpu_dsv41_quantize(x,width,3,DS4_V41_BF16));
            float *rounded=malloc(n*4);assert(rounded);
            assert(ds4_gpu_tensor_read(x,0,rounded,n*4));
            assert(memcmp(out,rounded,n*4)==0);free(rounded);
        }
        ds4_gpu_tensor_free(x);ds4_gpu_tensor_free(directions);free(input);free(out);free(d);
        printf("writer projection and BF16 passed: %ux%u, three rows, three strengths\n",layers,width);
    }
}
