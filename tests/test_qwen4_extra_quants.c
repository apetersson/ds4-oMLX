#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include "ds4_gpu.h"
bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }
static void check(int ok,const char *s) { if(!ok){fprintf(stderr,"FAIL %s\n",s);exit(1);} }
int main(int argc,char **argv) {
 check(argc==2,"fixture directory argument");check(ds4_gpu_init(),"gpu init");
 const uint32_t types[]={7,13,14,20};const size_t sizes[]={196608,180224,215040,147456};
 for(int k=0;k<4;k++) {
  void *map=NULL;check(!posix_memalign(&map,16384,1048576),"map allocation");memset(map,0,1048576);
  char path[4096];snprintf(path,sizeof(path),"%s/%u.bin",argv[1],types[k]);FILE *f=fopen(path,"rb");check(f!=NULL,"quant fixture");check(fread(map,1,sizes[k],f)==sizes[k],"quant read");fclose(f);
  float *ref=malloc(1048576);snprintf(path,sizeof(path),"%s/%u.f32",argv[1],types[k]);f=fopen(path,"rb");check(f!=NULL,"reference fixture");check(fread(ref,1,1048576,f)==1048576,"reference read");fclose(f);
  check(ds4_gpu_set_model_map(map,1048576),"register fixture");
  for(uint32_t T=1;T<=3;T++) {
   float x[768],y[24];for(uint32_t i=0;i<T*256;i++)x[i]=sinf((float)i*.13f);
   ds4_gpu_tensor *gx=ds4_gpu_tensor_alloc(sizeof(x)),*gy=ds4_gpu_tensor_alloc(sizeof(y));
   check(ds4_gpu_tensor_write(gx,0,x,T*256*sizeof(float)),"write");
   ds4_gpu_tensor *outs[]={gy};uint64_t off[]={0};uint32_t rows[]={8};
   check(ds4_gpu_qwen4_multi_gemv_tensor(gx,T,256,1,outs,map,1048576,off,&types[k],rows),"quant gemv");
   check(ds4_gpu_tensor_read(gy,0,y,T*8*sizeof(float)),"read");double worst=0;
   for(uint32_t t=0;t<T;t++)for(uint32_t r=0;r<8;r++) {double e=0,absum=0;for(int i=0;i<256;i++){double v=ref[r*256+i]*x[t*256+i];e+=v;absum+=fabs(v);}double err=fabs(y[t*8+r]-e);if(err>worst)worst=err;check(isfinite(y[t*8+r])&&err<1e-4+absum*2e-5,"reference dot equality");}
   printf("PASS type=%u T=%u max_abs_error=%.9g\n",types[k],T,worst);
   ds4_gpu_tensor_free(gx);ds4_gpu_tensor_free(gy);
   /* HC up uses a different element-reader path from the SIMD row-dot. */
   float mixed[192];
   gx=ds4_gpu_tensor_alloc(sizeof(x));gy=ds4_gpu_tensor_alloc(sizeof(mixed));
   ds4_gpu_tensor *lo=ds4_gpu_tensor_alloc(sizeof(x));
   check(ds4_gpu_tensor_write(gx,0,x,T*256*sizeof(float)),"HC xn write");
   check(ds4_gpu_tensor_write(lo,0,x,T*256*sizeof(float)),"HC lo write");
   check(ds4_gpu_qwen4_hc_gate_mix_tensor(gy,gx,lo,map,1048576,0,types[k],T,64,4,256),"HC quant gate");
   check(ds4_gpu_tensor_read(gy,0,mixed,T*64*sizeof(float)),"HC read");
   for(uint32_t t=0;t<T;t++)for(uint32_t d=0;d<64;d++) {
    double want=0;
    for(uint32_t h=0;h<4;h++) {
     double dot=0;for(uint32_t j=0;j<256;j++) {double v=x[t*256+j]/4.0;dot+=ref[(h*64+d)*256+j]*v/(1+exp(-v));}
     want+=x[t*256+h*64+d]/(1+exp(-dot));
    }
    want/=4;check(isfinite(mixed[t*64+d])&&fabs(mixed[t*64+d]-want)<2e-4,"HC quant reference");
   }
   printf("PASS HC type=%u T=%u\n",types[k],T);
   ds4_gpu_tensor_free(gx);ds4_gpu_tensor_free(gy);ds4_gpu_tensor_free(lo);
  }
  free(ref); /* Fixture model views remain registered until process exit. */
 }
 return 0;
}
