/* Scalar row readers against independently decoded GGML fixtures. */
#define DS4_NO_GPU
#include "../ds4.c"
int main(int argc,char **argv) {
 if(argc!=2)return 2;
 const uint32_t types[]={7,13,14,20};
 for(int k=0;k<4;k++) {
  uint64_t bytes=0;if(!tensor_nbytes(types[k],262144,&bytes))return 3;
  char path[4096];snprintf(path,sizeof(path),"%s/%u.bin",argv[1],types[k]);FILE *f=fopen(path,"rb");if(!f)return 4;
  uint8_t *data=malloc(bytes);if(fread(data,1,bytes,f)!=bytes)return 5;fclose(f);
  snprintf(path,sizeof(path),"%s/%u.f32",argv[1],types[k]);f=fopen(path,"rb");if(!f)return 6;
  float *ref=malloc(1048576);if(fread(ref,1,1048576,f)!=1048576)return 7;fclose(f);
  ds4_model model={0};model.map=data;model.size=bytes;
  ds4_tensor t={0};t.type=types[k];t.dim[0]=256;t.dim[1]=1024;t.ndim=2;t.elements=262144;t.bytes=bytes;
  float out[256];double worst=0;
  for(uint32_t r=0;r<1024;r++) {
   qwen4_ref_row(&model,&t,r,out);
   for(uint32_t i=0;i<256;i++) {double e=fabs(out[i]-ref[r*256+i]);if(e>worst)worst=e;if(!isfinite(out[i])||e>1e-4+fabs(ref[r*256+i])*2e-6){fprintf(stderr,"FAIL CPU type=%u row=%u i=%u error=%g\n",types[k],r,i,e);return 8;}}
  }
  printf("PASS CPU type=%u rows=1024 max_abs_error=%g\n",types[k],worst);free(data);free(ref);
 }
 uint16_t raw[320];float out[160];for(int i=0;i<320;i++)raw[i]=(uint16_t)(0x3800+(i%128)*3+(i%2?0x8000:0));
 ds4_model m={0};m.map=(const uint8_t *)raw;m.size=sizeof(raw);ds4_tensor t={0};t.type=DS4_TENSOR_BF16;t.ndim=2;t.dim[0]=160;t.dim[1]=2;t.bytes=sizeof(raw);
 qwen4_ref_row(&m,&t,1,out);
 for(int i=0;i<160;i++){uint32_t expected=(uint32_t)raw[160+i]<<16,actual;memcpy(&actual,&out[i],4);if(expected!=actual)return 9;}
 puts("PASS CPU BF16 160-value PLE row");return 0;
}
