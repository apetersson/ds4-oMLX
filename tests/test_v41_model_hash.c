#include "../ds4_model_hash.h"
#include <assert.h>
#include <stdio.h>
#include <fcntl.h>

int main(void) {
#if defined(__APPLE__)
    unsigned char digest[32];
    FILE *f=tmpfile(); assert(f);
    assert(fwrite("abc",1,3,f)==3 && fflush(f)==0);
    const unsigned char abc[32]={0xba,0x78,0x16,0xbf,0x8f,0x01,0xcf,0xea,0x41,0x41,0x40,0xde,0x5d,0xae,0x22,0x23,0xb0,0x03,0x61,0xa3,0x96,0x17,0x7a,0x9c,0xb4,0x10,0xff,0x61,0xf2,0x00,0x15,0xad};
    assert(ds4_model_fd_sha256(fileno(f),3,digest)==0 && !memcmp(digest,abc,32));
    assert(ftell(f)==3); /* pread leaves the loader's offset untouched. */
    assert(ds4_model_fd_sha256(fileno(f),4,digest)!=0);
    assert(ds4_model_fd_sha256(-1,3,digest)!=0);
    assert(ds4_model_fd_sha256(fileno(f),3,NULL)!=0);
    fclose(f);
    f=tmpfile();assert(f);
    /* Cross two 1 MiB boundaries and leave a partial final block. */
    size_t count=2*1024*1024+19;
    unsigned char *data=malloc(count); assert(data);
    for(size_t i=0;i<count;i++) data[i]=(unsigned char)(i*17);
    unsigned char expected[32];CC_SHA256(data,(CC_LONG)count,expected);
    assert(fwrite(data,1,count,f)==count && fflush(f)==0);
    assert(ds4_model_fd_sha256(fileno(f),count,digest)==0 && !memcmp(digest,expected,32));
    free(data);fclose(f);
    puts("Opened-model bounded SHA256 fixtures: OK");
#else
    unsigned char digest[32];assert(ds4_model_fd_sha256(-1,0,digest)!=0);
    puts("Model SHA256 unsupported-platform rejection: OK");
#endif
}
