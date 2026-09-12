#include "../ds4.h"
#include "../ds4_v41_intervention.h"
#include <assert.h>

int main(int argc, char **argv) {
    assert(argc == 3 || argc == 4);
    uint32_t site = argc == 4 ? (uint32_t)atoi(argv[3]) : DS41_SITE_WRITER;
    int context = getenv("DS41_TEST_CONTEXT") ? atoi(getenv("DS41_TEST_CONTEXT")) : 4096;
    assert(context > 1 && context <= 32768);
    ds4_engine_options opt = {.model_path=argv[1], .backend=DS4_BACKEND_METAL,
        .ssd_streaming=true, .context_size=context};
    ds4_engine *e = NULL;
    assert(ds4_engine_open(&e, &opt) == 0);
    ds4_tokens prompt = {0};
    ds4_tokenize_text(e, "What is the capital of Austria?", &prompt);
    assert(prompt.len > 1);
    const char *mode = getenv("DS41_TEST_MODE");
    int scalar = mode && !strcmp(mode,"scalar");
    const char *length = getenv("DS41_TEST_TOKENS");
    if (length) {
        int target = atoi(length), original = prompt.len;
        assert(target >= original && target < context);
        while (prompt.len < target) ds4_tokens_push(&prompt,prompt.v[prompt.len % original]);
    }
    printf("site=%u tokens=%d mode=%s\n",site,prompt.len,scalar ? "scalar-final" : "prefill"); fflush(stdout);
    const char *position = getenv("DS41_TEST_CAPTURE_POS");
    int capture_pos = position ? atoi(position) : prompt.len-1;
    assert(capture_pos >= 0 && capture_pos < prompt.len);
    const char *mask_text = getenv("DS41_TEST_CAPTURE_MASK");
    uint64_t mask = mask_text ? strtoull(mask_text,NULL,0) : (UINT64_C(1)<<40)-1;
    assert(mask && !(mask >> 40));
    unsigned char digest[32] = {0x1c,0xe6,0xa8,0xf8,0x80,0x62,0x05,0xc1,0x33,0x30,0xd7,0xca,0x28,0x7b,0xd1,0x98,0x33,0x1d,0xc5,0xca,0x35,0xcc,0xc5,0xd8,0xa9,0xa9,0x2a,0x18,0x8a,0x6f,0x6f,0x42};
    const int vocab = ds4_engine_vocab_size(e);
    float *base = malloc(vocab*sizeof(float)), *logits = malloc(vocab*sizeof(float));
    float *capture = malloc(DS41_DIRECTION_VALUES*sizeof(float));
    float *base_capture = malloc(DS41_DIRECTION_VALUES*sizeof(float));
    assert(base && logits && capture && base_capture);
    for (int arm = -1; arm < 3; arm++) {
        ds4_session *s = NULL;
        char error[256] = {0};
        uint64_t layers = 0;
        assert(ds4_session_create(&s,e,context)==0);
        if (arm == -1) {
            assert(ds4_session_v41_configure(s,NULL,NULL,site,0,0,1)!=0);
            assert(ds4_session_v41_configure(s,NULL,digest,99,0,0,1)!=0);
            assert(ds4_session_v41_configure(s,NULL,digest,site,0,context,1)!=0);
            assert(ds4_session_v41_configure(s,NULL,digest,site,0,0,UINT64_C(1)<<40)!=0);
            assert(ds4_session_v41_configure(s,NULL,digest,site,1,0,1)!=0);
            volatile uint32_t bad_bits = 0x7fc00001u;
            uint32_t raw = bad_bits;
            float bad_alpha; memcpy(&bad_alpha,&raw,4);
            assert(ds4_session_v41_configure(s,NULL,digest,site,bad_alpha,0,1)!=0);
            unsigned char wrong_digest[32]; memcpy(wrong_digest,digest,32); wrong_digest[0]^=1;
            assert(ds4_session_v41_configure(s,argv[2],wrong_digest,site,0,0,1)!=0);
            assert(ds4_session_v41_configure(s,argv[2],digest,site==1?2:1,0,0,1)!=0);
        }
        if (arm >= 0) assert(ds4_session_v41_configure(s,arm ? argv[2] : NULL,digest,site,
            arm == 2 ? 1.0f : 0.0f,capture_pos,mask)==0);
        if (scalar) {
            ds4_tokens prefix = prompt;
            prefix.len--;
            assert(ds4_session_sync(s,&prefix,error,sizeof(error))==0);
            if (arm >= 0 && capture_pos == prompt.len-1) assert(ds4_session_v41_writer_capture(s,capture,DS41_DIRECTION_VALUES,&layers)!=0);
            assert(ds4_session_eval(s,prompt.v[prompt.len-1],error,sizeof(error))==0);
        } else {
            int rc = ds4_session_sync(s,&prompt,error,sizeof(error));
            if (rc) fprintf(stderr,"sync failed: %s\n",error);
            assert(rc==0);
        }
        if (arm >= 0) {
        assert(ds4_session_v41_writer_capture(s,capture,DS41_DIRECTION_VALUES,&layers)==0);
        assert(layers == mask);
        ds4_v41_capture_info info;
        assert(ds4_session_v41_capture(s,capture,DS41_DIRECTION_VALUES,&info)==0);
        assert(info.version==1 && info.layer_count==40 && info.width==5120);
        assert(info.site==site && info.position==(uint32_t)capture_pos && info.layers==mask);
        assert(memcmp(info.model_sha256,digest,32)==0);
        if (arm==0) memcpy(base_capture,capture,DS41_DIRECTION_VALUES*sizeof(float));
        if (arm==1) assert(memcmp(base_capture,capture,DS41_DIRECTION_VALUES*sizeof(float))==0);
        if (arm==2 && (mask & 1)) assert(memcmp(base_capture,capture,5120*sizeof(float))==0);
        for (uint32_t layer=0;layer<40;layer++) if (!(mask & (UINT64_C(1)<<layer)))
            for (uint32_t j=0;j<5120;j++) assert(capture[layer*5120+j]==0);

        }
        assert(ds4_session_copy_logits(s,logits,vocab)==vocab);
        for (unsigned i=0;arm>=0 && i<DS41_DIRECTION_VALUES;i++) {
            uint32_t bits; memcpy(&bits,&capture[i],4); assert(ds41_bits_finite(bits));
        }
        if (arm==-1) memcpy(base,logits,vocab*sizeof(float));
        else if(arm<=1) assert(memcmp(base,logits,vocab*sizeof(float))==0);
        else assert(memcmp(base,logits,vocab*sizeof(float))!=0);
        assert(ds4_session_v41_writer_configure(s,NULL,digest,0,0,0)!=0);
        printf("arm=%d capture_mask=%llx passed\n",arm,(unsigned long long)layers); fflush(stdout);
        ds4_session_free(s);
    }
    ds4_tokens_free(&prompt); ds4_engine_close(e);
    free(base); free(logits); free(capture); free(base_capture);
    puts("V4.1 intervention live fixtures passed");
}
