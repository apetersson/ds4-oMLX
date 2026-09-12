#include "../ds4.h"
#include "../ds4_v41_intervention.h"
#include <assert.h>

/* SSD graphs use the upstream sequential batch fallback, not native batching.
 * Different sites and prompts make accidental shared intervention state visible. */
int main(int argc, char **argv) {
    assert(argc==4);
    unsigned char digest[32];
    FILE *f=fopen(argv[2],"rb"); assert(f);
    assert(fseek(f,40,SEEK_SET)==0 && fread(digest,1,32,f)==32); fclose(f);
    ds4_engine_options options={.model_path=argv[1],.backend=DS4_BACKEND_METAL,
        .ssd_streaming=true,.context_size=4096};
    ds4_engine *engine=NULL; assert(ds4_engine_open(&engine,&options)==0);
    ds4_tokens prompt[2]={{0}};
    ds4_tokenize_text(engine,"Vienna is the capital of",&prompt[0]);
    ds4_tokenize_text(engine,"Seventeen plus twenty five equals",&prompt[1]);
    int vocab=ds4_engine_vocab_size(engine);
    float *reference[2],*capture_reference[2];
    for(int i=0;i<2;i++) {
        reference[i]=malloc(vocab*sizeof(float));
        capture_reference[i]=malloc(DS41_DIRECTION_VALUES*sizeof(float));
        assert(reference[i] && capture_reference[i]);
    }
    float *logits=malloc(vocab*sizeof(float));
    float *capture=malloc(DS41_DIRECTION_VALUES*sizeof(float));
    assert(logits && capture);
    char error[256]={0};
    for(int mode=0;mode<2;mode++) {
        ds4_session *sessions[2]={0};
        ds4_decode_item items[2];
        for(int i=0;i<2;i++) {
            assert(ds4_session_create(&sessions[i],engine,4096)==0);
            assert(ds4_session_v41_configure(sessions[i],argv[i+2],digest,i+1,
                i ? 0.5f : 1.0f,prompt[i].len,(UINT64_C(1)<<40)-1)==0);
            assert(ds4_session_sync(sessions[i],&prompt[i],error,sizeof(error))==0);
            items[i]=(ds4_decode_item){.session=sessions[i],.token=prompt[i].v[0]};
            if(!mode) {
                assert(ds4_session_eval(sessions[i],items[i].token,error,sizeof(error))==0);
                ds4_v41_capture_info info;
                assert(ds4_session_v41_capture(sessions[i],capture_reference[i],DS41_DIRECTION_VALUES,&info)==0);
                assert(ds4_session_copy_logits(sessions[i],reference[i],vocab)==vocab);
                ds4_session_free(sessions[i]); sessions[i]=NULL;
            }
        }
        if(mode) {
            assert(ds4_sessions_eval_batch(items,2,error,sizeof(error))==0);
            for(int i=0;i<2;i++) {
                ds4_v41_capture_info info;
                assert(ds4_session_v41_capture(sessions[i],capture,DS41_DIRECTION_VALUES,&info)==0);
                assert(info.site==(uint32_t)i+1 && info.position==(uint32_t)prompt[i].len);
                assert(ds4_session_copy_logits(sessions[i],logits,vocab)==vocab);
                assert(memcmp(reference[i],logits,vocab*sizeof(float))==0);
                assert(memcmp(capture_reference[i],capture,DS41_DIRECTION_VALUES*sizeof(float))==0);
                ds4_session_free(sessions[i]);
            }
        }
    }
    for(int i=0;i<2;i++) {ds4_tokens_free(&prompt[i]);free(reference[i]);free(capture_reference[i]);}
    free(logits);free(capture);ds4_engine_close(engine);
    puts("V4.1 SSD serial-batch fallback matches isolated scalar sessions, both sites, logits and captures");
}
