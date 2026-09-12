/* Real SSD regression of the CLI's interactive session creation/reset paths.
 * Usage: tests/test_v41_cli_live MODEL writer.dir residual.dir */
#define main ds4_cli_main
#include "../ds4_cli.c"
#undef main
#include <assert.h>

int main(int argc, char **argv) {
    assert(argc == 4);
    cli_config cfg = {.engine = {.model_path=argv[1], .backend=DS4_BACKEND_METAL,
        .ssd_streaming=true, .context_size=256}, .gen = {.ctx_size=256,
        .prompt="Vienna is the capital of", .raw_prompt=true, .n_predict=4,
        .temperature=0, .seed=1, .think_mode=DS4_THINK_NONE}};
    ds4_engine *engine = NULL;
    assert(ds4_engine_open(&engine,&cfg.engine)==0);
    assert(ds4_engine_model_sha256(engine,cfg.v41_digest)==0);
    ds4_tokens prompt={0};
    ds4_tokenize_text(engine,"Vienna is the capital of",&prompt);
    int vocab=ds4_engine_vocab_size(engine);
    float *base=malloc(vocab*sizeof(float)), *reference=malloc(vocab*sizeof(float));
    float *logits=malloc(vocab*sizeof(float));assert(base && reference && logits);
    char error[256]={0};
    ds4_session *stock=NULL;
    assert(cli_session_create(&stock,engine,256,&cfg)==0);
    assert(ds4_session_sync(stock,&prompt,error,sizeof(error))==0);
    assert(ds4_session_copy_logits(stock,base,vocab)==vocab);
    ds4_session_free(stock);
    for(uint32_t site=1;site<=2;site++) for(int alpha=0;alpha<=1;alpha++) {
        cfg.v41_direction_file=argv[site+1];cfg.v41_site=site;cfg.v41_strength=(float)alpha;
        repl_chat chat={0};
        assert(repl_chat_create_session(engine,&chat,256,&cfg)==0);
        assert(ds4_session_sync(chat.session,&prompt,error,sizeof(error))==0);
        assert(ds4_session_copy_logits(chat.session,reference,vocab)==vocab);
        double delta=0;
        for(int i=0;i<vocab;i++) if(fabs((double)reference[i]-base[i])>delta)
            delta=fabs((double)reference[i]-base[i]);
        assert(alpha ? delta>0 : memcmp(base,reference,vocab*sizeof(float))==0);
        ds4_session_invalidate(chat.session);
        assert(ds4_session_sync(chat.session,&prompt,error,sizeof(error))==0);
        assert(ds4_session_copy_logits(chat.session,logits,vocab)==vocab);
        assert(memcmp(reference,logits,vocab*sizeof(float))==0);
        assert(repl_chat_set_ctx(engine,&chat,512,&cfg)==0);
        assert(ds4_session_sync(chat.session,&prompt,error,sizeof(error))==0);
        assert(ds4_session_copy_logits(chat.session,logits,vocab)==vocab);
        assert(memcmp(reference,logits,vocab*sizeof(float))==0);
        repl_chat_free(&chat);
        printf("CLI lifecycle site=%u alpha=%d max_logit_delta=%.9g invalidate/recreate=exact\n",site,alpha,delta);
        fflush(stdout);
        if (alpha) {
            /* Also exercise the ordinary prompt dispatcher and nonzero decode
             * after the reset checks, without another full-model hash. */
            assert(run_generation(engine,&cfg)==0);
            printf("CLI nonzero ordinary generation site=%u: OK\n",site);
            fflush(stdout);
        }
    }
    ds4_tokens_free(&prompt);free(base);free(reference);free(logits);
    ds4_engine_close(engine);
    puts("V4.1 CLI live lifecycle: OK");
}
