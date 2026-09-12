/* Exercise real CLI parsing and the shared session factory without a model. */
#define main ds4_cli_main
#define ds4_session_create fixture_create
#define ds4_session_free fixture_free
#define ds4_session_v41_configure fixture_configure
#include "../ds4_cli.c"
#undef main
#include <assert.h>
#include <sys/wait.h>

static int creates, configures, frees, fail_configure;
static const cli_config *expected;
int fixture_create(ds4_session **out, ds4_engine *e, int ctx) {
    (void)e; assert(ctx == 256 || ctx == 512);
    creates++; *out = (ds4_session *)(uintptr_t)creates; return 0;
}
void fixture_free(ds4_session *s) { if (s) frees++; }
int fixture_configure(ds4_session *s, const char *path, const unsigned char *sha,
                      uint32_t site, float alpha, uint32_t pos, uint64_t layers) {
    assert(s && expected && !strcmp(path, expected->v41_direction_file));
    assert(!memcmp(sha, expected->v41_digest, 32));
    assert(site == expected->v41_site && alpha == expected->v41_strength);
    assert(pos == 0 && layers == 0); configures++; return fail_configure;
}
static void cleanup(cli_config *cfg) {
    ds4_dist_options_free(cfg->dist); ds4_prompt_prefix_free(&cfg->gen.prefix);
    free(cfg->prompt_owned);
}
static void parse_case(char **args, int count, int want, int legacy) {
    pid_t pid = fork(); assert(pid >= 0);
    if (!pid) {
        cli_config cfg = parse_options(count, args);
        if (legacy) {
            assert(!cfg.v41_direction_file);
            assert(cfg.engine.directional_steering_ffn == (legacy == 1 ? 1 : 0));
            assert(cfg.engine.directional_steering_attn == (legacy == 1 ? 0 : 2));
        } else {
            assert(cfg.v41_direction_file && !cfg.engine.directional_steering_file);
            assert(cfg.engine.directional_steering_ffn == 0);
        }
        cleanup(&cfg); _exit(0);
    }
    int status; assert(waitpid(pid, &status, 0) == pid);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == want);
}
int main(void) {
    for (ds4_help_tool tool=DS4_HELP_DS4;tool<=DS4_HELP_AGENT;tool++) {
        FILE *help=tmpfile(); assert(help);
        ds4_help_print(help,tool,"steering"); rewind(help);
        char text[8192]={0}; fread(text,1,sizeof(text)-1,help); fclose(help);
        assert((strstr(text,"--dir-steering-strength")!=NULL)==(tool==DS4_HELP_DS4 || tool==DS4_HELP_SERVER));
    }
    char path[] = "/tmp/ds41-cli-XXXXXX";
    int fd = mkstemp(path); assert(fd >= 0);
    FILE *fp = fdopen(fd, "wb"); assert(fp);
    unsigned char *file = calloc(1, DS41_DIRECTION_HEADER + DS41_DIRECTION_VALUES * 4);
    assert(file);
    memcpy(file, "DS41DIR\0", 8);
    const uint32_t words[] = {1,40,5120,1,1,0,1,0};
    for (unsigned i=0;i<8;i++) for (unsigned j=0;j<4;j++) file[8+4*i+j] = words[i]>>(8*j);
    file[74] = 0x80; file[75] = 0x3f;
    assert(fwrite(file,1,DS41_DIRECTION_HEADER+DS41_DIRECTION_VALUES*4,fp) == DS41_DIRECTION_HEADER+DS41_DIRECTION_VALUES*4);
    fclose(fp);
    char *base[] = {"ds4", "--dir-steering-file",path,"--dir-steering-strength","0"};
    parse_case(base,5,0,0); parse_case(base,3,2,0);
    char *bad[] = {"ds4","--dir-steering-file",path,"--dir-steering-strength","nan"};
    parse_case(bad,5,2,0);
    const char *numbers[] = {"inf","-inf","1e99","101","-101","wat",""};
    for (unsigned i=0;i<sizeof(numbers)/sizeof(*numbers);i++) {bad[4]=(char*)numbers[i];parse_case(bad,5,2,0);}
    const char *options[] = {"--cpu","--inspect","--head-test","--metal-graph-test","--dump-tokens","--mtp"};
    for (unsigned i=0;i<sizeof(options)/sizeof(*options);i++) {
        char *args[] = {"ds4","--dir-steering-file",path,"--dir-steering-strength","1",(char*)options[i]};
        parse_case(args,6,2,0);
    }
    char *mixed[] = {"ds4","--dir-steering-file",path,"--dir-steering-strength","1","--dir-steering-ffn","0"};
    parse_case(mixed,7,2,0);
    char *missing[] = {"ds4","--dir-steering-strength","1"}; parse_case(missing,3,2,0);
    /* Both sites pass; corrupt rank is rejected before engine/model startup. */
    file[24]=2; fp=fopen(path,"wb");fwrite(file,1,DS41_DIRECTION_HEADER+DS41_DIRECTION_VALUES*4,fp);fclose(fp);
    parse_case(base,5,0,0);
    file[20]=2; fp=fopen(path,"wb");fwrite(file,1,DS41_DIRECTION_HEADER+DS41_DIRECTION_VALUES*4,fp);fclose(fp);
    parse_case(base,5,2,0);
    fp=fopen(path,"wb");fwrite("legacy raw",1,10,fp);fclose(fp);
    parse_case(base,5,2,0); parse_case(base,3,0,1);
    char *legacy[] = {"ds4","--dir-steering-file",path,"--dir-steering-attn","2"}; parse_case(legacy,5,0,2);
    for (uint32_t site=1;site<=2;site++) {
        cli_config cfg={.v41_direction_file=path,.v41_site=site,.v41_strength=0.75f};
        cfg.v41_digest[0]=123; expected=&cfg;
        repl_chat chat={0}; int before=configures;
        assert(repl_chat_create_session(NULL,&chat,256,&cfg)==0);
        assert(configures==before+1);
        assert(repl_chat_set_ctx(NULL,&chat,512,&cfg)==0);
        assert(configures==before+2);
        fixture_free(chat.session);
        fail_configure=1; ds4_session *s=NULL;
        assert(cli_session_create(&s,NULL,256,&cfg)!=0 && !s);
        fail_configure=0;
    }
    assert(creates==frees);
    unlink(path);free(file); puts("V4.1 CLI parsing and session lifecycle: OK");
}
