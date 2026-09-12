/* Exercise real CLI parsing and the shared session factory without a model. */
#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#define ds4_engine_model_sha256 fixture_hash
#define ds4_engine_is_deepseek41 fixture_is_v41
#define ds4_session_create fixture_create
#define ds4_session_free fixture_free
#define ds4_session_v41_configure fixture_configure
#include "../ds4_server.c"

#include <assert.h>
#include <sys/wait.h>

static int creates, configures, frees, fail_configure;
static const server_config *expected;
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
static int hashes, hash_failure, wrong_digest, wrong_model;
int fixture_hash(ds4_engine *e, unsigned char digest[32]) {
    (void)e; hashes++; memset(digest, 0, 32); digest[0] = wrong_digest; return hash_failure;
}
bool fixture_is_v41(ds4_engine *e) { (void)e; return !wrong_model; }
static void parse_case(char **args, int count, int want, int legacy) {
    pid_t pid = fork(); assert(pid >= 0);
    if (!pid) {
        server_config cfg = parse_options(count, args);
        if (server_prepare_v41(&cfg)) _exit(2);
        if (legacy) {
            assert(!cfg.v41_direction_file);
            assert(cfg.engine.directional_steering_ffn == (legacy == 1 ? 1 : 0));
            assert(cfg.engine.directional_steering_attn == (legacy == 1 ? 0 : 2));
        } else {
            assert(cfg.v41_direction_file && !cfg.engine.directional_steering_file);
            assert(cfg.engine.directional_steering_ffn == 0);
        }
        _exit(0);
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
    char path[] = "/tmp/ds41-server-XXXXXX";
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
    char *base[] = {"ds4-server", "--dir-steering-file",path,"--dir-steering-strength","0"};
    parse_case(base,5,0,0); parse_case(base,3,2,0);
    char *bad[] = {"ds4-server","--dir-steering-file",path,"--dir-steering-strength","nan"};
    parse_case(bad,5,2,0);
    const char *numbers[] = {"inf","-inf","1e99","101","-101","wat",""};
    for (unsigned i=0;i<sizeof(numbers)/sizeof(*numbers);i++) {bad[4]=(char*)numbers[i];parse_case(bad,5,2,0);}
    const char *options[] = {"--cpu","--mtp"};
    for (unsigned i=0;i<sizeof(options)/sizeof(*options);i++) {
        char *args[] = {"ds4-server","--dir-steering-file",path,"--dir-steering-strength","1",(char*)options[i]};
        parse_case(args,6,2,0);
    }
    char *mixed[] = {"ds4-server","--dir-steering-file",path,"--dir-steering-strength","1","--dir-steering-ffn","0"};
    parse_case(mixed,7,2,0);
    char *missing[] = {"ds4-server","--dir-steering-strength","1"}; parse_case(missing,3,2,0);
    char *disk[] = {"ds4-server","--dir-steering-file",path,"--dir-steering-strength","0","--kv-disk-dir","cache"};
    parse_case(disk,7,2,0);
    server_config verified=parse_options(5,base);
    assert(server_prepare_v41(&verified)==0);
    assert(server_verify_v41(&verified,NULL)==0 && hashes==1);
    wrong_digest=1; assert(server_verify_v41(&verified,NULL)!=0); wrong_digest=0;
    hash_failure=1; assert(server_verify_v41(&verified,NULL)!=0); hash_failure=0;
    wrong_model=1; assert(server_verify_v41(&verified,NULL)!=0); wrong_model=0;
    /* Both sites pass; corrupt rank is rejected before engine/model startup. */
    file[24]=2; fp=fopen(path,"wb");fwrite(file,1,DS41_DIRECTION_HEADER+DS41_DIRECTION_VALUES*4,fp);fclose(fp);
    parse_case(base,5,0,0);
    file[20]=2; fp=fopen(path,"wb");fwrite(file,1,DS41_DIRECTION_HEADER+DS41_DIRECTION_VALUES*4,fp);fclose(fp);
    parse_case(base,5,2,0);
    fp=fopen(path,"wb");fwrite("legacy raw",1,10,fp);fclose(fp);
    parse_case(base,5,2,0); parse_case(base,3,0,1);
    char *legacy[] = {"ds4-server","--dir-steering-file",path,"--dir-steering-attn","2"}; parse_case(legacy,5,0,2);
    for (uint32_t site=1;site<=2;site++) {
        server_config cfg={.v41_direction_file=path,.v41_site=site,.v41_strength=0.75f};
        cfg.v41_digest[0]=123; expected=&cfg;
        cfg.ctx_size=256;
        int before=configures;
        for (int slot=0;slot<3;slot++) {
            ds4_session *session=NULL;
            assert(server_create_session(&session,NULL,&cfg)==0);
            fixture_free(session);
        }
        assert(configures==before+3);
        fail_configure=1; ds4_session *session=NULL;
        assert(server_create_session(&session,NULL,&cfg)!=0 && !session);
        fail_configure=0;
    }
    assert(creates==frees);
    unlink(path);free(file); puts("V4.1 server parsing, model binding and slot lifecycle: OK");
}
