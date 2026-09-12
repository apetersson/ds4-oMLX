#include "../ds4_v41_intervention.h"
#include <assert.h>

static void le32(unsigned char *p, uint32_t v) {
    for (int j = 0; j < 4; j++) p[j] = (unsigned char)(v >> (8*j));
}
static int parse(unsigned char *data, size_t size, ds41_direction *d) {
    unsigned char digest[32] = {1};
    const char *error = NULL;
    FILE *f = tmpfile();
    assert(f && fwrite(data, 1, size, f) == size);
    rewind(f);
    int result = ds41_direction_read(f, digest, d, &error);
    fclose(f);
    assert(result ? !error : error != NULL);
    if (!result) assert(d->layers == 0 && d->values[0] == 0);
    return result;
}
int main(void) {
    const size_t n = DS41_DIRECTION_HEADER + 4 * DS41_DIRECTION_VALUES;
    unsigned char *b = calloc(n + 1, 1);
    ds41_direction *d = malloc(sizeof(*d));
    assert(b && d);
    memcpy(b, "DS41DIR\0", 8);
    le32(b+8, 1); le32(b+12, 40); le32(b+16, 5120); le32(b+20, 1);
    le32(b+24, DS41_SITE_WRITER); le32(b+32, 1); b[40] = 1;
    le32(b+72, 0x3f800000);
    assert(parse(b, n, d));
    assert(d->site == DS41_SITE_WRITER && d->values[0] == 1);
    const unsigned offsets[] = {0, 8, 12, 16, 20, 24, 28, 37, 40};
    for (size_t i = 0; i < sizeof(offsets)/sizeof(offsets[0]); i++) {
        b[offsets[i]] ^= 0x80; assert(!parse(b, n, d)); b[offsets[i]] ^= 0x80;
    }
    le32(b+32, 0); assert(!parse(b,n,d)); le32(b+32,1);
    const uint32_t invalid[] = {0, 0x40000000, 0x7f800000, 0xff800000, 0x7fc00001};
    for (size_t i=0; i<sizeof(invalid)/sizeof(invalid[0]); i++) {
        le32(b+72,invalid[i]); assert(!parse(b,n,d));
    }
    le32(b+72,0x3f800000);
    le32(b+72+5120*4,0x3f800000); assert(!parse(b,n,d)); le32(b+72+5120*4,0);
    assert(!parse(b,n-1,d)); assert(!parse(b,71,d)); assert(!parse(b,n+1,d));
    le32(b+24, DS41_SITE_RESIDUAL_MEAN_EQUAL); assert(parse(b,n,d));
    free(b); free(d);
    puts("V4.1 direction parser fixtures passed");
}
