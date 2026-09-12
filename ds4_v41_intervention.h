#ifndef DS4_V41_INTERVENTION_H
#define DS4_V41_INTERVENTION_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Version 1 wire format is little endian: magic[8], version/layers/width/
 * rank/site/mode (u32), layer mask (u64), model SHA256[32], then 40x5120
 * IEEE float32 values. Unselected layers are zero. The digest binds this
 * deployment direction to its actual model; cross-quant transfers need a
 * separately recorded bridge and a new file. Alpha belongs to runtime state,
 * never to the stored unit directions. */
#define DS41_DIRECTION_LAYERS 40u
#define DS41_DIRECTION_WIDTH 5120u
#define DS41_DIRECTION_HEADER 72u
#define DS41_DIRECTION_VALUES (DS41_DIRECTION_LAYERS * DS41_DIRECTION_WIDTH)
enum ds41_intervention_site {
    DS41_SITE_WRITER = 1,
    /* Independently defined: remove mean-HC component equally from all four
     * streams after layer rounding, then round the edited result to BF16 in
     * every path (including compact carry). This is not GLP equivalence. */
    DS41_SITE_RESIDUAL_MEAN_EQUAL = 2
};
typedef struct {
    uint32_t site;
    uint64_t layers;
    unsigned char model_sha256[32];
    float values[DS41_DIRECTION_VALUES];
} ds41_direction;

static inline uint32_t ds41_read_le32(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 |
           (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
/* isfinite can be folded to true under the runtime's -ffast-math. */
static inline int ds41_bits_finite(uint32_t bits) {
    return (bits & 0x7f800000u) != 0x7f800000u;
}

static inline int ds41_direction_read(FILE *fp, const unsigned char expected_sha256[32],
                                      ds41_direction *out, const char **error) {
    unsigned char h[DS41_DIRECTION_HEADER], word[4];
    const char *why = NULL;
    if (!fp || !expected_sha256 || !out) { why = "missing direction input"; goto fail; }
    if (fread(h, 1, sizeof(h), fp) != sizeof(h)) { why = "truncated direction header"; goto fail; }
    if (memcmp(h, "DS41DIR\0", 8) || ds41_read_le32(h + 8) != 1 ||
        ds41_read_le32(h + 12) != DS41_DIRECTION_LAYERS ||
        ds41_read_le32(h + 16) != DS41_DIRECTION_WIDTH ||
        ds41_read_le32(h + 20) != 1) { why = "unsupported direction model/shape/rank/version"; goto fail; }
    uint32_t site = ds41_read_le32(h + 24);
    if ((site != DS41_SITE_WRITER && site != DS41_SITE_RESIDUAL_MEAN_EQUAL) ||
        ds41_read_le32(h + 28) != 0) { why = "unsupported direction site/mode"; goto fail; }
    uint64_t layers = ds41_read_le32(h + 32) | (uint64_t)ds41_read_le32(h + 36) << 32;
    if (!layers || layers >> DS41_DIRECTION_LAYERS) { why = "invalid direction layer mask"; goto fail; }
    if (memcmp(h + 40, expected_sha256, 32)) { why = "direction model digest mismatch"; goto fail; }
    for (uint32_t layer = 0; layer < DS41_DIRECTION_LAYERS; layer++) {
        double norm = 0;
        for (uint32_t j = 0; j < DS41_DIRECTION_WIDTH; j++) {
            if (fread(word, 1, 4, fp) != 4) { why = "truncated direction payload"; goto fail; }
            uint32_t bits = ds41_read_le32(word);
            float value;
            memcpy(&value, &bits, 4);
            if (!ds41_bits_finite(bits)) { why = "nonfinite direction"; goto fail; }
            if (!(layers & (UINT64_C(1) << layer)) && value != 0) { why = "nonzero unselected layer"; goto fail; }
            out->values[layer * DS41_DIRECTION_WIDTH + j] = value;
            norm += (double)value * value;
        }
        if ((layers & (UINT64_C(1) << layer)) && fabs(norm - 1.0) > 1e-4) {
            why = "direction is not unit length"; goto fail;
        }
    }
    if (fgetc(fp) != EOF || ferror(fp)) { why = "extra direction bytes or read failure"; goto fail; }
    out->site = site;
    out->layers = layers;
    memcpy(out->model_sha256, h + 40, 32);
    if (error) *error = NULL;
    return 1;
fail:
    /* A rejected partial payload must never remain usable by a caller. */
    if (out) memset(out, 0, sizeof(*out));
    if (error) *error = why;
    return 0;
}
#endif
