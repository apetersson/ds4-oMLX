#ifndef DS4_MODEL_HASH_H
#define DS4_MODEL_HASH_H
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <sys/stat.h>
#if defined(__APPLE__)
#include <CommonCrypto/CommonDigest.h>
#endif

/* Read the already-open model, not a path that may now name a different file.
 * Models must remain immutable while in use; reject changes during hashing. */
static inline int ds4_model_fd_sha256(int fd, uint64_t file_size, unsigned char digest[32]) {
#if defined(__APPLE__)
    if (!digest) return 1;
    memset(digest, 0, 32);
    struct stat before, after;
    if (fstat(fd, &before) || !S_ISREG(before.st_mode) ||
        before.st_size < 0 || (uint64_t)before.st_size != file_size) return 1;
    unsigned char *buffer = malloc(1024 * 1024);
    if (!buffer) return 1;
    CC_SHA256_CTX hash;
    CC_SHA256_Init(&hash);
    off_t offset = 0;
    while (offset < before.st_size) {
        size_t count = (size_t)((before.st_size - offset) < 1024 * 1024 ?
                               before.st_size - offset : 1024 * 1024);
        ssize_t n = pread(fd, buffer, count, offset);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { free(buffer); return 1; }
        CC_SHA256_Update(&hash, buffer, (CC_LONG)n);
        offset += n;
    }
    free(buffer);
    if (fstat(fd, &after) || before.st_size != after.st_size ||
        before.st_mtimespec.tv_sec != after.st_mtimespec.tv_sec ||
        before.st_mtimespec.tv_nsec != after.st_mtimespec.tv_nsec ||
        before.st_ctimespec.tv_sec != after.st_ctimespec.tv_sec ||
        before.st_ctimespec.tv_nsec != after.st_ctimespec.tv_nsec) return 1;
    CC_SHA256_Final(digest, &hash);
    return 0;
#else
    (void)fd; (void)file_size;
    if (digest) memset(digest, 0, 32);
    return 1;
#endif
}

#endif
