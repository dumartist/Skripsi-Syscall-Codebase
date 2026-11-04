/*
 * LD_PRELOAD shim untuk validasi fungsional injeksi syscall.
 *
 * Menyuntik no-op syscall NYATA (getpid, getuid, gettimeofday) ke dalam proses
 * serangan yang sedang berjalan, dengan mencegat connect() (Hydra membuka banyak
 * koneksi selama brute-force). Logika serangan asli TIDAK diubah — hanya
 * ditambahi no-op di sekitar tiap connect(). Ini realisasi problem-space no-op
 * injection pada executable nyata: jika serangan tetap berhasil, ΦF (functional
 * preservation) terbukti secara eksekusional, bukan hanya statistik.
 *
 * No-op yang dipakai memetakan ke set generator ADFA:
 *   getpid (i386 20), getuid (i386 24), gettimeofday (i386 78) — idempoten,
 *   tanpa efek samping, hanya membaca state. Aman untuk disisipkan.
 *
 * Build:
 *   gcc -shared -fPIC -o /tmp/noop_shim.so deploy/validation/noop_shim.c -ldl
 * Pakai:
 *   LD_PRELOAD=/tmp/noop_shim.so NOOP_PER_CALL=5 <program>
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <sys/syscall.h>
#include <sys/socket.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>

static int (*real_connect)(int, const struct sockaddr *, socklen_t) = NULL;
static long g_noops = 0;   /* total no-op yang disuntik (statistik) */

static int noop_per_call(void) {
    const char *e = getenv("NOOP_PER_CALL");
    return e ? atoi(e) : 3;
}

/* Sisipkan n putaran no-op syscall NYATA (idempoten, tanpa efek samping). */
static void inject_noops(int n) {
    struct timeval tv;
    for (int i = 0; i < n; i++) {
        syscall(SYS_getpid);
        syscall(SYS_getuid);
        syscall(SYS_gettimeofday, &tv, NULL);
        g_noops += 3;
    }
}

int connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen) {
    if (!real_connect)
        real_connect = (int (*)(int, const struct sockaddr *, socklen_t))
                       dlsym(RTLD_NEXT, "connect");
    int n = noop_per_call();
    inject_noops(n);                                   /* no-op sebelum */
    int r = real_connect(sockfd, addr, addrlen);       /* aksi serangan ASLI */
    inject_noops(n);                                   /* no-op sesudah */
    return r;
}
