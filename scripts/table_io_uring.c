/* Persistent read-only io_uring helper, compiled on first use by smlm.table_io.
 * gcc -O3 -std=c11 -Wall -Wextra -Werror -shared -fPIC table_io_uring.c -o table_io_uring.so -luring
 */
#define _GNU_SOURCE
#include <errno.h>
#include <limits.h>
#include <liburing.h>
#include <sched.h>
#include <stdint.h>
#include <stdlib.h>

struct table_io {
    struct io_uring ring;
    unsigned depth;
    int live;
    uint64_t *retry;
};

int table_io_open(unsigned depth, void **result)
{
    struct table_io *ctx = calloc(1, sizeof(*ctx));
    if (!ctx)
        return -ENOMEM;
    ctx->retry = calloc(depth, sizeof(*ctx->retry));
    if (!ctx->retry) {
        free(ctx);
        return -ENOMEM;
    }
    int rc = io_uring_queue_init(depth, &ctx->ring, 0);
    if (rc < 0) {
        free(ctx->retry);
        free(ctx);
        return rc;
    }
    ctx->depth = depth;
    ctx->live = 1;
    *result = ctx;
    return 0;
}

int table_io_read(void *handle, int fd, void *buffer, const uint64_t *offsets,
                  const uint64_t *lengths, const uint64_t *targets, size_t count,
                  uint64_t file_size)
{
    struct table_io *ctx = handle;
    if (!ctx || !ctx->live)
        return -EBADF;
    for (size_t i = 0; i < count; i++) {
        if (!lengths[i] || lengths[i] > INT_MAX || offsets[i] >= file_size)
            return -EINVAL;
    }
    size_t next = 0;
    unsigned pending = 0, inflight = 0, retries = 0;
    int error = 0, broken = 0;
    while (next < count || retries || pending || inflight) {
        while (!error && pending + inflight < ctx->depth && (next < count || retries)) {
            struct io_uring_sqe *sqe = io_uring_get_sqe(&ctx->ring);
            if (!sqe)
                break;
            uint64_t index = retries ? ctx->retry[--retries] : next++;
            io_uring_prep_read(sqe, fd, (char *)buffer + targets[index],
                               (unsigned)lengths[index], offsets[index]);
            io_uring_sqe_set_data64(sqe, index);
            pending++;
        }
        while (pending && !error) {
            int rc = io_uring_submit(&ctx->ring);
            if (rc == -EINTR)
                continue;
            if (rc <= 0) {
                error = rc < 0 ? rc : -EIO;
                broken = 1;
                break;
            }
            pending -= (unsigned)rc;
            inflight += (unsigned)rc;
        }
        if (!inflight)
            break;

        struct io_uring_cqe *cqe = NULL;
        int rc;
        do {
            rc = io_uring_wait_cqe(&ctx->ring, &cqe);
        } while (rc == -EINTR);
        if (rc < 0) {
            if (!error)
                error = rc;
            broken = 1;
            /* Submitted direct reads own the buffer until their CQEs arrive. Even a
             * wait-syscall failure must not return a still-active buffer to Python. */
            while (io_uring_peek_cqe(&ctx->ring, &cqe) < 0)
                sched_yield();
        }
        do {
            uint64_t index = io_uring_cqe_get_data64(cqe);
            int result = cqe->res;
            uint64_t expected = lengths[index];
            if (expected > file_size - offsets[index])
                expected = file_size - offsets[index];
            io_uring_cqe_seen(&ctx->ring, cqe);
            inflight--;
            if (result == -EINTR && !error) {
                ctx->retry[retries++] = index;
            } else if (result < 0 || (uint64_t)result != expected) {
                if (!error)
                    error = result < 0 ? result : -EIO;
            }
        } while (inflight && io_uring_peek_cqe(&ctx->ring, &cqe) == 0);
        if (error && !inflight)
            break;
    }
    if (broken) {
        /* No SQPOLL: unsubmitted SQEs cannot start after all submitted reads drained. */
        io_uring_queue_exit(&ctx->ring);
        ctx->live = 0;
    }
    return error;
}

void table_io_close(void *handle)
{
    struct table_io *ctx = handle;
    if (!ctx)
        return;
    if (ctx->live)
        io_uring_queue_exit(&ctx->ring);
    free(ctx->retry);
    free(ctx);
}
