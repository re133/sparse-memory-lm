/* Row loops over host-resident value tables (smlm/host_optim.py, smlm/host_values.py), split over plain
 * pthreads so they don't compete with PyTorch's OpenMP runtimes. Built on first use with
 * gcc -O3 -ffp-contract=off -fno-math-errno (flags in host_optim.FLAGS): the loops vectorize, and results don't
 * depend on the vector width.
 *
 *   host_rows_adam    one pass per touched row: Adam on values and moments, accumulator zeroed, row untouched
 *   host_rows_add     acc[rows[i]] += src[i], touched[rows[i]] = 1 (rows unique within one call)
 *   host_rows_gather  out[i] = table[rows[i]]
 */
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <string.h>

#define MAX_THREADS 256

typedef struct {
    float *p, *acc, *m, *v, *out;
    const float *src, *table;
    const int64_t *rows;
    uint8_t *touched;
    int64_t width;
    float grad_scale, one_minus_b1, b2, one_minus_b2, bc2_sqrt, eps, neg_step_size;
} job_t;

typedef struct {
    void (*fn)(const job_t *, int64_t, int64_t);
    const job_t *job;
    int64_t begin, end;
} part_t;

static void *run_part(void *arg) {
    part_t *part = arg;
    part->fn(part->job, part->begin, part->end);
    return NULL;
}

static void run(void (*fn)(const job_t *, int64_t, int64_t), const job_t *job, int64_t n, int threads) {
    if (threads < 1) threads = 1;
    if (threads > MAX_THREADS) threads = MAX_THREADS;
    if (n < 64 * (int64_t)threads) threads = n < 64 ? 1 : (int)(n / 64);
    part_t parts[MAX_THREADS];
    pthread_t ids[MAX_THREADS];
    int started[MAX_THREADS];
    for (int t = 0; t < threads; t++) {
        parts[t] = (part_t){fn, job, n * t / threads, n * (t + 1) / threads};
        started[t] = t > 0 && pthread_create(&ids[t], NULL, run_part, &parts[t]) == 0;
    }
    run_part(&parts[0]);
    for (int t = 1; t < threads; t++) {
        if (started[t]) pthread_join(ids[t], NULL);
        else run_part(&parts[t]);     /* thread creation failed: do that share here */
    }
}

/* Same operation order as the torch path in HostLazyRowAdam.step (lerp, mul + addcmul, sqrt / bc2 + eps,
 * addcdiv), all in fp32. */
static void adam_part(const job_t *j, int64_t begin, int64_t end) {
    const int64_t d = j->width;
    for (int64_t i = begin; i < end; i++) {
        const int64_t r = j->rows[i];
        float *p = j->p + r * d, *g = j->acc + r * d, *m = j->m + r * d, *v = j->v + r * d;
        for (int64_t k = 0; k < d; k++) {
            const float gk = g[k] * j->grad_scale;
            const float mk = m[k] + j->one_minus_b1 * (gk - m[k]);
            const float vk = v[k] * j->b2 + j->one_minus_b2 * gk * gk;
            const float denom = sqrtf(vk) / j->bc2_sqrt + j->eps;
            p[k] = p[k] + j->neg_step_size * (mk / denom);
            m[k] = mk;
            v[k] = vk;
            g[k] = 0.0f;
        }
        j->touched[r] = 0;
    }
}

static void add_part(const job_t *j, int64_t begin, int64_t end) {
    const int64_t d = j->width;
    for (int64_t i = begin; i < end; i++) {
        const int64_t r = j->rows[i];
        float *a = j->acc + r * d;
        const float *s = j->src + i * d;
        for (int64_t k = 0; k < d; k++) a[k] += s[k];
        j->touched[r] = 1;
    }
}

static void gather_part(const job_t *j, int64_t begin, int64_t end) {
    const int64_t d = j->width;
    for (int64_t i = begin; i < end; i++)
        memcpy(j->out + i * d, j->table + j->rows[i] * d, (size_t)d * sizeof(float));
}

void host_rows_adam(float *p, float *acc, float *m, float *v, uint8_t *touched, const int64_t *rows, int64_t n,
                    int64_t width, float grad_scale, float one_minus_b1, float b2, float one_minus_b2,
                    float bc2_sqrt, float eps, float neg_step_size, int threads) {
    job_t j = {.p = p, .acc = acc, .m = m, .v = v, .touched = touched, .rows = rows, .width = width,
               .grad_scale = grad_scale, .one_minus_b1 = one_minus_b1, .b2 = b2, .one_minus_b2 = one_minus_b2,
               .bc2_sqrt = bc2_sqrt, .eps = eps, .neg_step_size = neg_step_size};
    run(adam_part, &j, n, threads);
}

void host_rows_add(float *acc, uint8_t *touched, const int64_t *rows, const float *src, int64_t n, int64_t width,
                   int threads) {
    job_t j = {.acc = acc, .touched = touched, .rows = rows, .src = src, .width = width};
    run(add_part, &j, n, threads);
}

void host_rows_gather(float *out, const float *table, const int64_t *rows, int64_t n, int64_t width, int threads) {
    job_t j = {.out = out, .table = table, .rows = rows, .width = width};
    run(gather_part, &j, n, threads);
}
