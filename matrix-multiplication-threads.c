#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <string.h>

typedef enum {
    NT_INT = 1,
    NT_FLOAT = 2,
    NT_MIXED = 3
} NumType;

typedef struct {
    int r;
    int c;
    float *buf;
    float **row;
} Matrix;

typedef struct {
    const Matrix *A;
    const Matrix *B;
    Matrix *C;
    int row_start;
    int row_end;
} Work;

static void line(void) {
    printf("\n--------------------------------------------\n");
}

static void clear_stdin_line(void) {
    int ch;
    while ((ch = getchar()) != '\n' && ch != EOF) {}
}

static double seconds_between(struct timespec a, struct timespec b) {
    return (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9;
}

static Matrix mat_create(int r, int c) {
    Matrix M;
    M.r = r; M.c = c;
    M.buf = (float*)malloc((size_t)r * (size_t)c * sizeof(float));
    if (!M.buf) { fprintf(stderr, "Alloc failed\n"); exit(1); }
    M.row = (float**)malloc((size_t)r * sizeof(float*));
    if (!M.row) { fprintf(stderr, "Alloc failed\n"); exit(1); }
    for (int i = 0; i < r; ++i) M.row[i] = M.buf + (size_t)i * c;
    return M;
}

static void mat_free(Matrix *M) {
    if (!M) return;
    free(M->row);
    free(M->buf);
    M->row = NULL;
    M->buf = NULL;
    M->r = M->c = 0;
}

static void mat_print(const Matrix *M, NumType t) {
    for (int i = 0; i < M->r; ++i) {
        for (int j = 0; j < M->c; ++j) {
            if (t == NT_INT) printf("%4d ", (int)M->row[i][j]);
            else             printf("%7.3f ", M->row[i][j]);
        }
        printf("\n");
    }
}

static Matrix mat_from_file(int r, int c, const char *path) {
    Matrix M = mat_create(r, c);
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Error: cannot open %s\n", path); exit(1); }
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            if (fscanf(f, "%f", &M.row[i][j]) != 1) {
                fprintf(stderr, "Error: cannot read value at (%d,%d) from %s\n", i, j, path);
                fclose(f);
                exit(1);
            }
        }
    }
    fclose(f);
    return M;
}

static void mat_fill_random(Matrix *M, NumType t, float lo, float hi) {
    double range = (double)hi - (double)lo;
    for (int i = 0; i < M->r; ++i) {
        for (int j = 0; j < M->c; ++j) {
            if (t == NT_INT) {
                int irange = (int)(range >= 0 ? (range + 0.999999) : 0);
                int v = (int)lo + (irange > 0 ? rand() % (irange + 1) : 0);
                M->row[i][j] = (float)v;
            } else if (t == NT_FLOAT) {
                M->row[i][j] = (float)((double)lo + (rand() / (double)RAND_MAX) * range);
            } else {
                if (rand() & 1) {
                    int irange = (int)(range >= 0 ? (range + 0.999999) : 0);
                    int v = (int)lo + (irange > 0 ? rand() % (irange + 1) : 0);
                    M->row[i][j] = (float)v;
                } else {
                    M->row[i][j] = (float)((double)lo + (rand() / (double)RAND_MAX) * range);
                }
            }
        }
    }
}

static void mat_mul_single(const Matrix *A, const Matrix *B, Matrix *C) {
    for (int i = 0; i < A->r; ++i) {
        float *c = C->row[i];
        for (int k = 0; k < B->c; ++k) c[k] = 0.0f;
        for (int j = 0; j < A->c; ++j) {
            const float aij = A->row[i][j];
            const float *bcol = &B->row[j][0];
            for (int k = 0; k < B->c; ++k) c[k] += aij * bcol[k];
        }
    }
}

static void *worker_mul(void *arg) {
    Work *w = (Work*)arg;
    const Matrix *A = w->A;
    const Matrix *B = w->B;
    Matrix *C = w->C;
    for (int i = w->row_start; i < w->row_end; ++i) {
        float *c = C->row[i];
        for (int k = 0; k < B->c; ++k) c[k] = 0.0f;
        for (int j = 0; j < A->c; ++j) {
            const float aij = A->row[i][j];
            const float *brow = B->row[j];
            for (int k = 0; k < B->c; ++k) c[k] += aij * brow[k];
        }
    }
    return NULL;
}

static void mat_mul_threads(const Matrix *A, const Matrix *B, Matrix *C, int thread_count) {
    if (thread_count < 1) thread_count = 1;
    if (thread_count > A->r) thread_count = A->r;
    pthread_t *ts = (pthread_t*)malloc((size_t)thread_count * sizeof(pthread_t));
    Work *works = (Work*)malloc((size_t)thread_count * sizeof(Work));
    if (!ts || !works) { fprintf(stderr, "Alloc failed\n"); exit(1); }
    int base = A->r / thread_count;
    int rem  = A->r % thread_count;
    int rs = 0;
    for (int t = 0; t < thread_count; ++t) {
        int cnt = base + (t < rem ? 1 : 0);
        works[t].A = A; works[t].B = B; works[t].C = C;
        works[t].row_start = rs;
        works[t].row_end   = rs + cnt;
        rs += cnt;
        if (pthread_create(&ts[t], NULL, worker_mul, &works[t]) != 0) {
            fprintf(stderr, "Error creating thread %d\n", t);
            exit(1);
        }
    }
    for (int t = 0; t < thread_count; ++t) pthread_join(ts[t], NULL);
    free(ts);
    free(works);
}

static void save_results(
    const char *path,
    const Matrix *A, const Matrix *B,
    const Matrix *C_single, const Matrix *C_multi,
    NumType t,
    double t_single, double t_multi,
    int iters, int threads_used)
{
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Error: cannot create %s\n", path); return; }
    fprintf(f, "Matrix Multiply Report\n");
    fprintf(f, "======================\n\n");
    fprintf(f, "A: %d x %d\n", A->r, A->c);
    fprintf(f, "B: %d x %d\n", B->r, B->c);
    fprintf(f, "C: %d x %d\n\n", A->r, B->c);
    fprintf(f, "Timing (averaged over %d runs):\n", iters);
    fprintf(f, "Single-thread: %.9f s\n", t_single);
    fprintf(f, "Multi-thread (%d threads): %.9f s\n", threads_used, t_multi);
    if (t_multi > 0.0) {
        double speedup = t_single / t_multi;
        fprintf(f, "Speedup = %.3fx\n", speedup);
        fprintf(f, "Improvement = %.2f%%\n\n", (speedup - 1.0) * 100.0);
    } else {
        fprintf(f, "(multi-thread time = 0?)\n\n");
    }
    for (int pass = 0; pass < 4; ++pass) {
        const Matrix *M = NULL;
        const char *title = NULL;
        if (pass == 0) { M = A; title = "Matrix A:"; }
        else if (pass == 1) { M = B; title = "Matrix B:"; }
        else if (pass == 2) { M = C_single; title = "Result (single-thread):"; }
        else { M = C_multi; title = "Result (multi-thread):"; }
        fprintf(f, "%s\n", title);
        for (int i = 0; i < M->r; ++i) {
            for (int j = 0; j < M->c; ++j) {
                if (t == NT_INT) fprintf(f, "%4d ", (int)M->row[i][j]);
                else             fprintf(f, "%7.3f ", M->row[i][j]);
            }
            fprintf(f, "\n");
        }
        fprintf(f, "\n");
    }
    fclose(f);
    printf("\nSaved report to %s\n", path);
}

int main(int argc, char **argv) {
    line();
    printf("         Threaded Matrix Multiplication\n");
    line();
    if (argc != 5) {
        printf("Usage:\n  %s <rowsA> <colsA> <rowsB> <colsB>\n", argv[0]);
        printf("Example:\n  %s 5 4 4 2\n", argv[0]);
        return 1;
    }
    int ra = atoi(argv[1]);
    int ca = atoi(argv[2]);
    int rb = atoi(argv[3]);
    int cb = atoi(argv[4]);
    if (ra <= 0 || ca <= 0 || rb <= 0 || cb <= 0) {
        fprintf(stderr, "Error: all dimensions must be positive\n");
        return 1;
    }
    if (ca != rb) {
        fprintf(stderr, "Error: cols(A) must equal rows(B) (%d != %d)\n", ca, rb);
        return 1;
    }
    printf("\nMatrix dimensions:\n");
    printf("A: %d x %d\nB: %d x %d\n", ra, ca, rb, cb);
    line();
    srand((unsigned)time(NULL));
    Matrix A, B;
    Matrix C1 = mat_create(ra, cb);
    Matrix C2 = mat_create(ra, cb);
    int input_mode = 0;
    printf("Choose input method:\n");
    printf("1) Load A and B from files\n");
    printf("2) Generate random A and B\n");
    printf("Enter 1 or 2: ");
    if (scanf("%d", &input_mode) != 1) { fprintf(stderr, "Bad input\n"); return 1; }
    clear_stdin_line();
    NumType nt = NT_FLOAT;
    float lo = 0.0f, hi = 10.0f;
    if (input_mode == 1) {
        char path[256];
        printf("Path to matrix A (%d x %d): ", ra, ca);
        if (scanf("%255s", path) != 1) { fprintf(stderr, "Bad path\n"); return 1; }
        A = mat_from_file(ra, ca, path);
        printf("Path to matrix B (%d x %d): ", rb, cb);
        if (scanf("%255s", path) != 1) { fprintf(stderr, "Bad path\n"); return 1; }
        B = mat_from_file(rb, cb, path);
    } else if (input_mode == 2) {
        int nt_in;
        line();
        printf("Number type for random fill:\n");
        printf("1) Integers\n2) Floats\n3) Mixed\n");
        printf("Enter 1-3: ");
        if (scanf("%d", &nt_in) != 1) { fprintf(stderr, "Bad input\n"); return 1; }
        clear_stdin_line();
        if (nt_in == 1) nt = NT_INT; else if (nt_in == 2) nt = NT_FLOAT; else nt = NT_MIXED;
        printf("\nEnter range [min, max]:\n");
        printf("min: "); if (scanf("%f", &lo) != 1) { fprintf(stderr, "Bad input\n"); return 1; }
        printf("max: "); if (scanf("%f", &hi) != 1) { fprintf(stderr, "Bad input\n"); return 1; }
        clear_stdin_line();
        printf("\nGenerating random matrices ...\n");
        A = mat_create(ra, ca);  mat_fill_random(&A, nt, lo, hi);
        B = mat_create(rb, cb);  mat_fill_random(&B, nt, lo, hi);
    } else {
        fprintf(stderr, "Invalid choice\n");
        return 1;
    }
    int iterations = 0;
    printf("\nIterations for timing (recommend >= 5): ");
    if (scanf("%d", &iterations) != 1) { fprintf(stderr, "Bad input\n"); return 1; }
    clear_stdin_line();
    if (iterations <= 0) { printf("Non-positive; defaulting to 10.\n"); iterations = 10; }
    int thread_hint = ra;
    printf("Thread count (1-%d, 0 for auto=%d): ", ra, ra);
    if (scanf("%d", &thread_hint) != 1) { fprintf(stderr, "Bad input\n"); return 1; }
    clear_stdin_line();
    if (thread_hint <= 0) thread_hint = ra;
    if (thread_hint > ra) thread_hint = ra;
    mat_mul_single(&A, &B, &C1);
    mat_mul_threads(&A, &B, &C2, thread_hint);
    struct timespec s, e;
    double t_single = 0.0, t_multi = 0.0;
    for (int i = 0; i < iterations; ++i) {
        clock_gettime(CLOCK_MONOTONIC, &s);
        mat_mul_single(&A, &B, &C1);
        clock_gettime(CLOCK_MONOTONIC, &e);
        t_single += seconds_between(s, e);
    }
    t_single /= iterations;
    for (int i = 0; i < iterations; ++i) {
        clock_gettime(CLOCK_MONOTONIC, &s);
        mat_mul_threads(&A, &B, &C2, thread_hint);
        clock_gettime(CLOCK_MONOTONIC, &e);
        t_multi += seconds_between(s, e);
    }
    t_multi /= iterations;
    line();
    printf("                 RESULTS\n");
    line();
    printf("A: %d x %d, B: %d x %d, C: %d x %d\n", ra, ca, rb, cb, ra, cb);
    printf("Averaged over %d runs:\n", iterations);
    printf(" - Single-thread: %.9f s\n", t_single);
    printf(" - Multi-thread (%d threads): %.9f s\n", thread_hint, t_multi);
    if (t_multi > 0.0) {
        double speedup = t_single / t_multi;
        printf("Speedup: %.3fx\n", speedup);
        printf("Improvement: %.2f%%\n", (speedup - 1.0) * 100.0);
    }
    int save_flag = 0;
    printf("\nSave matrices and results to a file? (1=yes, 0=no): ");
    if (scanf("%d", &save_flag) != 1) { fprintf(stderr, "Bad input\n"); return 1; }
    clear_stdin_line();
    if (save_flag == 1) {
        char base[256];
        printf("Output filename (without .txt): ");
        if (scanf("%255s", base) != 1) { fprintf(stderr, "Bad input\n"); return 1; }
        char outpath[300];
        snprintf(outpath, sizeof(outpath), "%s.txt", base);
        save_results(outpath, &A, &B, &C1, &C2, nt, t_single, t_multi, iterations, thread_hint);
    }
    mat_free(&A);
    mat_free(&B);
    mat_free(&C1);
    mat_free(&C2);
    return 0;
}
