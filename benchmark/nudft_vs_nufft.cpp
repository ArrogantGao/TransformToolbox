#include <benchmark/benchmark.h>
#include <finufft.h>
#include <complex>
#include <vector>
#include <random>
#include <fstream>
#include "transtb.hpp"

using namespace std;

static void GenerateData(int M, int N,
                         vector<float> &x, vector<float> &y, vector<float> &z,
                         vector<complex<float>> &c, vector<complex<float>> &coeffs) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-1, 1);

    x.resize(M);
    y.resize(M);
    z.resize(M);
    c.resize(M);
    coeffs.resize(N * N * N);

    for (int i = 0; i < M; i++) {
        x[i] = dis(gen);
        y[i] = dis(gen);
        z[i] = dis(gen);
        c[i] = complex<float>(dis(gen), dis(gen));
    }
}

static void BM_Finufft_Setpts(benchmark::State &state) {
    int M = state.range(0);
    int N = state.range(1);

    vector<float> x, y, z;
    vector<complex<float>> c, coeffs;
    GenerateData(M, N, x, y, z, c, coeffs);

    int64_t nmodes[3] = {N, N, N};
    finufftf_plan p;
    finufft_opts opts;
    finufftf_default_opts(&opts);
    opts.nthreads = 1;
    opts.spread_sort = 0;
    finufftf_makeplan(1, 3, nmodes, 1, 1, 1e-4, &p, &opts);

    for (auto _ : state) {
        finufftf_setpts(p, M, x.data(), y.data(), z.data(), 0, NULL, NULL, NULL);
    }

    finufftf_destroy(p);
}

static void BM_Finufft_Execute(benchmark::State &state) {
    int M = state.range(0);
    int N = state.range(1);

    vector<float> x, y, z;
    vector<complex<float>> c, coeffs;
    GenerateData(M, N, x, y, z, c, coeffs);

    int64_t nmodes[3] = {N, N, N};
    finufftf_plan p;
    finufft_opts opts;
    finufftf_default_opts(&opts);
    opts.nthreads = 1;
    opts.spread_sort = 0;
    finufftf_makeplan(1, 3, nmodes, 1, 1, 1e-4, &p, &opts);

    // 先固定采样点
    finufftf_setpts(p, M, x.data(), y.data(), z.data(), 0, NULL, NULL, NULL);

    for (auto _ : state) {
        finufftf_execute(p, c.data(), coeffs.data());
    }

    finufftf_destroy(p);
}

static void BM_NUDFT3D1(benchmark::State &state) {
    int M = state.range(0);
    int N = state.range(1);

    vector<float> x, y, z;
    vector<complex<float>> c, coeffs;
    GenerateData(M, N, x, y, z, c, coeffs);

    for (auto _ : state) {
        transtb::nudft3d1(M, x.data(), y.data(), z.data(), c.data(), 1, N, N, N, coeffs.data());
    }
}

#define ARGSETS ->ArgsProduct({{1, 8, 64, 512, 4096, 32768}, {8, 16, 32, 64}})->Unit(benchmark::kMicrosecond)

BENCHMARK(BM_Finufft_Setpts) ARGSETS;
BENCHMARK(BM_Finufft_Execute) ARGSETS;
BENCHMARK(BM_NUDFT3D1) ARGSETS;

BENCHMARK_MAIN();