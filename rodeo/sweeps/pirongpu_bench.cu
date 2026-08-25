// Benchmark driver for PIRonGPU: striped large records, timed phases.
// Usage: bench_pir <items_per_stripe> <stripe_item_bytes> <num_stripes> <trials> [batch]
// A record = num_stripes * stripe_item_bytes; one query serves all stripes.
#include "pir.cuh"
#include "pir_client.cuh"
#include "pir_server.cuh"
#include <omp.h>
#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

using namespace pirongpu;
using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

int main(int argc, char* argv[])
{
    cudaSetDevice(0);
    uint64_t number_of_items = argc > 1 ? std::stoull(argv[1]) : (1ULL << 16);
    uint64_t size_per_item   = argc > 2 ? std::stoull(argv[2]) : 288;
    int num_stripes          = argc > 3 ? std::stoi(argv[3]) : 1;
    int trials               = argc > 4 ? std::stoi(argv[4]) : 5;
    int batch                = argc > 5 ? std::stoi(argv[5]) : 1;

    uint32_t N = 1 << 12;
    uint32_t d = 2;
    std::vector<int> logQ = {36, 36};
    std::vector<int> logP = {37};
    int plain_modulus = 1179649;

    auto context = std::make_shared<heongpu::Parameters>(
        heongpu::scheme_type::bfv, heongpu::keyswitching_type::KEYSWITCHING_METHOD_I);
    context->set_poly_modulus_degree(N);
    context->set_coeff_modulus(logQ, logP);
    context->set_plain_modulus(plain_modulus);
    context->generate();

    PirParams pir_params;
    gen_pir_params(number_of_items, size_per_item, d, *context, pir_params,
                   false, true, false);

    PIRClient client(context, pir_params);
    heongpu::Galoiskey galois_keys = client.generate_galois_keys();

    std::random_device rd;
    uint64_t ele = rd() % number_of_items;
    uint64_t fv_index = client.get_fv_index(ele);
    uint64_t fv_offset = client.get_fv_offset(ele);

    auto tq0 = clk::now();
    PirQuery query = client.generate_query(fv_index);
    auto tq1 = clk::now();

    // one server per stripe; same query reused (query encodes only the index)
    std::vector<std::unique_ptr<PIRServer>> servers;
    std::vector<std::unique_ptr<uint8_t[]>> db_copies;
    auto tp0 = clk::now();
    for (int s = 0; s < num_stripes; s++) {
        auto db = std::make_unique<uint8_t[]>(number_of_items * size_per_item);
        auto keep = std::make_unique<uint8_t[]>(number_of_items * size_per_item);
        std::mt19937 gen(1234 + s);
        std::uniform_int_distribution<int> dis(0, 255);
        for (uint64_t i = 0; i < number_of_items * size_per_item; i++) {
            uint8_t v = (uint8_t) dis(gen);
            db.get()[i] = v; keep.get()[i] = v;
        }
        auto sv = std::make_unique<PIRServer>(context, pir_params);
        sv->set_galois_key(0, galois_keys);
        sv->set_database(std::move(db), number_of_items, size_per_item);
        sv->preprocess_database();
        servers.push_back(std::move(sv));
        db_copies.push_back(std::move(keep));
    }
    cudaDeviceSynchronize();
    auto tp1 = clk::now();

    // batch streams (their concurrency model)
    std::vector<cudaStream_t> streams(batch);
    for (int i = 0; i < batch; i++) cudaStreamCreate(&streams[i]);
    std::vector<PirQuery> queries(batch, query);

    std::vector<double> times;
    std::vector<PirReply> replies(num_stripes);
    for (int t = 0; t < trials + 1; t++) {
        auto r0 = clk::now();
        for (int s = 0; s < num_stripes; s++) {
            if (batch == 1) {
                replies[s] = servers[s]->generate_reply(query, 0, streams[0]);
            } else {
                std::vector<PirReply> br(batch);
#pragma omp parallel for num_threads(batch)
                for (int b = 0; b < batch; b++)
                    br[b] = servers[s]->generate_reply(queries[b], 0, streams[b]);
                replies[s] = br[0];
            }
        }
        cudaDeviceSynchronize();
        auto r1 = clk::now();
        if (t > 0) times.push_back(ms(r0, r1)); // first is warmup
    }

    // verify every stripe byte-for-byte
    bool ok = true;
    for (int s = 0; s < num_stripes; s++) {
        std::vector<uint8_t> elems = client.decode_reply(replies[s], fv_offset);
        if (elems.size() != size_per_item) { ok = false; break; }
        for (uint32_t i = 0; ok && i < size_per_item; i++)
            if (elems[i] != db_copies[s].get()[ele * size_per_item + i]) ok = false;
    }

    double mean = 0, var = 0;
    for (double x : times) mean += x;
    mean /= times.size();
    for (double x : times) var += (x - mean) * (x - mean);
    double sd = times.size() > 1 ? std::sqrt(var / (times.size() - 1)) : 0.0;

    double db_gb = (double) number_of_items * size_per_item * num_stripes / (1ULL << 30);
    std::cout << "RESULT items=" << number_of_items
              << " item_B=" << size_per_item
              << " stripes=" << num_stripes
              << " batch=" << batch
              << " db_GB=" << db_gb
              << " querygen_ms=" << ms(tq0, tq1)
              << " preprocess_ms=" << ms(tp0, tp1)
              << " reply_ms_mean=" << mean
              << " reply_ms_sd=" << sd
              << " reply_cts_stripe0=" << replies[0].size()
              << " eff_GBps=" << (db_gb / (mean / 1000.0 / (batch)))
              << " correct=" << (ok ? "yes" : "NO")
              << std::endl;
    return ok ? 0 : 1;
}
