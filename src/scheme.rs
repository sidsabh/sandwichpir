//! SandwichPIR: Hybrid RLWE->WordPIR with InspiRING packing.
//!
//! Parameters: W=2^32, Q=4294955009, d=2048, p=256, Xs=Xe=D(0.5),
//! t=4, z=256, q21=2^18, q22=2^10. InspiRING only. 192-bit security, log2(delta)=-105.

use std::time::Instant;

use log::debug;
use rand::{thread_rng, Rng};

use spiral_rs::client::*;
use spiral_rs::params::*;
use spiral_rs::poly::{PolyMatrix, PolyMatrixRaw};

use crate::client::*;
use crate::measurement::*;
use crate::modulus_switch::ModulusSwitch;
use crate::packing::{PackingKeys, PackingType};
use crate::params::*;
#[cfg(not(feature = "cuda"))]
use crate::server::ToU64;

pub const STATIC_PUBLIC_SEED: [u8; 32] = [0u8; 32];
pub const SEED_0: u8 = 0;
pub const SEED_1: u8 = 1;

pub const STATIC_SEED_2: [u8; 32] = [
    2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0,
];

pub const W_SEED: [u8; 32] = [
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7,
];
pub const V_SEED: [u8; 32] = [
    8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7,
];

macro_rules! dispatch_const {
    ($val:expr, [$($n:literal),+], |$name:ident| $body:expr) => {
        match $val {
            $($n => { const $name: usize = $n; $body },)+
            _ => panic!("Unsupported value: {}", $val),
        }
    };
}

pub trait Sample {
    fn sample() -> Self;
}

impl Sample for u8 {
    fn sample() -> Self {
        fastrand::u8(..)
    }
}

impl Sample for u16 {
    fn sample() -> Self {
        fastrand::u16(..)
    }
}

pub fn generate_packing_keys<'a>(
    packing: PackingType,
    params: &'a Params,
    sk_reg: &PolyMatrixRaw<'a>,
    offline_packing_params: Option<&crate::packing::PackParams<'a>>,
) -> (usize, PackingKeys<'a>) {
    match packing {
        PackingType::InspiRING => {
            let pp = offline_packing_params.unwrap();
            let packing_keys = PackingKeys::init_full(pp, sk_reg, W_SEED, V_SEED);
            let size = packing_keys.get_size_bytes();
            debug!("InspiRING packing key size: {} bytes", size);
            (size, packing_keys)
        }
        _ => {
            use rand::SeedableRng;
            use rand_chacha::ChaCha20Rng;
            use crate::packing::condense_matrix;
            let pack_pub_params = raw_generate_expansion_params(
                params,
                sk_reg,
                params.poly_len_log2,
                params.t_exp_left,
                &mut ChaCha20Rng::from_entropy(),
                &mut ChaCha20Rng::from_seed(STATIC_SEED_2),
            );
            let mut pack_pub_params_row_1s = pack_pub_params.to_vec();
            for i in 0..pack_pub_params.len() {
                pack_pub_params_row_1s[i] =
                    pack_pub_params[i].submatrix(1, 0, 1, pack_pub_params[i].cols);
                pack_pub_params_row_1s[i] = condense_matrix(params, &pack_pub_params_row_1s[i]);
            }
            let size = get_vec_pm_size_bytes(&pack_pub_params_row_1s);
            debug!("pub params size: {} bytes", size);
            let packing_keys =
                PackingKeys::init_cdks_from_keys(params.clone(), pack_pub_params_row_1s);
            (size, packing_keys)
        }
    }
}

pub fn finalize_measurements(measurements: &mut Vec<Measurement>, trials: usize) -> Measurement {
    if trials > 1 {
        measurements[1].offline = measurements[0].offline.clone();
        measurements.remove(0);
    }
    let mut final_measurement = measurements[0].clone();
    final_measurement.online.server_time_ms = mean(
        &measurements
            .iter()
            .map(|m| m.online.server_time_ms)
            .collect::<Vec<_>>(),
    );
    final_measurement.online.all_server_times_ms = measurements
        .iter()
        .map(|m| m.online.server_time_ms)
        .collect::<Vec<_>>();
    final_measurement.online.std_dev_server_time_ms =
        std_dev(&final_measurement.online.all_server_times_ms);
    final_measurement.online.client_query_gen_time_ms = mean(
        &measurements
            .iter()
            .map(|m| m.online.client_query_gen_time_ms)
            .collect::<Vec<_>>(),
    );
    final_measurement.online.client_decode_time_ms = mean(
        &measurements
            .iter()
            .map(|m| m.online.client_decode_time_ms)
            .collect::<Vec<_>>(),
    );
    final_measurement
}

fn mean(data: &[f64]) -> f64 {
    let sum: f64 = data.iter().sum();
    sum / data.len() as f64
}

fn std_dev(data: &[f64]) -> f64 {
    let m = mean(data);
    let variance: f64 = data.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / data.len() as f64;
    variance.sqrt()
}

// ==================== SandwichPIR result types ====================

#[derive(serde::Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct SandwichResult {
    pub scheme: &'static str,
    pub num_clients: usize,
    pub num_gpus: usize,
    pub num_items: usize,
    pub item_size_bytes: usize,
    pub db_size_bytes: usize,
    pub offline: SandwichOffline,
    pub online: SandwichOnline,
}

#[derive(serde::Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct SandwichOffline {
    pub hint_mode: &'static str,
    pub hint_time_ms: f64,
    pub precomp_time_ms: f64,
    pub total_time_ms: f64,
    pub hint_throughput_gbs: f64,
    pub throughput_gbs: f64,
}

#[derive(serde::Serialize, Debug)]
#[serde(rename_all = "camelCase")]
pub struct SandwichOnline {
    pub server_time_ms: f64,
    pub upload_bytes: usize,
    pub download_bytes: usize,
    pub client_query_gen_time_ms: f64,
    pub client_decode_time_ms: f64,
    pub throughput_gbs: f64,
    pub all_server_times_ms: Vec<f64>,
    pub std_dev_server_time_ms: f64,
}

// ==================== Entry point ====================

pub fn run_sandwichpir_batched(
    num_items: usize,
    item_size_bits: usize,
    num_clients: usize,
    trials: usize,
) -> SandwichResult {
    let params = params_for_sandwichpir(num_items, item_size_bits);

    #[cfg(feature = "cuda")]
    let measurement = dispatch_const!(num_clients, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64, 128, 256, 512, 1024, 2048], |N| {
        run_sandwichpir_on_params::<N>(params, trials)
    });

    #[cfg(not(feature = "cuda"))]
    let measurement = dispatch_const!(num_clients, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], |N| {
        run_sandwichpir_on_params::<N>(params, trials)
    });

    let db_size_bytes = (num_items * item_size_bits + 7) / 8;
    let hint_ms = measurement.offline.simplepir_prep_time_ms;
    let offline_ms = measurement.offline.server_time_ms;
    let precomp_ms = offline_ms - hint_ms;
    let online_ms = measurement.online.server_time_ms;
    let hint_tput = (db_size_bytes as f64) / (hint_ms / 1000.0) / (1u64 << 30) as f64;
    let offline_tput = (db_size_bytes as f64) / (offline_ms / 1000.0) / (1u64 << 30) as f64;
    let online_tput =
        (num_clients * db_size_bytes) as f64 / (online_ms / 1000.0) / (1u64 << 30) as f64;

    // Build the status block as one string and write it with one
    // `eprint!` call so the whole thing hits stderr as a single
    // atomic `write(2)` syscall. Multiple separate `eprintln`s were
    // racing against the C-side `printf`s from CUDA `SW_LOG` and
    // against the Rust stdout `BufWriter` that flushes at exit,
    // producing apparent mid-line truncation in redirected logs.
    // A single write_all avoids all of that.
    let status = format!(
        "Hint Throughput: {:.2} GiB/sec ({:.2} ms)\n\
         Offline Throughput: {:.2} GiB/sec (hint={:.2} + precomp={:.2} = {:.2} ms)\n\
         Online Throughput: {:.2} GiB/sec ({:.2} ms per {} clients)\n",
        hint_tput, hint_ms,
        offline_tput, hint_ms, precomp_ms, offline_ms,
        online_tput, online_ms, num_clients,
    );
    eprint!("{}", status);
    use std::io::Write;
    let _ = std::io::stderr().flush();

    let num_gpus = std::env::var("MULTI_GPU")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(1);

    SandwichResult {
        scheme: "SandwichPIR",
        num_clients,
        num_gpus,
        num_items,
        item_size_bytes: item_size_bits / 8,
        db_size_bytes,
        offline: SandwichOffline {
            hint_mode: if std::env::var("HINT").map_or(false, |v| v == "gemm") {
                "gemm"
            } else {
                "ntt"
            },
            hint_time_ms: measurement.offline.simplepir_prep_time_ms,
            precomp_time_ms: measurement.offline.server_time_ms
                - measurement.offline.simplepir_prep_time_ms,
            total_time_ms: measurement.offline.server_time_ms,
            hint_throughput_gbs: (hint_tput * 100.0).round() / 100.0,
            throughput_gbs: (offline_tput * 100.0).round() / 100.0,
        },
        online: SandwichOnline {
            server_time_ms: measurement.online.server_time_ms,
            upload_bytes: measurement.online.upload_bytes,
            download_bytes: measurement.online.download_bytes,
            client_query_gen_time_ms: measurement.online.client_query_gen_time_ms,
            client_decode_time_ms: measurement.online.client_decode_time_ms,
            throughput_gbs: (online_tput * 100.0).round() / 100.0,
            all_server_times_ms: measurement.online.all_server_times_ms,
            std_dev_server_time_ms: measurement.online.std_dev_server_time_ms,
        },
    }
}

// ==================== Self-contained pipeline ====================

fn run_sandwichpir_on_params<const K: usize>(params: Params, trials: usize) -> Measurement {
    let packing = PackingType::InspiRING;
    let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
    let db_cols = params.instances * params.poly_len;
    let num_rlwe_outputs = db_cols / params.poly_len;

    let q = params.modulus;
    let rlwe_q_prime_1 = params.get_q_prime_1();
    let rlwe_q_prime_2 = params.get_q_prime_2();

    let mut rng = thread_rng();

    // ── Create server ──
    let mut measurements = vec![Measurement::default(); trials + 1];

    // CUDA path: the random DB is generated **directly** into each
    // shard's aligned YServer buffer via `new_random_db`, avoiding the
    // ~16 GB construction peak (caller's 8 GB Vec + YServer's 8 GB
    // aligned copy) and the ~15–40 s per-byte fill + tile-transpose
    // of the old flat-Vec path. At 8 GB DB the fill drops to ~1–2 s
    // and peaks at 8 GB instead of 16.
    //
    // CPU path: fall back to the existing iterator-based YServer pipeline.
    #[cfg(feature = "cuda")]
    let mut gpu_server = {
        debug!("Constructing SandwichGpuServer with random DB (direct fill)...");
        let t_start = Instant::now();
        let server = crate::server_gpu::SandwichGpuServer::new_random_db(
            params.clone(),
            K,
            Some(&mut measurements[0]),
        );
        debug!(
            "SandwichGpuServer random-DB construction: {:.2} s",
            t_start.elapsed().as_secs_f64()
        );
        server
    };

    // CUDA mode: build a PackParams locally for query generation. (The
    // existing code reads it from offline_values.packing_params; with
    // SandwichGpuServer the caller owns its own copy. PackParams is cheap
    // — just NTT tables / generator powers / monomials setup.)
    #[cfg(feature = "cuda")]
    let cuda_packing_params = crate::packing::PackParams::new_for_gpu(&params, params.poly_len);

    #[cfg(not(feature = "cuda"))]
    let y_server = {
        use crate::server::YServer;
        type T = u8; // p=256, 1 byte per entry — no wasted memory
        let now = Instant::now();
        let pt_iter =
            std::iter::repeat_with(|| (T::sample() as u64 % params.pt_modulus) as T);
        let y_server = YServer::<T>::new(&params, pt_iter, true, false, true);
        debug!("Created server in {} us", now.elapsed().as_micros());
        debug!("Database of {} bytes", y_server.db().len());
        y_server
    };

    #[cfg(not(feature = "cuda"))]
    let offline_values = {
        let start_offline = Instant::now();
        let ov = y_server
            .perform_offline_precomputation_simplepir(Some(&mut measurements[0]), packing);
        measurements[0].offline.server_time_ms =
            start_offline.elapsed().as_micros() as f64 / 1000.0;
        ov
    };

    // ── TRIALS ──
    for trial in 0..trials + 1 {
        debug!("trial: {}", trial);
        let measurement = &mut measurements[trial];
        // Both paths set measurements[0].offline.server_time_ms during the offline block;
        // finalize_measurements propagates it to all trials.

        // ── QUERY GENERATION ──
        let mut query_meta: Vec<(YClient, usize, PackingKeys)> = Vec::new();
        let mut word_queries: Vec<Vec<u64>> = Vec::new();
        let mut online_upload_bytes: usize = 0;
        let mut client_query_gen_sum_ms: f64 = 0.0;

        let mut clients = (0..K).map(|_| Client::init(&params)).collect::<Vec<_>>();

        for (_batch, client) in (0..K).zip(clients.iter_mut()) {
            let target_idx: usize = rng.gen::<usize>() % (db_rows * db_cols);
            let target_row = target_idx / db_cols;
            debug!(
                "Target item: {} ({}, {})",
                target_idx,
                target_row,
                target_idx % db_cols
            );

            let start = Instant::now();
            client.generate_secret_keys();

            let sk_reg = &client.get_sk_reg();
            #[cfg(feature = "cuda")]
            let (pub_params_size, pk) = generate_packing_keys(
                packing,
                &params,
                sk_reg,
                Some(&cuda_packing_params),
            );
            #[cfg(not(feature = "cuda"))]
            let (pub_params_size, pk) = generate_packing_keys(
                packing,
                &params,
                sk_reg,
                offline_values.packing_params.as_ref(),
            );
            debug!("InspiRING packing key size: {} bytes", pub_params_size);

            let y_client = YClient::new(client, &params);

            // RLWE query -> extract scalar LWEs -> modswitch Q -> W=2^32
            let q_vals =
                y_client.generate_query(SEED_0, params.db_dim_1, packing, target_row, None, None);
            let wq: Vec<u64> = q_vals
                .iter()
                .map(|&v| {
                    ((v as u128 * (1u128 << 32) + (q as u128 / 2)) / q as u128) as u64
                        & 0xFFFFFFFF
                })
                .collect();
            assert_eq!(wq.len(), db_rows);

            online_upload_bytes += wq.len() * 4 + pub_params_size;
            word_queries.push(wq);
            query_meta.push((y_client, target_idx, pk));

            let dt_ms = start.elapsed().as_micros() as f64 / 1000.0;
            client_query_gen_sum_ms += dt_ms;
            debug!("Generated query in {} us", start.elapsed().as_micros());
        }
        // Mean per-query client encryption time over this trial's K clients.
        measurement.online.client_query_gen_time_ms = client_query_gen_sum_ms / K as f64;

        // ── ONLINE ──
        #[allow(unused_mut)]
        let mut packing_keys: Vec<PackingKeys> =
            query_meta.iter().map(|(_, _, pk)| pk.clone()).collect();

        // Build query + packing keys directly into PINNED host memory so the
        // GPU's cudaMemcpyAsync sees pinned sources and skips the driver's
        // internal pageable→staging copy. This is NOT a pre-dispatch memcpy
        // (which the failed earlier experiment showed is wash) — we replace
        // the original Vec destination with a pinned destination, so the
        // number of host copies is unchanged; only the allocator changes.
        #[cfg(feature = "cuda")]
        let (queries_pinned, y_body_pinned, z_body_pinned) = {
            use crate::cuda::sandwich::PinnedHostBuffer;

            // Compute exact total sizes from the first packing key (all
            // packing keys share the same shape since params is fixed).
            let queries_total: usize = word_queries.iter().map(|q| q.len()).sum();
            let (y_len_per, z_len_per) = {
                let pk0 = &packing_keys[0];
                let y0 = pk0.y_body_condensed.as_ref().unwrap();
                let z0 = pk0.z_body_condensed.as_ref().unwrap();
                // All (r,c) polys have the same length for a given params.
                (
                    y0.rows * y0.cols * y0.get_poly(0, 0).len(),
                    z0.rows * z0.cols * z0.get_poly(0, 0).len(),
                )
            };

            let mut queries_pinned = PinnedHostBuffer::<u64>::new(queries_total);
            let mut y_body_pinned = PinnedHostBuffer::<u64>::new(K * y_len_per);
            let mut z_body_pinned = PinnedHostBuffer::<u64>::new(K * z_len_per);

            // Queries: copy each per-client u32→u64 modswitched vector into
            // its slot in the pinned buffer.
            {
                let dst = queries_pinned.as_mut_slice();
                let mut off = 0;
                for q in word_queries.iter() {
                    dst[off..off + q.len()].copy_from_slice(q);
                    off += q.len();
                }
                debug_assert_eq!(off, queries_total);
            }

            // Packing keys: flatten each client's condensed (y, z) PolyMatrices
            // directly into the pinned buffers at their per-client offsets.
            {
                let y_dst = y_body_pinned.as_mut_slice();
                let z_dst = z_body_pinned.as_mut_slice();
                let mut y_off = 0;
                let mut z_off = 0;
                for pk in packing_keys.iter() {
                    let y = pk.y_body_condensed.as_ref().unwrap();
                    for r in 0..y.rows {
                        for c in 0..y.cols {
                            let p = y.get_poly(r, c);
                            y_dst[y_off..y_off + p.len()].copy_from_slice(p);
                            y_off += p.len();
                        }
                    }
                    let z = pk.z_body_condensed.as_ref().unwrap();
                    for r in 0..z.rows {
                        for c in 0..z.cols {
                            let p = z.get_poly(r, c);
                            z_dst[z_off..z_off + p.len()].copy_from_slice(p);
                            z_off += p.len();
                        }
                    }
                }
                debug_assert_eq!(y_off, K * y_len_per);
                debug_assert_eq!(z_off, K * z_len_per);
            }

            (queries_pinned, y_body_pinned, z_body_pinned)
        };

        let start_online = Instant::now();

        #[cfg(feature = "cuda")]
        let resp_split = gpu_server.compute_batch(
            queries_pinned.as_slice(),
            y_body_pinned.as_slice(),
            z_body_pinned.as_slice(),
            K,
        );

        #[cfg(feature = "cuda")]
        let online_ms = start_online.elapsed().as_micros() as f64 / 1000.0;

        #[cfg(feature = "cuda")]
        let responses = {
            // full_response_output returns zero-copy &[u8] slices into the
            // persistent response buffer (mask was pre-filled at offline,
            // body was scatter-written by compute_batch). We convert to
            // owned Vec<u8> only because the decode loop below takes owned
            // slices — the actual mask bytes are never copied online.
            (0..K)
                .map(|c| {
                    (0..num_rlwe_outputs)
                        .map(|o| resp_split.full_response_output(c, o).to_vec())
                        .collect::<Vec<Vec<u8>>>()
                })
                .collect::<Vec<Vec<Vec<u8>>>>()
        };

        #[cfg(not(feature = "cuda"))]
        let responses = {
            let query_slices: Vec<&[u64]> =
                word_queries.iter().map(|q| q.as_slice()).collect();
            y_server.perform_online_computation_simplepir_word_cpu(
                &query_slices,
                &offline_values,
                &mut packing_keys,
                Some(measurement),
            )
        };

        #[cfg(not(feature = "cuda"))]
        let online_ms = start_online.elapsed().as_micros() as f64 / 1000.0;
        let online_download_bytes = get_size_bytes(&responses);

        // ── DECODE + VERIFY ──
        let mut client_decode_sum_ms: f64 = 0.0;
        for (response_switched, (y_client, target_idx, _)) in
            responses.iter().zip(query_meta.iter())
        {
            let target_row = target_idx / db_cols;
            #[cfg(feature = "cuda")]
            let corr_result: Vec<u64> = gpu_server
                .get_row_direct(target_row)
                .iter()
                .map(|&x| x as u64)
                .collect();
            #[cfg(not(feature = "cuda"))]
            let corr_result = y_server
                .get_row(target_row)
                .iter()
                .map(|x| x.to_u64())
                .collect::<Vec<_>>();

            let start_decode = Instant::now();

            debug!("rescaling response...");
            let mut response = Vec::new();
            for ct_bytes in response_switched.iter() {
                let ct =
                    PolyMatrixRaw::recover(&params, rlwe_q_prime_1, rlwe_q_prime_2, ct_bytes);
                response.push(ct);
            }

            debug!("decrypting outer cts...");
            let outer_ct: Vec<u64> = response
                .iter()
                .flat_map(|ct| {
                    decrypt_ct_reg_measured(y_client.client(), &params, &ct.ntt(), params.poly_len)
                        .as_slice()
                        .to_vec()
                })
                .collect();
            assert_eq!(outer_ct.len(), num_rlwe_outputs * params.poly_len);

            client_decode_sum_ms += start_decode.elapsed().as_micros() as f64 / 1000.0;

            if outer_ct.as_slice() != corr_result {
                let mismatches: Vec<usize> = outer_ct
                    .iter()
                    .zip(corr_result.iter())
                    .enumerate()
                    .filter(|(_, (a, b))| a != b)
                    .map(|(i, _)| i)
                    .collect();
                eprintln!(
                    "MISMATCH: {} / {} values differ",
                    mismatches.len(),
                    outer_ct.len()
                );
                for &i in mismatches.iter().take(10) {
                    eprintln!("  [{i}] got={}, expected={}", outer_ct[i], corr_result[i]);
                }
            }
            assert_eq!(outer_ct.as_slice(), corr_result.as_slice());
        }

        // Per-client semantics: report bytes and client-side times per query,
        // not per batch. Server time stays batch-wall since that is how the
        // server schedules work; divide by K downstream for per-client latency.
        measurement.online.upload_bytes = online_upload_bytes / K;
        measurement.online.download_bytes = online_download_bytes / K;
        measurement.online.client_decode_time_ms = client_decode_sum_ms / K as f64;
        measurement.online.server_time_ms = online_ms;
    }

    finalize_measurements(&mut measurements, trials)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_sandwichpir_basic() {
        run_sandwichpir_batched(1 << 10, 65536 * 8, 1, 0);
    }

    #[test]
    fn test_sandwichpir_batched() {
        run_sandwichpir_batched(1 << 10, 65536 * 8, 2, 0);
    }
}
