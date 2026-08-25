//! PIR Server library.
//!
//! `PirServer` wraps the SandwichPIR offline precomputation and answers
//! stateless queries. Each query includes fresh packing keys + encrypted
//! query. No sessions, no per-client state.
//!
//! Applications (Wikipedia, DNS, etc.) construct a `PirServer` with their
//! database and expose it however they like (HTTP, gRPC, etc.).

use std::time::Instant;

use log::{debug, info};

use sandwichpir::packing::{PackParams, PackingKeys, PackingType, uncondense_matrix};
use sandwichpir::params::{params_for_sandwichpir, GetQPrime};
use sandwichpir::server::*;

use spiral_rs::params::Params;
use spiral_rs::poly::{PolyMatrix, PolyMatrixNTT};

/// Server metadata returned by `info()`.
#[derive(serde::Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct PirServerInfo {
    pub scheme: &'static str,
    pub num_items: usize,
    pub item_size_bytes: usize,
    pub db_rows: usize,
    pub db_cols: usize,
    pub poly_len: usize,
    pub db_dim_1: usize,
    pub instances: usize,
    pub modulus: u64,
    pub pt_modulus: u64,
    pub t_exp_left: usize,
    pub q_prime_1: u64,
    pub q_prime_2: u64,
    pub num_outputs: usize,
    pub response_bytes_per_output: usize,
}

/// Stateless PIR server. Holds database + offline precomputation.
///
/// In CUDA mode the database, hint, precomp, and per-device GPU contexts
/// all live inside `gpu_server: SandwichGpuServer` (sandwichpir core lib).
/// `SandwichGpuServer` transparently handles single-GPU vs multi-GPU
/// sharding via the `MULTI_GPU` env var — `PirServer` does not need to
/// know about sharding at all. CPU mode keeps the old direct fields since
/// it has no notion of multiple devices.
pub struct PirServer {
    pub info: PirServerInfo,
    params: &'static Params,
    #[cfg(feature = "cuda")]
    gpu_server: std::sync::Mutex<sandwichpir::server_gpu::SandwichGpuServer>,
    #[cfg(not(feature = "cuda"))]
    y_server: &'static YServer<'static, u8>,
    #[cfg(not(feature = "cuda"))]
    offline_values: OfflinePrecomputedValues<'static>,
}

impl PirServer {
    /// Create a PIR server from a flat database.
    ///
    /// `db` must be exactly `num_items * (item_size_bits / 8)` bytes.
    /// Runs offline precomputation (GPU if available, CPU otherwise).
    pub fn new(db: Vec<u8>, num_items: usize, item_size_bits: usize) -> Self {
        let item_size_bytes = item_size_bits / 8;
        let expected = num_items * item_size_bytes;
        if db.len() < expected {
            info!("DB has {} bytes, padding to {} ({} items x {} bytes)",
                  db.len(), expected, num_items, item_size_bytes);
        }

        let params: &'static Params =
            Box::leak(Box::new(params_for_sandwichpir(num_items, item_size_bits)));
        let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
        let db_cols = params.instances * params.poly_len;
        let num_outputs = db_cols / params.poly_len;

        let q_prime_1 = params.get_q_prime_1();
        let q_prime_2 = params.get_q_prime_2();
        let q1_bits = (q_prime_2 as f64).log2().ceil() as usize;
        let q2_bits = (q_prime_1 as f64).log2().ceil() as usize;
        let response_bytes_per_output = ((q1_bits + q2_bits) * params.poly_len + 7) / 8;

        let num_real_rows = db.len() / item_size_bytes;
        info!("PIR server: {} items x {} bytes, db_rows={}, db_cols={}, num_outputs={}, real_rows={}",
              num_items, item_size_bytes, db_rows, db_cols, num_outputs, num_real_rows);

        let info = PirServerInfo {
            scheme: "SandwichPIR",
            num_items,
            item_size_bytes,
            db_rows,
            db_cols,
            poly_len: params.poly_len,
            db_dim_1: params.db_dim_1,
            instances: params.instances,
            modulus: params.modulus,
            pt_modulus: params.pt_modulus,
            t_exp_left: params.t_exp_left,
            q_prime_1,
            q_prime_2,
            num_outputs,
            response_bytes_per_output,
        };

        // ── CUDA path: delegate to SandwichGpuServer (handles MULTI_GPU sharding) ──
        #[cfg(feature = "cuda")]
        {
            info!("Using GPU (CUDA) for offline precomputation and online queries");
            info!("Running offline precomputation...");
            let start = Instant::now();
            let gpu_server = sandwichpir::server_gpu::SandwichGpuServer::new(
                db, params.clone(), /* max_batch_size */ 64, None,
            );
            info!("Offline done in {:.1}s", start.elapsed().as_secs_f64());

            return PirServer { info, params, gpu_server: std::sync::Mutex::new(gpu_server) };
        }

        // ── CPU path: existing single-threaded pipeline ──
        #[cfg(not(feature = "cuda"))]
        {
            info!("Using CPU (no CUDA) for offline precomputation and online queries");
            // Fast bulk load + tiled transpose (instead of byte-at-a-time iterator)
            let y_server: &'static YServer<'static, u8> =
                Box::leak(Box::new(YServer::<u8>::new_from_flat_db(params, &db, num_real_rows)));
            drop(db);

            info!("Running offline precomputation...");
            let start = Instant::now();
            let offline_values =
                y_server.perform_offline_precomputation_simplepir(None, PackingType::InspiRING);
            info!("Offline done in {:.1}s", start.elapsed().as_secs_f64());

            return PirServer { info, params, y_server, offline_values };
        }
    }

    /// Construct a PirServer from a file path. Two I/O strategies:
    ///
    /// * **Default** (fast, 16 GB peak): `std::fs::read` loads the
    ///   whole file into a `Vec<u8>`, then hands it to
    ///   `PirServer::new`. Leverages the kernel's efficient
    ///   `read_to_end` path which coalesces Lustre reads and hits
    ///   ~600 MB/s; the tile-transpose then runs on in-memory data
    ///   with no page-fault overhead. Peak memory during construction
    ///   is caller's 8 GB `Vec` + YServer's 8 GB aligned buffer =
    ///   16 GB. Fits comfortably on workstation / cloud nodes with
    ///   32+ GB RAM.
    ///
    /// * **`LOW_MEM=1`** (slow on Lustre, 8 GB peak): streams the
    ///   file through a `BufReader` into the YServer's aligned
    ///   buffer row-batch by row-batch, never holding the full DB
    ///   host-side. Use this on hosts with RAM close to the DB size
    ///   (e.g. g6.xlarge = 16 GB RAM with an 8 GB DB) where the
    ///   default path would OOM. **Avoid on network filesystems**
    ///   (Lustre, NFS): underlying `read(2)` calls often return
    ///   smaller chunks than requested over the network, turning
    ///   each 8 MB logical request into many RPCs and killing
    ///   throughput (measured ~100 s overhead per 8 GB on an HPC cluster).
    pub fn new_from_file(
        path: &std::path::Path,
        num_items: usize,
        item_size_bits: usize,
    ) -> std::io::Result<Self> {
        let low_mem = std::env::var("LOW_MEM").map_or(false, |v| v == "1" || v == "true");
        if low_mem {
            Self::new_from_file_streaming(path, num_items, item_size_bits)
        } else {
            Self::new_from_file_eager(path, num_items, item_size_bits)
        }
    }

    /// Eager path: `std::fs::read` → `PirServer::new`. Peak ~16 GB.
    fn new_from_file_eager(
        path: &std::path::Path,
        num_items: usize,
        item_size_bits: usize,
    ) -> std::io::Result<Self> {
        info!("Loading DB from {} via fs::read (eager) ...", path.display());
        let start = Instant::now();
        let db = std::fs::read(path)?;
        info!(
            "Read {:.2} GB in {:.1}s",
            db.len() as f64 / (1u64 << 30) as f64,
            start.elapsed().as_secs_f64()
        );
        Ok(Self::new(db, num_items, item_size_bits))
    }

    /// Streaming path: `BufReader` → `YServer::new_from_reader`.
    /// Peak ~8 GB. Slow on network filesystems — use only when RAM
    /// is tight.
    fn new_from_file_streaming(
        path: &std::path::Path,
        num_items: usize,
        item_size_bits: usize,
    ) -> std::io::Result<Self> {
        use std::io::BufReader;

        let item_size_bytes = item_size_bits / 8;
        let file = std::fs::File::open(path)?;
        let file_len = file.metadata()?.len() as usize;
        let num_real_rows = file_len / item_size_bytes;

        info!(
            "Loading DB from {} via streaming reader (LOW_MEM) ...",
            path.display()
        );
        info!(
            "file = {} bytes, real_rows = {}",
            file_len, num_real_rows
        );

        let mut reader = BufReader::with_capacity(8 * 1024 * 1024, file);

        let params: &'static Params =
            Box::leak(Box::new(params_for_sandwichpir(num_items, item_size_bits)));
        let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
        let db_cols = params.instances * params.poly_len;
        let num_outputs = db_cols / params.poly_len;

        let q_prime_1 = params.get_q_prime_1();
        let q_prime_2 = params.get_q_prime_2();
        let q1_bits = (q_prime_2 as f64).log2().ceil() as usize;
        let q2_bits = (q_prime_1 as f64).log2().ceil() as usize;
        let response_bytes_per_output = ((q1_bits + q2_bits) * params.poly_len + 7) / 8;

        let info = PirServerInfo {
            scheme: "SandwichPIR",
            num_items,
            item_size_bytes,
            db_rows,
            db_cols,
            poly_len: params.poly_len,
            db_dim_1: params.db_dim_1,
            instances: params.instances,
            modulus: params.modulus,
            pt_modulus: params.pt_modulus,
            t_exp_left: params.t_exp_left,
            q_prime_1,
            q_prime_2,
            num_outputs,
            response_bytes_per_output,
        };

        #[cfg(feature = "cuda")]
        {
            info!("Using GPU (CUDA) for offline precomputation and online queries");
            info!("Running offline precomputation...");
            let start = Instant::now();
            let gpu_server = sandwichpir::server_gpu::SandwichGpuServer::new_from_reader(
                &mut reader,
                num_real_rows,
                params.clone(),
                /* max_batch_size */ 64,
                None,
            )?;
            info!("Offline done in {:.1}s", start.elapsed().as_secs_f64());
            return Ok(PirServer { info, params, gpu_server: std::sync::Mutex::new(gpu_server) });
        }

        #[cfg(not(feature = "cuda"))]
        {
            info!("Using CPU (no CUDA) for offline precomputation and online queries");
            let y_server: &'static YServer<'static, u8> =
                Box::leak(Box::new(YServer::<u8>::new_from_reader(params, &mut reader, num_real_rows)?));
            info!("Running offline precomputation...");
            let start = Instant::now();
            let offline_values =
                y_server.perform_offline_precomputation_simplepir(None, PackingType::InspiRING);
            info!("Offline done in {:.1}s", start.elapsed().as_secs_f64());
            return Ok(PirServer { info, params, y_server, offline_values });
        }
    }

    /// Internal helper: shared construction from a pre-loaded
    /// `&[u8]` slice. Used by both `new(Vec<u8>)` (legacy) and
    /// `new_from_file(Path)` (mmap'd).
    fn new_with_slice(db: &[u8], num_items: usize, item_size_bits: usize) -> Self {
        let item_size_bytes = item_size_bits / 8;
        let expected = num_items * item_size_bytes;
        if db.len() < expected {
            info!("DB has {} bytes, padding to {} ({} items x {} bytes)",
                  db.len(), expected, num_items, item_size_bytes);
        }

        let params: &'static Params =
            Box::leak(Box::new(params_for_sandwichpir(num_items, item_size_bits)));
        let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
        let db_cols = params.instances * params.poly_len;
        let num_outputs = db_cols / params.poly_len;

        let q_prime_1 = params.get_q_prime_1();
        let q_prime_2 = params.get_q_prime_2();
        let q1_bits = (q_prime_2 as f64).log2().ceil() as usize;
        let q2_bits = (q_prime_1 as f64).log2().ceil() as usize;
        let response_bytes_per_output = ((q1_bits + q2_bits) * params.poly_len + 7) / 8;

        let num_real_rows = db.len() / item_size_bytes;
        info!("PIR server (mmap): {} items x {} bytes, db_rows={}, db_cols={}, num_outputs={}, real_rows={}",
              num_items, item_size_bytes, db_rows, db_cols, num_outputs, num_real_rows);

        let info = PirServerInfo {
            scheme: "SandwichPIR",
            num_items,
            item_size_bytes,
            db_rows,
            db_cols,
            poly_len: params.poly_len,
            db_dim_1: params.db_dim_1,
            instances: params.instances,
            modulus: params.modulus,
            pt_modulus: params.pt_modulus,
            t_exp_left: params.t_exp_left,
            q_prime_1,
            q_prime_2,
            num_outputs,
            response_bytes_per_output,
        };

        #[cfg(feature = "cuda")]
        {
            info!("Using GPU (CUDA) for offline precomputation and online queries");
            info!("Running offline precomputation...");
            let start = Instant::now();
            // Route the mmap'd slice straight through SandwichGpuServer::
            // new_from_slice → YServer::new_from_flat_db. No intermediate
            // owned Vec, so the peak host memory during construction is
            // ~8 GB (YServer's aligned transposed buffer) plus a small
            // kernel-managed working set for the mmap, rather than the
            // 16 GB peak the Vec path would produce.
            let gpu_server = sandwichpir::server_gpu::SandwichGpuServer::new_from_slice(
                db, params.clone(), /* max_batch_size */ 64, None,
            );
            info!("Offline done in {:.1}s", start.elapsed().as_secs_f64());
            return PirServer { info, params, gpu_server: std::sync::Mutex::new(gpu_server) };
        }

        #[cfg(not(feature = "cuda"))]
        {
            info!("Using CPU (no CUDA) for offline precomputation and online queries");
            let y_server: &'static YServer<'static, u8> =
                Box::leak(Box::new(YServer::<u8>::new_from_flat_db(params, db, num_real_rows)));
            info!("Running offline precomputation...");
            let start = Instant::now();
            let offline_values =
                y_server.perform_offline_precomputation_simplepir(None, PackingType::InspiRING);
            info!("Offline done in {:.1}s", start.elapsed().as_secs_f64());
            return PirServer { info, params, y_server, offline_values };
        }
    }

    /// Parse a request into its components.
    ///
    /// Wire format (little-endian):
    ///   [1B]  format: 0 = want full response (mask+body), 1 = want body only
    ///   [4B]  y_body_len (u64 count)
    ///   [4B]  z_body_len (u64 count)
    ///   [y_body_len * 8B] y_body condensed
    ///   [z_body_len * 8B] z_body condensed
    ///   [db_rows * 4B] encrypted query (u32 values)
    pub fn parse_query(&self, request: &[u8]) -> Result<ParsedQuery, String> {
        let params = self.params;
        let poly_len = params.poly_len;
        let t = params.t_exp_left;
        let cpn = params.crt_count * poly_len;

        if request.len() < 9 {
            return Err("Request too short".into());
        }

        let body_only = request[0] == 1;
        let y_len = u32::from_le_bytes(request[1..5].try_into().unwrap()) as usize;
        let z_len = u32::from_le_bytes(request[5..9].try_into().unwrap()) as usize;

        let expected = 9 + y_len * 4 + z_len * 4 + self.info.db_rows * 4;
        if request.len() < expected {
            return Err(format!("Request too short: {} < {}", request.len(), expected));
        }
        if y_len != t * cpn || z_len != t * cpn {
            return Err(format!("Key size mismatch: y={} z={} expected={}", y_len, z_len, t * cpn));
        }

        // Parse keys (u32 on wire → u64 in memory)
        let y_start = 9;
        let mut y_data = vec![0u64; y_len];
        for i in 0..y_len {
            y_data[i] = u32::from_le_bytes(request[y_start + i * 4..y_start + (i + 1) * 4].try_into().unwrap()) as u64;
        }
        let z_start = y_start + y_len * 4;
        let mut z_data = vec![0u64; z_len];
        for i in 0..z_len {
            z_data[i] = u32::from_le_bytes(request[z_start + i * 4..z_start + (i + 1) * 4].try_into().unwrap()) as u64;
        }

        // Parse query (u32 → u64)
        let q_start = z_start + z_len * 4;
        let mut query = vec![0u64; self.info.db_rows];
        for i in 0..self.info.db_rows {
            query[i] = u32::from_le_bytes(request[q_start + i * 4..q_start + (i + 1) * 4].try_into().unwrap()) as u64;
        }

        Ok(ParsedQuery { query, y_data, z_data, body_only })
    }

    /// Answer a single parsed query.
    /// Returns full response (mask+body) or body-only based on `parsed.body_only`.
    pub fn answer(&self, request: &[u8]) -> Result<Vec<u8>, String> {
        let parsed = self.parse_query(request)?;

        // CUDA path: delegate to answer_batch with k=1 so MULTI_GPU sharding
        // applies transparently. The single-shard path through SandwichGpuServer
        // is byte-identical to a direct compute_batch call.
        #[cfg(feature = "cuda")]
        {
            let mut responses = self.answer_batch(&[&parsed]);
            return Ok(responses.pop().expect("answer_batch returned empty for k=1"));
        }

        // CPU path: existing single-query CPU pipeline.
        #[cfg(not(feature = "cuda"))]
        {
            let full_resp = self.execute_query(&parsed)?;
            if parsed.body_only {
                Ok(self.strip_mask(&full_resp))
            } else {
                Ok(full_resp)
            }
        }
    }

    /// Execute a single query and return full response bytes.
    /// CPU-only path; CUDA queries go through `answer_batch` (which routes
    /// through `SandwichGpuServer` for shard-transparent dispatch).
    #[cfg(not(feature = "cuda"))]
    fn execute_query(&self, parsed: &ParsedQuery) -> Result<Vec<u8>, String> {
        let params = self.params;
        let poly_len = params.poly_len;
        let t = params.t_exp_left;

        let y_condensed = {
            let mut m = PolyMatrixNTT::zero(params, 1, t);
            m.as_mut_slice().copy_from_slice(&parsed.y_data);
            m
        };
        let z_condensed = {
            let mut m = PolyMatrixNTT::zero(params, 1, t);
            m.as_mut_slice().copy_from_slice(&parsed.z_data);
            m
        };

        let y_body = uncondense_matrix(params, &y_condensed);
        let z_body = uncondense_matrix(params, &z_condensed);

        let pk = PackingKeys::init_from_uploaded(
            params.clone(), poly_len,
            Some(y_body), Some(z_body),
            Some(y_condensed), Some(z_condensed),
        );

        let start = Instant::now();
        let query_slices: Vec<&[u64]> = vec![parsed.query.as_slice()];
        let mut packing_keys = vec![pk];
        let responses = self.y_server.perform_online_computation_simplepir_word_cpu(
            &query_slices, &self.offline_values, &mut packing_keys, None,
        );
        let flat: Vec<u8> = responses.into_iter().next().unwrap().into_iter().flatten().collect();
        debug!("CPU query answered in {} ms", start.elapsed().as_millis());
        Ok(flat)
    }

    /// Answer a batch of parsed queries in one GPU call.
    /// Routes through `SandwichGpuServer::compute_batch` which handles the
    /// single-GPU fast path AND multi-GPU sharded dispatch transparently.
    ///
    /// The GPU produces body-only bytes; the static mask template lives
    /// host-side on `SandwichGpuServer`. For each query:
    ///   - `body_only == true` (steady-state clients that cached the
    ///     mask): return the body slice directly, zero assembly cost.
    ///   - `body_only == false` (first query per client, or clients
    ///     that did not cache the mask): assemble the full wire format
    ///     `[mask | body]` per output. Assembly is a pair of memcpys
    ///     per output — ~288 KB per client for default params.
    #[cfg(feature = "cuda")]
    pub fn answer_batch(&self, queries: &[&ParsedQuery]) -> Vec<Vec<u8>> {
        let k = queries.len();
        let queries_flat: Vec<u64> = queries.iter().flat_map(|q| q.query.iter().copied()).collect();
        let y_flat: Vec<u64> = queries.iter().flat_map(|q| q.y_data.iter().copied()).collect();
        let z_flat: Vec<u64> = queries.iter().flat_map(|q| q.z_data.iter().copied()).collect();

        let start = Instant::now();
        let mut gpu = self.gpu_server.lock().unwrap();
        let resp = gpu.compute_batch(&queries_flat, &y_flat, &z_flat, k);
        debug!("GPU batch of {} answered in {} ms", k, start.elapsed().as_millis());

        (0..k).map(|i| {
            if queries[i].body_only {
                resp.body_only(i)
            } else {
                resp.full_response(i).to_vec()
            }
        }).collect()
    }

    /// Get a row directly (NO privacy — for development/testing only).
    pub fn get_row_direct(&self, row: usize) -> Vec<u8> {
        #[cfg(feature = "cuda")]
        {
            self.gpu_server.lock().unwrap().get_row_direct(row)
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.y_server.get_row(row).iter().map(|&x| x as u8).collect()
        }
    }
}

/// A parsed but not yet executed PIR query.
pub struct ParsedQuery {
    pub query: Vec<u64>,
    pub y_data: Vec<u64>,
    pub z_data: Vec<u64>,
    pub body_only: bool,
}
