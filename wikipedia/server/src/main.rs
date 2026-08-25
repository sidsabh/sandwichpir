//! Wikipedia PIR Server
//!
//! Wraps pir_server with the Wikipedia database. Serves:
//!   - PIR endpoints (proxied from pir_server): /api/info, /api/query
//!   - Article index: /data/index.tsv.br
//!   - Web UI: /* (static files from wikipedia/web/)
//!   - Dev endpoint: /api/article?row=N&offset=O (cleartext, no privacy)

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex, Condvar};
use std::time::{Duration, Instant};

use clap::Parser;
use log::{debug, info, warn};

use pir_server::{ParsedQuery, PirServer};

#[derive(Parser, Debug)]
#[command(version, about = "Private Wikipedia — powered by SandwichPIR")]
struct Args {
    /// Path to wikipedia.bin
    #[clap(long, default_value = "../../wikipedia-artifacts/wikipedia.bin")]
    db: String,

    /// Number of rows in DB (padded to power of 2)
    #[clap(long, default_value = "65536")]
    num_items: usize,

    /// Item size in bits (128KB = 1048576)
    #[clap(long, default_value = "1048576")]
    item_size_bits: usize,

    /// Listen address
    #[clap(long, default_value = "0.0.0.0:8080")]
    listen: String,

    /// Path to web/ directory (static files)
    #[clap(long, default_value = "../web")]
    web_dir: String,

    /// Path to data/ directory (index files)
    #[clap(long, default_value = "data")]
    data_dir: String,

    /// Max queries to batch before firing GPU (must match GPU buffer allocation)
    #[clap(long, default_value = "64")]
    max_batch: usize,

    /// Max ms to wait for a batch to fill (adaptive: fires early if idle)
    #[clap(long, default_value = "32")]
    batch_timeout_ms: u64,

    /// Verbose
    #[clap(long, short, action)]
    verbose: bool,
}

fn main() {
    let args = Args::parse();

    if args.verbose {
        env_logger::Builder::new()
            .filter_level(log::LevelFilter::Debug)
            .init();
    } else {
        env_logger::Builder::new()
            .filter_level(log::LevelFilter::Info)
            .init();
    }

    // Load Wikipedia DB via mmap — the file is paged in on demand
    // during YServer's tile-transpose inside PirServer::new_from_file,
    // so the peak host memory during construction is ~8 GB (aligned
    // buffer) instead of ~16 GB (caller Vec + aligned). Fits on 16 GB
    // boxes like g6.xlarge.
    info!("Loading Wikipedia database from {} (mmap)...", args.db);
    let start = Instant::now();
    let pir = Arc::new(
        PirServer::new_from_file(
            std::path::Path::new(&args.db),
            args.num_items,
            args.item_size_bits,
        )
        .expect("Failed to open wikipedia.bin"),
    );
    info!("Offline+mmap loaded in {:.1}s", start.elapsed().as_secs_f64());

    // Batch queue: incoming queries wait here until batch fires
    let batch_queue: Arc<BatchQueue> = Arc::new(BatchQueue::new(args.max_batch, args.batch_timeout_ms));

    // Start batch processor thread (GPU runs here)
    #[cfg(feature = "cuda")]
    {
        let bq = Arc::clone(&batch_queue);
        let pir2 = Arc::clone(&pir);
        std::thread::spawn(move || batch_processor(&bq, &pir2));
    }

    info!("==========================================================");
    info!("  Private Wikipedia is ready!");
    info!("  Web UI:  http://{}/", args.listen);
    info!("  API:     http://{}/api/info", args.listen);
    info!("  {} articles, {} rows x {} KB", "6.4M", args.num_items, args.item_size_bits / 8 / 1024);
    info!("  Batching: max_batch={}, timeout={}ms", args.max_batch, args.batch_timeout_ms);
    info!("==========================================================");

    let listener = TcpListener::bind(&args.listen).expect("Failed to bind");

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let pir = Arc::clone(&pir);
                let bq = Arc::clone(&batch_queue);
                let web_dir = args.web_dir.clone();
                let data_dir = args.data_dir.clone();
                std::thread::spawn(move || {
                    if let Err(e) = handle(stream, &pir, &bq, &web_dir, &data_dir) {
                        debug!("Request error: {}", e);
                    }
                });
            }
            Err(e) => warn!("Accept error: {}", e),
        }
    }
}

// ── Batch queue ──

struct BatchResult {
    data: Vec<u8>,
    batch_size: usize,
    timed_out: bool,
}

struct BatchEntry {
    query: ParsedQuery,
    result: Arc<(Mutex<Option<BatchResult>>, Condvar)>,
}

struct BatchQueue {
    entries: Mutex<Vec<BatchEntry>>,
    notify: Condvar,
    max_batch: usize,
    timeout: Duration,
}

impl BatchQueue {
    fn new(max_batch: usize, timeout_ms: u64) -> Self {
        BatchQueue {
            entries: Mutex::new(Vec::new()),
            notify: Condvar::new(),
            max_batch,
            timeout: Duration::from_millis(timeout_ms),
        }
    }

    /// Submit a query and block until the batch result is ready.
    fn submit(&self, query: ParsedQuery) -> BatchResult {
        let result = Arc::new((Mutex::new(None), Condvar::new()));
        {
            let mut entries = self.entries.lock().unwrap();
            entries.push(BatchEntry { query, result: Arc::clone(&result) });
            if entries.len() >= self.max_batch {
                self.notify.notify_one();
            }
        }
        self.notify.notify_one();

        let (lock, cvar) = &*result;
        let mut guard = lock.lock().unwrap();
        while guard.is_none() {
            guard = cvar.wait(guard).unwrap();
        }
        guard.take().unwrap()
    }
}

/// Background thread: collects queries, fires batch on GPU.
/// Adaptive timeout: waits up to `bq.timeout` total, but fires early
/// if no new query arrives within 2ms (idle detection).
#[cfg(feature = "cuda")]
fn batch_processor(bq: &BatchQueue, pir: &PirServer) {
    const IDLE_MS: u64 = 2;

    loop {
        // Wait for at least one query
        let mut entries = bq.entries.lock().unwrap();
        while entries.is_empty() {
            entries = bq.notify.wait(entries).unwrap();
        }

        // Adaptive: wait for batch to fill, hard deadline, or idle timeout
        let deadline = Instant::now() + bq.timeout;
        let mut timed_out = false;
        while entries.len() < bq.max_batch {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() { timed_out = true; break; }
            // Wait at most 2ms for next query (idle detection)
            let wait = remaining.min(Duration::from_millis(IDLE_MS));
            let prev_len = entries.len();
            let (guard, timeout_result) = bq.notify.wait_timeout(entries, wait).unwrap();
            entries = guard;
            if timeout_result.timed_out() && entries.len() == prev_len {
                // No new queries arrived in 2ms — fire now
                timed_out = true;
                break;
            }
        }

        // Drain the batch
        let batch: Vec<BatchEntry> = entries.drain(..).collect();
        drop(entries); // unlock

        let k = batch.len();
        let queries: Vec<&ParsedQuery> = batch.iter().map(|e| &e.query).collect();

        let start = Instant::now();
        let results = pir.answer_batch(&queries);
        info!("Batch of {} queries answered in {} ms", k, start.elapsed().as_millis());

        // Dispatch results back to waiting threads
        for (entry, data) in batch.into_iter().zip(results.into_iter()) {
            let (lock, cvar) = &*entry.result;
            *lock.lock().unwrap() = Some(BatchResult { data, batch_size: k, timed_out });
            cvar.notify_one();
        }
    }
}

fn handle(
    mut stream: std::net::TcpStream,
    pir: &PirServer,
    batch_queue: &BatchQueue,
    web_dir: &str,
    data_dir: &str,
) -> Result<(), String> {
    let (method, path, body) = read_request(&mut stream)?;

    match (method.as_str(), path.as_str()) {
        ("OPTIONS", _) => {
            let h = "HTTP/1.1 200 OK\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type\r\nContent-Length: 0\r\n\r\n";
            stream.write_all(h.as_bytes()).map_err(|e| e.to_string())?;
        }

        // ── PIR endpoints ──
        ("GET", "/api/info") => {
            let body = serde_json::to_string(&pir.info).unwrap();
            respond(&mut stream, 200, "application/json", body.as_bytes());
        }

        ("POST", "/api/query") => {
            let start = Instant::now();

            // Parse the query first
            let parsed = match pir.parse_query(&body) {
                Ok(p) => p,
                Err(e) => {
                    warn!("PIR query parse failed: {}", e);
                    respond(&mut stream, 400, "text/plain", e.as_bytes());
                    return Ok(());
                }
            };

            // GPU: submit to batch queue. CPU: answer directly.
            #[cfg(feature = "cuda")]
            let (resp, batch_size, batch_timed_out) = {
                let br = batch_queue.submit(parsed);
                (br.data, br.batch_size, br.timed_out)
            };

            #[cfg(not(feature = "cuda"))]
            let (resp, batch_size, batch_timed_out) = match pir.answer(&body) {
                Ok(r) => (r, 1usize, false),
                Err(e) => {
                    warn!("PIR query failed: {}", e);
                    respond(&mut stream, 400, "text/plain", e.as_bytes());
                    return Ok(());
                }
            };

            let server_ms = start.elapsed().as_millis();
            let trigger = if batch_timed_out { "timeout" } else { "full" };
            info!("PIR query: {} ms, {} B, batch={} ({})", server_ms, resp.len(), batch_size, trigger);
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Expose-Headers: X-Server-Time-Ms, X-Batch-Size, X-Batch-Timeout\r\nX-Server-Time-Ms: {}\r\nX-Batch-Size: {}\r\nX-Batch-Timeout: {}\r\nContent-Length: {}\r\n\r\n",
                server_ms, batch_size, batch_timed_out, resp.len()
            );
            let _ = stream.write_all(header.as_bytes());
            let _ = stream.write_all(&resp);
        }

        // ── Dev: cleartext article fetch (NO privacy) ──
        ("GET", p) if p.starts_with("/api/article") => {
            let row: usize = extract_param(p, "row").unwrap_or(0);
            let offset: usize = extract_param(p, "offset").unwrap_or(0);

            let row_bytes = pir.get_row_direct(row);
            if offset + 4 > row_bytes.len() {
                respond(&mut stream, 400, "text/plain", b"Offset out of range");
                return Ok(());
            }

            // Extract article: [4-byte u32 length][brotli compressed bytes]
            let len = u32::from_le_bytes(row_bytes[offset..offset + 4].try_into().unwrap()) as usize;
            let end = (offset + 4 + len).min(row_bytes.len());
            let compressed = &row_bytes[offset + 4..end];

            respond(&mut stream, 200, "application/octet-stream", compressed);
        }

        ("GET", p) if p.starts_with("/api/direct") => {
            let row: usize = extract_param(p, "row").unwrap_or(0);
            let data = pir.get_row_direct(row);
            respond(&mut stream, 200, "application/octet-stream", &data);
        }

        // ── Data files (article index) ──
        ("GET", p) if p.starts_with("/data/") => {
            let file = format!("{}/{}", data_dir, &p[6..]);
            serve_file(&mut stream, &file);
        }

        // ── WASM pkg files ──
        ("GET", p) if p.starts_with("/pkg/") => {
            let file = format!("{}/{}", web_dir, p);
            serve_file(&mut stream, &file);
        }

        // ── Static web files ──
        ("GET", "/") => serve_file(&mut stream, &format!("{}/index.html", web_dir)),
        ("GET", p) => serve_file(&mut stream, &format!("{}{}", web_dir, p)),

        _ => respond(&mut stream, 405, "text/plain", b"Method Not Allowed"),
    }
    Ok(())
}

fn extract_param(path: &str, key: &str) -> Option<usize> {
    let pattern = format!("{}=", key);
    path.split(&pattern)
        .nth(1)
        .and_then(|s| s.split('&').next())
        .and_then(|s| s.parse().ok())
}

fn serve_file(stream: &mut std::net::TcpStream, path: &str) {
    match std::fs::read(path) {
        Ok(data) => {
            let ct = match path.rsplit('.').next() {
                Some("html") => "text/html; charset=utf-8",
                Some("js") => "application/javascript",
                Some("css") => "text/css",
                Some("wasm") => "application/wasm",
                Some("json") => "application/json",
                Some("br") => "application/octet-stream",
                Some("tsv") => "text/tab-separated-values",
                _ => "application/octet-stream",
            };
            respond(stream, 200, ct, &data);
        }
        Err(_) => respond(stream, 404, "text/plain", b"Not Found"),
    }
}

fn read_request(stream: &mut std::net::TcpStream) -> Result<(String, String, Vec<u8>), String> {
    let mut header_buf = Vec::with_capacity(4096);
    let mut byte = [0u8; 1];
    loop {
        stream.read_exact(&mut byte).map_err(|e| e.to_string())?;
        header_buf.push(byte[0]);
        if header_buf.len() >= 4 && &header_buf[header_buf.len() - 4..] == b"\r\n\r\n" {
            break;
        }
        if header_buf.len() > 16384 {
            return Err("Headers too large".into());
        }
    }
    let header_str = String::from_utf8_lossy(&header_buf).to_string();
    let first_line = header_str.lines().next().unwrap_or("");
    let parts: Vec<&str> = first_line.split_whitespace().collect();
    if parts.len() < 2 {
        return Err("Bad request".into());
    }

    let content_length: usize = header_str
        .lines()
        .find(|l| l.to_lowercase().starts_with("content-length:"))
        .and_then(|l| l.split(':').nth(1))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0);

    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        stream.read_exact(&mut body).map_err(|e| e.to_string())?;
    }

    Ok((parts[0].to_string(), parts[1].to_string(), body))
}

fn respond(stream: &mut std::net::TcpStream, status: u16, content_type: &str, body: &[u8]) {
    let status_text = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        _ => "Error",
    };
    let header = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\n\r\n",
        status, status_text, content_type, body.len()
    );
    let _ = stream.write_all(header.as_bytes());
    let _ = stream.write_all(body);
}
