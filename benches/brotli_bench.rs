//! Brotli decompression microbench for the Wikipedia end-to-end latency figure
//! (Chapter 6). Measures the exact path the browser runs: extract one article's
//! compressed slice from a 128 KB row of `wikipedia.bin`, then brotli-decompress
//! that slice. Row format is `[4B u32 len][compressed bytes]` entries, zero-
//! padded to ROW_SIZE = 131072 (see `wikipedia/build/build_wiki_db.py`).
//!
//! Usage:
//!   cargo run --release --bin brotli_bench -- \
//!       --wiki-bin wikipedia/server/data/wikipedia.bin \
//!       [--row N] [--iters N] [--json OUT]
//!
//! Default: iterates over every article entry in the first row of the .bin,
//! times decompression of each, and reports median / p95 / throughput across
//! all entries x iters samples.

use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::time::Instant;

use brotli::Decompressor;
use clap::Parser;
use serde::Serialize;

const ROW_SIZE: usize = 128 * 1024;
const HEADER_BYTES: usize = 4;

#[derive(Parser, Debug)]
struct Args {
    /// Path to wikipedia.bin (flat DB, each row = ROW_SIZE bytes).
    #[clap(long)]
    wiki_bin: PathBuf,

    /// Number of rows to sample (evenly spaced through the file).
    #[clap(long, default_value_t = 30)]
    num_rows: usize,

    /// Iterations per article (timings aggregated across all rows x articles x iters).
    #[clap(long, default_value_t = 50)]
    iters: usize,

    /// Optional JSON output path.
    #[clap(long)]
    json: Option<PathBuf>,
}

#[derive(Serialize)]
struct Report {
    sample_source: String,
    row_size_bytes: usize,
    total_rows_in_file: usize,
    rows_sampled: usize,
    sampled_row_indices: Vec<usize>,
    total_articles: usize,
    median_compressed_bytes: usize,
    median_uncompressed_bytes: usize,
    iters_per_article: usize,
    total_samples: usize,
    median_decompress_us: f64,
    mean_decompress_us: f64,
    p95_decompress_us: f64,
    min_decompress_us: f64,
    throughput_mib_s: f64,
    engine: &'static str,
}

fn decompress(compressed: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut reader = Decompressor::new(compressed, 4096);
    reader.read_to_end(&mut out).expect("brotli decompress");
    out
}

fn read_row(path: &PathBuf, row: usize) -> Vec<u8> {
    let mut f = File::open(path).expect("open wiki_bin");
    use std::io::{Seek, SeekFrom};
    f.seek(SeekFrom::Start((row * ROW_SIZE) as u64)).expect("seek row");
    let mut buf = vec![0u8; ROW_SIZE];
    f.read_exact(&mut buf).expect("read row");
    buf
}

/// Parse all `[4B len][compressed]` entries in a 128 KB row, stopping when the
/// next length prefix is zero (padding) or runs past the row.
fn parse_articles(row: &[u8]) -> Vec<&[u8]> {
    let mut entries = Vec::new();
    let mut off = 0;
    while off + HEADER_BYTES <= row.len() {
        let len = u32::from_le_bytes([row[off], row[off+1], row[off+2], row[off+3]]) as usize;
        if len == 0 || off + HEADER_BYTES + len > row.len() {
            break;
        }
        entries.push(&row[off + HEADER_BYTES .. off + HEADER_BYTES + len]);
        off += HEADER_BYTES + len;
    }
    entries
}

fn percentile(mut xs: Vec<f64>, p: f64) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if xs.is_empty() { return 0.0; }
    let idx = ((p * (xs.len() as f64 - 1.0)).round() as usize).min(xs.len() - 1);
    xs[idx]
}

fn median_f64(xs: Vec<f64>) -> f64 { percentile(xs, 0.5) }
fn median_usize(xs: Vec<usize>) -> usize {
    if xs.is_empty() { return 0; }
    let mut xs = xs; xs.sort(); xs[xs.len() / 2]
}

fn main() {
    let args = Args::parse();

    // Figure out how many rows the file has; pick num_rows indices evenly
    // spaced across [0, total_rows) for broad corpus coverage.
    let file_len = std::fs::metadata(&args.wiki_bin).expect("stat wiki_bin").len() as usize;
    let total_rows = file_len / ROW_SIZE;
    if total_rows == 0 {
        eprintln!("{} smaller than one row", args.wiki_bin.display());
        std::process::exit(1);
    }
    let rows_to_sample = args.num_rows.min(total_rows);
    let sampled_indices: Vec<usize> = (0..rows_to_sample)
        .map(|i| i * total_rows / rows_to_sample)
        .collect();

    // Collect all article slices across sampled rows.
    let mut row_buffers: Vec<Vec<u8>> = Vec::with_capacity(rows_to_sample);
    for &r in &sampled_indices {
        row_buffers.push(read_row(&args.wiki_bin, r));
    }
    let articles: Vec<&[u8]> = row_buffers.iter().flat_map(|b| parse_articles(b)).collect();
    if articles.is_empty() {
        eprintln!("no articles parsed from any sampled row");
        std::process::exit(1);
    }

    // Warm up once per article so the icache / allocator settle.
    for a in &articles {
        let _ = decompress(a);
    }

    let mut samples_us: Vec<f64> = Vec::with_capacity(articles.len() * args.iters);
    let mut compressed_sizes: Vec<usize> = Vec::with_capacity(articles.len());
    let mut uncompressed_sizes: Vec<usize> = Vec::with_capacity(articles.len());
    for a in &articles {
        compressed_sizes.push(a.len());
        uncompressed_sizes.push(decompress(a).len());
        for _ in 0..args.iters {
            let t0 = Instant::now();
            let out = decompress(a);
            let dt = t0.elapsed().as_secs_f64() * 1.0e6;
            std::hint::black_box(out);
            samples_us.push(dt);
        }
    }

    let median_us = median_f64(samples_us.clone());
    let mean_us = samples_us.iter().sum::<f64>() / samples_us.len() as f64;
    let p95_us = percentile(samples_us.clone(), 0.95);
    let min_us = samples_us.iter().cloned().fold(f64::INFINITY, f64::min);
    let median_uncompressed = median_usize(uncompressed_sizes);
    let median_compressed = median_usize(compressed_sizes);
    let throughput_mib_s =
        (median_uncompressed as f64) / (median_us * 1.0e-6) / (1024.0 * 1024.0);

    let report = Report {
        sample_source: format!(
            "{} ({} rows sampled, {} articles)",
            args.wiki_bin.display(),
            rows_to_sample,
            articles.len(),
        ),
        row_size_bytes: ROW_SIZE,
        total_rows_in_file: total_rows,
        rows_sampled: rows_to_sample,
        sampled_row_indices: sampled_indices,
        total_articles: articles.len(),
        median_compressed_bytes: median_compressed,
        median_uncompressed_bytes: median_uncompressed,
        iters_per_article: args.iters,
        total_samples: samples_us.len(),
        median_decompress_us: median_us,
        mean_decompress_us: mean_us,
        p95_decompress_us: p95_us,
        min_decompress_us: min_us,
        throughput_mib_s,
        engine: "rust brotli crate (native)",
    };

    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    println!(
        "# {}: {} articles across {} rows, median {} B -> {} B, decompress median {:.1} us ({:.1} MiB/s), p95 {:.1} us",
        args.wiki_bin.display(),
        report.total_articles,
        report.rows_sampled,
        report.median_compressed_bytes,
        report.median_uncompressed_bytes,
        report.median_decompress_us,
        report.throughput_mib_s,
        report.p95_decompress_us,
    );

    if let Some(out_path) = args.json {
        let mut f = File::create(&out_path).expect("create json");
        f.write_all(serde_json::to_string_pretty(&report).unwrap().as_bytes())
            .expect("write json");
        eprintln!("# wrote {}", out_path.display());
    }
}
