// SpiralPack benchmark driver (menonsamir/spiral-rs @ 2760622e).
//
// spiral-rs ships no binary; this drives its public API exactly as the crate's
// own `full_protocol_is_correct_for_params` end-to-end test does
// (src/server.rs), adding timing, communication measurement, and a JSON report.
//
// We use the SpiralPack (compact-query) configuration: the client sends a small
// query that the server EXPANDS via automorphisms (expand_queries = true, set by
// the t_exp_* params). This is the logN-communication + reusable-client-keys
// design point -- NOT SpiralStream, whose un-expanded query is linear-sized.
//
// CLI: spiral-bench <nu_1> <nu_2> <db_item_size_bytes> <trials> [warmup]
//   num_items = 2^(nu_1 + nu_2); DB bytes = num_items * db_item_size.
//   Record size is fixed by db_item_size (native, no ring-dimension change).
//
// A correctness assertion runs on the first trial: if the chosen (nu_1, nu_2)
// blow the Spiral noise budget, the run panics rather than reporting a wrong
// number. Fields reported: per-trial server seconds, per-query query/response
// bytes, and the one-time reusable public-key (client offline keys) bytes.

use spiral_rs::client::*;
use spiral_rs::server::*;
use spiral_rs::util::params_from_json;
use std::env;
use std::time::Instant;

fn main() {
    let a: Vec<String> = env::args().collect();
    if a.len() < 5 {
        eprintln!("usage: spiral-bench <nu_1> <nu_2> <db_item_size_bytes> <trials> [warmup]");
        std::process::exit(2);
    }
    let nu_1: usize = a[1].parse().expect("nu_1");
    let nu_2: usize = a[2].parse().expect("nu_2");
    let item: usize = a[3].parse().expect("db_item_size");
    let trials: usize = a[4].parse().expect("trials");
    let warmup: usize = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(1);

    // Base parameters are the crate's validated 32 KiB config
    // (server.rs `larger_full_protocol_is_correct`): n=2 (matrix dim),
    // instances=4 -> item = instances * n^2 * poly_len * logp / 8. Only nu_1/nu_2
    // and db_item_size vary here; nu_2 (the GSW-folded further dimensions) is held
    // and nu_1 (the first-dimension linear scan) grows the DB, which keeps the
    // noise growth bounded as the database scales.
    let cfg = format!(
        r#"{{ "n": 2, "nu_1": {}, "nu_2": {}, "p": 256, "q2_bits": 22, "t_gsw": 7, "t_conv": 3, "t_exp_left": 5, "t_exp_right": 5, "instances": 4, "db_item_size": {} }}"#,
        nu_1, nu_2, item
    );
    let params = params_from_json(&cfg);

    let num_items = 1usize << (params.db_dim_1 + params.db_dim_2);
    let db_bytes = num_items * params.db_item_size;
    eprintln!(
        "SpiralPack: nu_1={} nu_2={} num_items={} item={}B db={:.3}GiB expand_queries={}",
        params.db_dim_1,
        params.db_dim_2,
        num_items,
        params.db_item_size,
        db_bytes as f64 / (1u64 << 30) as f64,
        params.expand_queries,
    );
    assert!(
        params.expand_queries,
        "expand_queries must be true (SpiralPack/compact query); check t_exp_* params"
    );

    // Client one-time setup: reusable public keys (query-expansion + conversion
    // keys), uploaded once. generate_keys() returns NTT-form keys that
    // process_query uses directly. We measure serialize().len() for the true wire
    // (client-upload) size but do NOT round-trip through deserialize: spiral-rs's
    // deserialize/setup_bytes() count num_packing_mats as 2 (params.rs) while
    // generate_keys_impl uses 1 (client.rs), so the roundtrip's size assertion
    // fails for expand configs. The original pp is the internally-consistent
    // object for process_query; the roundtrip is irrelevant to server compute or
    // comm measurement, which is all this benchmark needs.
    let mut client = Client::init(&params);
    let pp = client.generate_keys();
    let key_bytes = pp.serialize().len();

    let target_idx = 7 % num_items;
    let query = client.generate_query(target_idx);
    let query_bytes = query.serialize().len();

    let (corr_item, db) = generate_random_db_and_get_item(&params, target_idx);

    let mut times: Vec<f64> = Vec::with_capacity(trials);
    let mut resp_bytes = 0usize;
    for t in 0..(trials + warmup) {
        let start = Instant::now();
        let response = process_query(&params, &pp, &query, db.as_slice());
        let dt = start.elapsed().as_secs_f64();
        resp_bytes = response.len();

        if t == 0 {
            // Correctness: decode and compare to the known record. A noise-budget
            // failure at this (nu_1, nu_2) shows up as a panic here.
            let result = client.decode_response(response.as_slice(), 1);
            let p_bits = (params.pt_modulus as f64).log2().ceil() as usize;
            let corr = corr_item.to_vec(p_bits, params.modp_words_per_chunk());
            assert_eq!(result.len(), corr.len(), "decode length mismatch");
            for z in 0..corr.len() {
                assert_eq!(result[z], corr[z], "MISMATCH at index {}", z);
            }
            eprintln!("correctness OK: recovered {} record bytes", result.len());
        }
        if t >= warmup {
            times.push(dt);
        }
    }

    let times_str: Vec<String> = times.iter().map(|t| format!("{:.6}", t)).collect();
    println!("{{");
    println!("  \"scheme\": \"spiral_pack\",");
    println!("  \"num_items\": {},", num_items);
    println!("  \"db_item_size\": {},", params.db_item_size);
    println!("  \"db_bytes\": {},", db_bytes);
    println!("  \"nu_1\": {},", params.db_dim_1);
    println!("  \"nu_2\": {},", params.db_dim_2);
    println!("  \"server_seconds\": [{}],", times_str.join(", "));
    println!("  \"query_bytes\": {},", query_bytes);
    println!("  \"response_bytes\": {},", resp_bytes);
    println!("  \"key_bytes\": {}", key_bytes);
    println!("}}");
}
