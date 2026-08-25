//! Native PIR client CLI.
//!
//! Usage: pir-query --server localhost:8080 --row 42

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Instant;

use clap::Parser;
use log::info;
use pir_client::PirClient;

#[derive(Parser, Debug)]
#[command(version, about = "SandwichPIR native client")]
struct Args {
    /// Server host:port (e.g. localhost:8080)
    #[clap(long)]
    server: String,

    /// Row index to fetch
    #[clap(long)]
    row: usize,

    /// Output file (raw row bytes)
    #[clap(long, short)]
    output: Option<String>,

    /// Verbose
    #[clap(long, short, action)]
    verbose: bool,
}

fn http_get_json(addr: &str, path: &str) -> serde_json::Value {
    let mut stream = TcpStream::connect(addr).expect("Failed to connect");
    let req = format!("GET {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n", path, addr);
    stream.write_all(req.as_bytes()).unwrap();
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).unwrap();
    let text = String::from_utf8_lossy(&buf);
    let body_start = text.find("\r\n\r\n").unwrap() + 4;
    serde_json::from_str(&text[body_start..]).expect("Bad JSON from server")
}

fn http_post_bytes(addr: &str, path: &str, body: &[u8]) -> Vec<u8> {
    let mut stream = TcpStream::connect(addr).expect("Failed to connect");
    let req = format!(
        "POST {} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/octet-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        path, addr, body.len()
    );
    stream.write_all(req.as_bytes()).unwrap();
    stream.write_all(body).unwrap();

    // Read response
    let mut resp = Vec::new();
    stream.read_to_end(&mut resp).unwrap();
    let text = String::from_utf8_lossy(&resp);
    let body_start = text.find("\r\n\r\n").map(|i| i + 4).unwrap_or(0);
    resp[body_start..].to_vec()
}

fn main() {
    let args = Args::parse();

    if args.verbose {
        env_logger::Builder::new().filter_level(log::LevelFilter::Debug).init();
    } else {
        env_logger::Builder::new().filter_level(log::LevelFilter::Info).init();
    }

    // Fetch server info
    info!("Fetching server info from {}...", args.server);
    let info = http_get_json(&args.server, "/api/info");

    let num_items = info["numItems"].as_u64().unwrap() as usize;
    let item_size_bytes = info["itemSizeBytes"].as_u64().unwrap() as usize;
    let item_size_bits = item_size_bytes * 8;

    info!("Server: {} items x {} bytes", num_items, item_size_bytes);

    // Create PIR client
    let mut client = PirClient::new(num_items, item_size_bits);
    info!("Generating query for row {}...", args.row);

    let start = Instant::now();
    let payload = client.query(args.row);
    info!("Query generated in {} ms ({} bytes)", start.elapsed().as_millis(), payload.len());

    // Send query
    info!("Sending query to server...");
    let start = Instant::now();
    let response_bytes = http_post_bytes(&args.server, "/api/query", &payload);
    info!("Response received in {} ms ({} bytes)", start.elapsed().as_millis(), response_bytes.len());

    // Decode
    let start = Instant::now();
    let row_data = client.decode(&response_bytes);
    info!("Decoded in {} ms ({} bytes)", start.elapsed().as_millis(), row_data.len());

    if let Some(output) = args.output {
        std::fs::write(&output, &row_data).expect("Failed to write output");
        info!("Wrote {} bytes to {}", row_data.len(), output);
    } else {
        let preview = String::from_utf8_lossy(&row_data[..row_data.len().min(200)]);
        println!("{}", preview);
    }
}
