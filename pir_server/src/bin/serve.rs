//! Thin HTTP wrapper around PirServer.
//!
//!   GET  /api/info   — server params
//!   POST /api/query  — stateless PIR query
//!   GET  /api/direct — dev: cleartext row fetch
//!   GET  /*          — static file serving

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Arc;

use clap::Parser;
use log::{debug, info, warn};

use pir_server::PirServer;

#[derive(Parser, Debug)]
#[command(version, about = "SandwichPIR HTTP server")]
struct Args {
    /// Path to flat database file
    #[clap(long)]
    db: String,

    /// Number of items (rows)
    #[clap(long)]
    num_items: usize,

    /// Item size in bits
    #[clap(long)]
    item_size_bits: usize,

    /// Listen address
    #[clap(long, default_value = "0.0.0.0:8080")]
    listen: String,

    /// Static file directory (optional)
    #[clap(long)]
    web_dir: Option<String>,

    /// Verbose
    #[clap(long, short, action)]
    verbose: bool,
}

fn main() {
    let args = Args::parse();

    if args.verbose {
        env_logger::Builder::new().filter_level(log::LevelFilter::Debug).init();
    } else {
        env_logger::Builder::new().filter_level(log::LevelFilter::Info).init();
    }

    // Mmap the DB so the peak host memory during construction is
    // ~8 GB (YServer's aligned buffer) instead of ~16 GB (caller Vec +
    // aligned). Fits on 16 GB boxes.
    let server = Arc::new(
        PirServer::new_from_file(
            std::path::Path::new(&args.db),
            args.num_items,
            args.item_size_bits,
        )
        .expect("Failed to open DB file"),
    );

    info!("Listening on {}", args.listen);
    let listener = TcpListener::bind(&args.listen).expect("Failed to bind");

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let server = Arc::clone(&server);
                let web_dir = args.web_dir.clone();
                std::thread::spawn(move || {
                    if let Err(e) = handle(stream, &server, web_dir.as_deref()) {
                        debug!("Request error: {}", e);
                    }
                });
            }
            Err(e) => warn!("Accept error: {}", e),
        }
    }
}

fn handle(
    mut stream: std::net::TcpStream,
    server: &PirServer,
    web_dir: Option<&str>,
) -> Result<(), String> {
    let (method, path, body) = read_request(&mut stream)?;

    match (method.as_str(), path.as_str()) {
        ("OPTIONS", _) => {
            let h = "HTTP/1.1 200 OK\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type\r\nContent-Length: 0\r\n\r\n";
            stream.write_all(h.as_bytes()).map_err(|e| e.to_string())?;
        }

        ("GET", "/api/info") => {
            let body = serde_json::to_string(&server.info).unwrap();
            respond(&mut stream, 200, "application/json", body.as_bytes());
        }

        ("POST", "/api/query") => {
            let start = std::time::Instant::now();
            match server.answer(&body) {
                Ok(resp) => {
                    info!("Query: {} ms, {} bytes", start.elapsed().as_millis(), resp.len());
                    respond(&mut stream, 200, "application/octet-stream", &resp);
                }
                Err(e) => {
                    warn!("Query failed: {}", e);
                    respond(&mut stream, 400, "text/plain", e.as_bytes());
                }
            }
        }

        ("GET", p) if p.starts_with("/api/direct") => {
            let row: usize = p.split("row=").nth(1)
                .and_then(|s| s.split('&').next())
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            let data = server.get_row_direct(row);
            respond(&mut stream, 200, "application/octet-stream", &data);
        }

        ("GET", p) => {
            if let Some(dir) = web_dir {
                let file = if p == "/" { format!("{}/index.html", dir) } else { format!("{}{}", dir, p) };
                match std::fs::read(&file) {
                    Ok(data) => {
                        let ct = match file.rsplit('.').next() {
                            Some("html") => "text/html",
                            Some("js") => "application/javascript",
                            Some("css") => "text/css",
                            Some("wasm") => "application/wasm",
                            Some("json") => "application/json",
                            _ => "application/octet-stream",
                        };
                        respond(&mut stream, 200, ct, &data);
                    }
                    Err(_) => respond(&mut stream, 404, "text/plain", b"Not Found"),
                }
            } else {
                respond(&mut stream, 404, "text/plain", b"No web directory configured");
            }
        }

        _ => respond(&mut stream, 405, "text/plain", b"Method Not Allowed"),
    }
    Ok(())
}

fn read_request(stream: &mut std::net::TcpStream) -> Result<(String, String, Vec<u8>), String> {
    let mut header_buf = Vec::with_capacity(4096);
    let mut byte = [0u8; 1];
    loop {
        stream.read_exact(&mut byte).map_err(|e| e.to_string())?;
        header_buf.push(byte[0]);
        if header_buf.len() >= 4 && &header_buf[header_buf.len() - 4..] == b"\r\n\r\n" { break; }
        if header_buf.len() > 16384 { return Err("Headers too large".into()); }
    }
    let header_str = String::from_utf8_lossy(&header_buf).to_string();
    let first_line = header_str.lines().next().unwrap_or("");
    let parts: Vec<&str> = first_line.split_whitespace().collect();
    if parts.len() < 2 { return Err("Bad request".into()); }

    let content_length: usize = header_str.lines()
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
    let status_text = match status { 200 => "OK", 400 => "Bad Request", 404 => "Not Found", 405 => "Method Not Allowed", _ => "Error" };
    let header = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\n\r\n",
        status, status_text, content_type, body.len()
    );
    let _ = stream.write_all(header.as_bytes());
    let _ = stream.write_all(body);
}
