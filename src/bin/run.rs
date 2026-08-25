use sandwichpir::scheme::run_sandwichpir_batched;

use clap::Parser;

/// Run SandwichPIR with the given parameters
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Number of items in the database
    num_items: usize,
    /// Size of each item in bits
    item_size_bits: usize,
    /// Number of clients (batch size, default 1)
    num_clients: Option<usize>,
    /// Number of trials (default 5)
    trials: Option<usize>,
    /// Output report file (JSON)
    out_report_json: Option<String>,
    /// Verbose mode (debug logging)
    #[clap(long, short, action)]
    verbose: bool,
}

fn main() {
    let args = Args::parse();

    if args.verbose {
        println!("Running in verbose mode.");
        env_logger::Builder::new()
            .filter_level(log::LevelFilter::Debug)
            .write_style(env_logger::WriteStyle::Always)
            .init();
    } else {
        env_logger::init();
    }

    let num_clients = args.num_clients.unwrap_or(1);
    let trials = args.trials.unwrap_or(5);

    #[cfg(feature = "cuda")]
    let num_gpus = std::env::var("MULTI_GPU")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(1);

    #[cfg(feature = "cuda")]
    let backend = format!("{} GPU{}", num_gpus, if num_gpus == 1 { "" } else { "s" });
    #[cfg(not(feature = "cuda"))]
    let backend = "CPU".to_string();

    println!(
        "Running SandwichPIR on {} items x {} bits = {} bytes, batching {} clients, {} trials, on {}.",
        args.num_items,
        args.item_size_bits,
        args.num_items * args.item_size_bits / 8,
        num_clients,
        trials,
        backend,
    );

    let result = run_sandwichpir_batched(args.num_items, args.item_size_bits, num_clients, trials);
    println!("{}", serde_json::to_string_pretty(&result).unwrap());

    if let Some(out_report_json) = args.out_report_json {
        let mut file = std::fs::File::create(&out_report_json).unwrap();
        serde_json::to_writer_pretty(&mut file, &result).unwrap();
        println!("Report written to {}", out_report_json);
    }

    // Explicitly drain Rust's stdout BufWriter before exit so the
    // JSON block is fully visible in the output. `setvbuf(_IONBF)` on
    // libc's stdout earlier handles the C side; this handles the
    // Rust side. Together they guarantee no partial-write truncation
    // when the process exits.
    use std::io::Write;
    let _ = std::io::stdout().flush();
}
