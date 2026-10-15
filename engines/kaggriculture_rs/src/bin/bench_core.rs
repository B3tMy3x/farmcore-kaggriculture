use kaggriculture_rs::{engine::Engine, model::Config};
use rayon::prelude::*;
use serde_json::json;
use std::io::Write;
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("1000")
        .parse()
        .unwrap();
    let threads: usize = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("1")
        .parse()
        .unwrap();
    let output = args
        .iter()
        .position(|a| a == "--verify-output")
        .map(|i| args.get(i + 1).expect("--verify-output needs a path"));
    assert!(n > 0 && threads > 0);
    let c = Config::parse(&json!({})).unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let a = std::array::from_fn(|_| json!({"farmer":["PASS"]}));
    let start = std::time::Instant::now();
    let engines: Vec<Engine> = pool.install(|| {
        (0..n)
            .into_par_iter()
            .map(|i| {
                let mut e = Engine::new(c.clone(), i.into());
                while !e.done {
                    e.advance(&a);
                }
                e
            })
            .collect()
    });
    let seconds = start.elapsed().as_secs_f64();
    let checksum: f64 = engines.iter().map(|e| e.farms[0].money).sum();
    // Snapshot serialization is outside the measured simulation interval.
    // benchmarks/run.py compares every record's canonical bytes with Python.
    if let Some(path) = output {
        let mut out = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
        for (seed, e) in engines.iter().enumerate() {
            serde_json::to_writer(&mut out, &json!({"seed":seed,"result":e.result()})).unwrap();
            writeln!(out).unwrap();
        }
    }
    println!(
        "{}",
        json!({"episodes":n,"threads":threads,"seconds":seconds,
        "steps_per_second":n as f64*719./seconds,"checksum":checksum,
        "snapshot_output":output})
    );
}
