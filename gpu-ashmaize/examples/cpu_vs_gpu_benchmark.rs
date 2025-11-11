/**
 * Comprehensive CPU vs GPU Performance Benchmark
 * 
 * Compares single-core CPU, multi-core CPU, and GPU performance
 * for AshMaize proof-of-work mining
 */

use ashmaize::{Rom, RomGenerationType, hash};
use gpu_ashmaize::{GpuMiner, hash_batch};
use std::time::Instant;
use std::thread;
use std::sync::Arc;

fn format_duration(secs: f64) -> String {
    if secs < 1.0 {
        format!("{:.1} ms", secs * 1000.0)
    } else if secs < 60.0 {
        format!("{:.2} sec", secs)
    } else {
        format!("{:.1} min", secs / 60.0)
    }
}

fn benchmark_cpu_single_core(
    rom: &Rom,
    count: usize,
    nb_loops: u32,
    nb_instrs: u32
) -> (f64, f64) {
    println!("  Warming up...");
    let _ = hash(b"warmup", rom, nb_loops, nb_instrs);
    
    println!("  Running {} hashes...", count);
    let start = Instant::now();
    
    for i in 0..count {
        let salt = (i as u64).to_le_bytes();
        let _ = hash(&salt, rom, nb_loops, nb_instrs);
    }
    
    let elapsed = start.elapsed().as_secs_f64();
    let throughput = count as f64 / elapsed;
    
    (throughput, elapsed)
}

fn benchmark_cpu_multi_core(
    rom: Arc<Rom>,
    num_threads: usize,
    count_per_thread: usize,
    nb_loops: u32,
    nb_instrs: u32
) -> (f64, f64) {
    println!("  Spawning {} threads...", num_threads);
    let start = Instant::now();
    
    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let rom = Arc::clone(&rom);
            let salt_base = (thread_id as u64) * (count_per_thread as u64);
            
            thread::spawn(move || {
                for i in 0..count_per_thread {
                    let salt = (salt_base + i as u64).to_le_bytes();
                    let _ = hash(&salt, &rom, nb_loops, nb_instrs);
                }
            })
        })
        .collect();
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    let elapsed = start.elapsed().as_secs_f64();
    let total_hashes = num_threads * count_per_thread;
    let throughput = total_hashes as f64 / elapsed;
    
    (throughput, elapsed)
}

fn benchmark_gpu_batch(
    rom: &Rom,
    batch_size: usize,
    nb_loops: u32,
    nb_instrs: u32
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    println!("  Preparing batch of {} nonces...", batch_size);
    let salts: Vec<_> = (0..batch_size)
        .map(|i| (i as u64).to_le_bytes().to_vec())
        .collect();
    
    // Warm-up
    println!("  Warming up GPU...");
    let small_batch: Vec<_> = salts.iter().take(256).cloned().collect();
    let _ = hash_batch(&small_batch, rom, nb_loops, nb_instrs)?;
    
    println!("  Running GPU batch...");
    let start = Instant::now();
    
    let _ = hash_batch(&salts, rom, nb_loops, nb_instrs)?;
    
    let elapsed = start.elapsed().as_secs_f64();
    let throughput = batch_size as f64 / elapsed;
    
    Ok((throughput, elapsed))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  AshMaize CPU vs GPU Performance Benchmark                ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();
    
    // System info
    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    println!("System Information:");
    println!("  CPU Cores: {}", num_cpus);
    
    // Try to get GPU info (will fail gracefully if no GPU)
    match GpuMiner::with_params(8, 256) {
        Ok(_) => println!("  GPU: Available (CUDA)"),
        Err(_) => {
            println!("  GPU: Not available");
            println!("\nCannot run GPU benchmarks without CUDA GPU");
            return Ok(());
        }
    }
    
    println!();
    
    // Parameters
    const MB: usize = 1024 * 1024;
    let rom_size = 10 * MB;
    let nb_loops = 8;
    let nb_instrs = 256;
    
    println!("Benchmark Parameters:");
    println!("  ROM size: {} MB", rom_size / MB);
    println!("  Loops per hash: {}", nb_loops);
    println!("  Instructions per loop: {}", nb_instrs);
    println!();
    
    // Generate ROM
    println!("Generating ROM ({} MB)...", rom_size / MB);
    let rom_start = Instant::now();
    let rom = Rom::new(
        b"benchmark_seed_2025",
        RomGenerationType::TwoStep {
            pre_size: 16 * 1024,
            mixing_numbers: 4,
        },
        rom_size,
    );
    let rom_time = rom_start.elapsed();
    println!("ROM generated in {:.2} seconds", rom_time.as_secs_f64());
    println!();
    
    let rom = Arc::new(rom);
    
    // ═══════════════════════════════════════════════════════════════
    // CPU Single-Core Benchmark
    // ═══════════════════════════════════════════════════════════════
    println!("┌────────────────────────────────────────────────────────────┐");
    println!("│ CPU Single-Core Benchmark                                  │");
    println!("└────────────────────────────────────────────────────────────┘");
    
    let cpu_single_count = 100;
    let (cpu_single_throughput, cpu_single_time) = benchmark_cpu_single_core(
        &rom, cpu_single_count, nb_loops, nb_instrs
    );
    
    println!();
    println!("Results:");
    println!("  Hashes computed: {}", cpu_single_count);
    println!("  Time taken: {}", format_duration(cpu_single_time));
    println!("  Throughput: {:.2} hash/sec", cpu_single_throughput);
    println!("  Time per hash: {:.2} ms", (cpu_single_time / cpu_single_count as f64) * 1000.0);
    println!();
    
    // ═══════════════════════════════════════════════════════════════
    // CPU Multi-Core Benchmark
    // ═══════════════════════════════════════════════════════════════
    println!("┌────────────────────────────────────────────────────────────┐");
    println!("│ CPU Multi-Core Benchmark ({} cores)                       │", num_cpus);
    println!("└────────────────────────────────────────────────────────────┘");
    
    let cpu_multi_count_per_thread = 100;
    let (cpu_multi_throughput, cpu_multi_time) = benchmark_cpu_multi_core(
        Arc::clone(&rom), num_cpus, cpu_multi_count_per_thread, nb_loops, nb_instrs
    );
    
    let cpu_total_hashes = num_cpus * cpu_multi_count_per_thread;
    println!();
    println!("Results:");
    println!("  Threads: {}", num_cpus);
    println!("  Hashes per thread: {}", cpu_multi_count_per_thread);
    println!("  Total hashes: {}", cpu_total_hashes);
    println!("  Time taken: {}", format_duration(cpu_multi_time));
    println!("  Throughput: {:.2} hash/sec", cpu_multi_throughput);
    println!("  Scaling efficiency: {:.1}%", 
             (cpu_multi_throughput / cpu_single_throughput / num_cpus as f64) * 100.0);
    println!();
    
    // ═══════════════════════════════════════════════════════════════
    // GPU Benchmark - Multiple Batch Sizes
    // ═══════════════════════════════════════════════════════════════
    println!("┌────────────────────────────────────────────────────────────┐");
    println!("│ GPU Batch Processing Benchmark                             │");
    println!("└────────────────────────────────────────────────────────────┘");
    println!();
    
    let batch_sizes = vec![256, 1024, 4096, 16384, 65536, 262144];
    
    println!("{:<12} {:<15} {:<15} {:<15} {:<15}", 
             "Batch Size", "Time", "Hash/sec", "vs 1-Core", format!("vs {}-Core", num_cpus));
    println!("{}", "─".repeat(75));
    
    let mut best_throughput = 0.0;
    let mut best_batch_size = 0;
    
    for &batch_size in &batch_sizes {
        match benchmark_gpu_batch(&rom, batch_size, nb_loops, nb_instrs) {
            Ok((throughput, time)) => {
                let vs_single = throughput / cpu_single_throughput;
                let vs_multi = throughput / cpu_multi_throughput;
                
                println!("{:<12} {:<15} {:<15.2} {:<15.2}x {:<15.2}x",
                         format!("{}", batch_size),
                         format_duration(time),
                         throughput,
                         vs_single,
                         vs_multi);
                
                if throughput > best_throughput {
                    best_throughput = throughput;
                    best_batch_size = batch_size;
                }
            }
            Err(e) => {
                println!("{:<12} ERROR: {}", batch_size, e);
            }
        }
    }
    
    println!();
    
    // ═══════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════
    println!("┌────────────────────────────────────────────────────────────┐");
    println!("│ Performance Summary                                        │");
    println!("└────────────────────────────────────────────────────────────┘");
    println!();
    
    println!("Throughput Comparison:");
    println!("  CPU (1 core):     {:>10.2} hash/sec", cpu_single_throughput);
    println!("  CPU ({} cores):   {:>10.2} hash/sec", num_cpus, cpu_multi_throughput);
    println!("  GPU (best):       {:>10.2} hash/sec (batch size: {})", 
             best_throughput, best_batch_size);
    println!();
    
    println!("Speedup vs Single-Core CPU:");
    println!("  CPU {} cores:     {:>10.2}x", num_cpus, 
             cpu_multi_throughput / cpu_single_throughput);
    println!("  GPU:              {:>10.2}x", 
             best_throughput / cpu_single_throughput);
    println!();
    
    println!("Speedup vs {}-Core CPU:", num_cpus);
    println!("  GPU:              {:>10.2}x", 
             best_throughput / cpu_multi_throughput);
    println!();
    
    println!("Time to Mine 1 Million Nonces:");
    let million = 1_000_000.0;
    println!("  CPU (1 core):     {}", 
             format_duration(million / cpu_single_throughput));
    println!("  CPU ({} cores):   {}", num_cpus,
             format_duration(million / cpu_multi_throughput));
    println!("  GPU:              {}", 
             format_duration(million / best_throughput));
    println!();
    
    println!("Optimal Configuration:");
    println!("  For mining: Use GPU with batch size {}", best_batch_size);
    println!("  For single hashes: Use CPU");
    println!("  For small batches (<1000): Use CPU multi-threaded");
    println!();
    
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  Benchmark Complete                                        ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    
    Ok(())
}
