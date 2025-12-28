mod loader;
mod parser;
mod codegen;

use clap::Parser;
use anyhow::Result;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input ODS file path
    input: String,

    /// Output WASM file path
    #[arg(short, long, default_value = "output.wasm")]
    output: String,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {:?}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    // Check if filename provided as bare argument (not -i)
    // The previous Args struct might require -i. Let's make it smarter or just check args.
    // The previous run command was `cargo run -- Example_GuessGame_Cell.ods`.
    // But Args defines `input` with `#[arg(short, long)]`.
    // So current usage should be `cargo run -- -i Example_GuessGame_Cell.ods`.
    // Or we update Args to make `input` positional.
    
    // Let's update struct Args to be positional index for input if we want convenience.
    // But keeping it as is:
    
    let args = Args::parse();

    println!("Compiling Cell project...");
    println!("Input: {}", args.input);
    println!("Output: {}", args.output);

    // 1. Load ODS
    let grid = loader::load_ods(&args.input).map_err(|e| anyhow::anyhow!("Failed to load ODS: {}", e))?;
    println!("Loaded grid with max_col: {}, max_row: {}", grid.max_col, grid.max_row);

    // 2. Start Codegen (which includes parsing)
    let mut compiler = codegen::WasmCompiler::new(&grid);
    let wasm_bytes = compiler.generate().map_err(|e| anyhow::anyhow!("Codegen failed: {}", e))?;
    
    // 3. Write Output
    std::fs::write(&args.output, wasm_bytes).map_err(|e| anyhow::anyhow!("Failed to write output: {}", e))?;
    println!("Successfully wrote WASM to {}", args.output);

    println!("Done.");
    Ok(())
}
