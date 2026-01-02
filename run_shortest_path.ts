const wasmCode = await Deno.readFile("examples/shortest_path.wasm");

let wasmInstance: WebAssembly.Instance;

const importObject = {
    env: {
        put: (val: number) => { console.log("Output:", val); return val; },
        input: () => 0.0,
        pow: (base: number, exp: number) => Math.pow(base, exp),
        log: (val: number) => Math.log(val),
        rand: (min: number, max: number) => Math.floor(Math.random() * (max - min + 1)) + min,
        print: (ptr: number, len: number) => {
            const mem = new Uint8Array((wasmInstance.exports.memory as WebAssembly.Memory).buffer);
            const bytes = mem.slice(ptr, ptr + len);
            const str = new TextDecoder().decode(bytes);
            console.log("Output:", str);
        },
        sin: Math.sin,
        cos: Math.cos,
        tan: Math.tan,
        asin: Math.asin,
        acos: Math.acos,
        atan: Math.atan,
        hypot: Math.hypot,
        fmod: (x: number, y: number) => x % y
    }
};

try {
    const wasmModule = new WebAssembly.Module(wasmCode);
    wasmInstance = new WebAssembly.Instance(wasmModule, importObject);

    const memory = wasmInstance.exports.memory as WebAssembly.Memory;
    const f64 = new Float64Array(memory.buffer);
    const max_col = 12;
    const max_row = 23;

    // Call init manually if possible? No, it's internal.
    // But 'run' calls it.
    console.log("Running WASM...");
    if (typeof wasmInstance.exports.run === "function") {
        (wasmInstance.exports.run as () => void)();
    }

    console.log("Final Grid Samples:");
    for (let r = 0; r < 5; r++) {
        let line = "";
        for (let c = 0; c < 5; c++) {
            line += f64[r * max_col + c] + "\t";
        }
        console.log(line);
    }

    const val = f64[22 * max_col + 9];
    console.log(`Value at J23 (9, 22): ${val}`);

    console.log("Done.");
} catch (e) {
    console.error("Runtime Error:", e);
}
