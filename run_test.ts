const wasmCode = await Deno.readFile("quake.wasm");

let wasmInstance: WebAssembly.Instance;

const importObject = {
    env: {
        put: (val: number) => { console.log("Output:", val); return val; },
        input: () => {
            return 0.0; // No input for this test
        },
        pow: (base: number, exp: number) => Math.pow(base, exp),
        log: (val: number) => Math.log(val),
        rand: (min: number, max: number) => Math.floor(Math.random() * (max - min + 1)) + min,
        print: (ptr: number, len: number) => {
            const mem = new Uint8Array((wasmInstance.exports.memory as WebAssembly.Memory).buffer);
            const bytes = mem.slice(ptr, ptr + len);
            const str = new TextDecoder().decode(bytes);
            console.log("Output:", str);
        },
        // Trig imports
        sin: Math.sin,
        cos: Math.cos,
        tan: Math.tan,
        asin: Math.asin,
        acos: Math.acos,
        atan: Math.atan,
        hypot: Math.hypot
    }
};

try {
    const wasmModule = new WebAssembly.Module(wasmCode);
    wasmInstance = new WebAssembly.Instance(wasmModule, importObject);

    console.log("Running WASM (test_output.wasm)...");
    if (typeof wasmInstance.exports._start === "function") {
        (wasmInstance.exports._start as () => void)();
    } else {
        console.error("Error: _start function not found in exports");
    }
    console.log("Done.");
} catch (e) {
    console.error("Runtime Error:", e);
}
