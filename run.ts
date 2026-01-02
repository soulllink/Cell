const filename = Deno.args[0] || "output.wasm";
const wasmCode = await Deno.readFile(filename);

let wasmInstance: WebAssembly.Instance;

const importObject = {
    env: {
        put: (val) => { console.log("Output:", val); return val; },
        input: () => {
            const val = prompt("Input number > ");
            return val ? parseFloat(val) : 0.0;
        },
        pow: (base, exp) => Math.pow(base, exp),
        log: (val) => Math.log(val),
        rand: (min, max) => Math.floor(Math.random() * (max - min + 1)) + min,
        print: (ptr, len) => {
            const mem = new Uint8Array(wasmInstance.exports.memory.buffer);
            const bytes = mem.slice(ptr, ptr + len);
            const str = new TextDecoder().decode(bytes);
            console.log("Output:", str);
        },
        sin: Math.sin, cos: Math.cos, tan: Math.tan,
        asin: Math.asin, acos: Math.acos, atan: Math.atan,
        hypot: Math.hypot,
        fmod: (x, y) => x % y,
    }
};

try {
    const wasmModule = new WebAssembly.Module(wasmCode);
    wasmInstance = new WebAssembly.Instance(wasmModule, importObject);

    console.log("Running WASM...");
    // _start is exported by the compiler
    // The compiler exports "run" as the entry point
    const runFunc = wasmInstance.exports.run as CallableFunction;
    if (typeof runFunc === "function") {
        runFunc();
    } else {
        console.error("Error: 'run' function not found in exports");
    }
    console.log("Done.");
} catch (e) {
    console.error("Runtime Error:", e);
}
