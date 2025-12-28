const wasmCode = await Deno.readFile("output.wasm");

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
    }
};

try {
    const wasmModule = new WebAssembly.Module(wasmCode);
    const wasmInstance = new WebAssembly.Instance(wasmModule, importObject);

    console.log("Running WASM...");
    // _start is exported by the compiler
    if (typeof wasmInstance.exports._start === "function") {
        wasmInstance.exports._start();
    } else {
        console.error("Error: _start function not found in exports");
    }
    console.log("Done.");
} catch (e) {
    console.error("Runtime Error:", e);
}
