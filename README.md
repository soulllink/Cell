# Cell

**Cell** is a powerful compiler that transforms OpenDocument Spreadsheets (`.ods`) into highly optimized, standalone WebAssembly (`.wasm`) binaries.

Unlike traditional spreadsheet engines that interpret formulas at runtime, Cell compiles the logic within your spreadsheet cells into machine-code-ready WebAssembly. This enables ultra-low-latency calculations, making it ideal for running complex financial models, physics simulations, or game logic directly in the browser or on the edge.

## Key Features

*   **Zero Overhead**: Formulas are compiled, not interpreted.
*   **Advanced Scripting**: Write full procedural code inside cells (Loops, Variables, Functions).
*   **APL-Inspired Power**: Access advanced mathematical operators for array manipulation and complex arithmetic.
*   **WebAssembly Native**: Outputs standard `.wasm` files compatible with any WASM runtime (JS/Browser, Wasmtime, etc.).

---

## ⚡ The Cell Language

Cell treats every cell in your spreadsheet as a potential script. While simple cells can contain just numbers or basic assignments, you can also write full programs using Cell's scripting syntax.

### Data Types
Currently, Cell primarily operates on **64-bit Floating Point Numbers (`f64`)**.
*   **Numbers**: `42`, `3.14`, `-100`
*   **Booleans**: `true` (1.0), `false` (0.0)
*   **Nil**: `nil` (0.0)

### Variables & Scope
Variables are created via assignment.
*   **Local Scope**: Variables declared inside `fn`, `do...end`, or loops are local to that block.
*   **Global/Cell Scope**: Variables defined at the top level of a cell are accessible globally if exported (implementation dependent) or within the cell's execution context.

```ruby
x = 10
y = x times 2
```

### Operators

Cell uses a mix of standard symbols and English keywords. **Note the APL-inspired behavior for `*`**.

| Operator | Keyword | Symbol | Description | Example | Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Add** | `plus` | `+` | Addition | `2 + 3` | `5` |
| **Subtract** | `minus` | `-` | Subtraction | `10 - 2` | `8` |
| **Multiply** | `times` | | Multiplication | `4 times 2` | `8` |
| **Divide** | `div` | `/` | Division | `10 / 2` | `5` |
| **Power** | `pow` | `*` | Exponentiation | `2 * 3` | `8` (2³) |
| **Modulo** | `mod` | | Remainder | `10 mod 3` | `1` |
| **Range** | `range` | | Range Generation (0..N-1) | `range 5` | `[0,1,2,3,4]` |
| **Reshape** | `reshape` | | Array Reshape | `3 reshape 1` | `[1,1,1]` |
| **Ceiling** | `ceil` | | Round Up | `ceil 2.1` | `3` |
| **Floor** | `floor` | | Round Down | `floor 2.9` | `2` |
| **Logarithm** | `log` | | Log base B of A | `100 log 10` | `2` |
| **Trig** | `trig` | | Circular Functions (Pi based) | `trig 1` | `~3.14` |
| **Square Root**| `sqrt` | | Square Root | `sqrt 16` | `4` |

### Comparison

| Operator | Symbol | Description |
| :--- | :--- | :--- |
| **Equal** | `=` | True if values are equal |
| **Not Equal**| `!=` | True if values differ |
| **Less** | `<` | True if LHS < RHS |
| **Greater** | `>` | True if LHS > RHS |

---

## 🔄 Control Flow

Cell supports robust control flow constructs, allowing you to build complex logic instead of just linear formulas.

### Blocks (`do ... end`)
Group multiple statements together.
```ruby
do
  x = 10
  y = 20
end
```

### Conditionals (`if ... else`)
Standard conditional logic.
```ruby
res = if x > 10 do
  100
else
  0
end
```

### Loops (`while`, `for`)

**While Loop:**
```ruby
i = 0
while i < 10 do
  put(i)
  i = i + 1
end
```

**For Loop:**
Iterate over a range or array.
```ruby
# Prints 0 to 9
for i in range 10 do
  put(i)
end
```

---

## 📦 Functions

You can define reusable functions within a cell.

```ruby
# Define a function
fn add_nums(a, b) do
  return a + b
end

# Call it
result = add_nums(10, 20)
```

Values can also be returned via `yield` (for generator-like behavior in future versions) or `return`.

---

## 🛠 Built-in Functions

Cell provides a standard library of intrinsic functions available in the execution environment.

| Function | Signature | Description |
| :--- | :--- | :--- |
| **`put`** | `put(val1, val2, ...)` | Prints the values to the host standard output (debug). |
| **`input`** | `input()` | Requests a numeric input from the host environment. |
| **`rand`** | `rand(min, max)` | Generates a random number between `min` and `max`. |
| **`sum`** | `sum(Range)` | Calculates the sum of a cell range (e.g. `sum(A1:A5)`). |

---

## 📎 Cell References

You can reference other cells in the spreadsheet using standard syntax.

*   **`A1`**: Reference value of cell at Column A, Row 1.
*   **`A1:B5`**: Reference a range of cells (used in aggregation functions like `sum`).
*   **Relative/Absolute**: `A1` (Relative), `$A$1` (Absolute) - *Support depends on compilation context.*

---

## 🚀 Usage

### Prerequisites
*   **Rust** (latest stable)
*   **Cargo**

### Compiling a Spreadsheet

1.  Prepare your `.ods` file.
2.  Run the compiler:

```bash
cargo run --release -- --input spreadsheet.ods --output module.wasm
```

### Running the WASM
The generated WASM file exports a `_start` function (and potentially others). You can run it with any WASM runtime.

**Example with `wasmtime`:**
```bash
wasmtime module.wasm --invoke _start
```

---

## 🏗 Architecture

1.  **Loader**: Reads ODS XML structure.
2.  **Parser**: Uses `nom` to parse the cell text into an Abstract Syntax Tree (AST).
3.  **Codegen**: Traverses the AST and emits raw WebAssembly binary opcodes using `wasm-encoder`.
4.  **Output**: A clean, dependency-free `.wasm` file.

## 🤝 Contributing

Contributions are welcome! Please submit a Pull Request or open an Issue to discuss new operators or features.

License: MIT
