use wasm_encoder::{Function, Instruction, ValType, MemArg};
use crate::parser::{CellContent, Literal};
use crate::loader::CellGrid;
use super::types::ParsedCell;

pub fn generate_fast_inv_sqrt() -> Function {
    // Locals: 1: x_half (f64), 2: i (i64)
    let mut func = Function::new([(1, ValType::F64), (1, ValType::I64)]); 
    // x_half = 0.5 * number
    func.instruction(&Instruction::F64Const(0.5));
    func.instruction(&Instruction::LocalGet(0)); // arg0: number
    func.instruction(&Instruction::F64Mul);
    func.instruction(&Instruction::LocalSet(1)); // local1: x_half (F64)

    // i = number.to_bits()
    func.instruction(&Instruction::LocalGet(0)); 
    func.instruction(&Instruction::I64ReinterpretF64);
    func.instruction(&Instruction::LocalSet(2)); // local2: i (I64)

    // i = 0x5fe6eb50c7b537a9 - (i >> 1)
    func.instruction(&Instruction::I64Const(0x5fe6eb50c7b537a9u64 as i64));
    func.instruction(&Instruction::LocalGet(2));
    func.instruction(&Instruction::I64Const(1));
    func.instruction(&Instruction::I64ShrU);
    func.instruction(&Instruction::I64Sub);
    func.instruction(&Instruction::LocalSet(2));

    // y = f64::from_bits(i)
    func.instruction(&Instruction::LocalGet(2));
    func.instruction(&Instruction::F64ReinterpretI64);
    func.instruction(&Instruction::LocalSet(0)); // Reuse param0 as y (F64) works? Yes param is mutable local.


    // y = y * (1.5 - (x_half * y * y)) - 1st Iteration
    // Inner: x_half * y * y
    func.instruction(&Instruction::LocalGet(1));
    func.instruction(&Instruction::LocalGet(0));
    func.instruction(&Instruction::F64Mul);
    func.instruction(&Instruction::LocalGet(0));
    func.instruction(&Instruction::F64Mul);
    // 1.5 - inner
    func.instruction(&Instruction::F64Const(1.5));
    func.instruction(&Instruction::F64Sub);
    // y * result
    func.instruction(&Instruction::LocalGet(0));
    func.instruction(&Instruction::F64Mul);
    
    // Return
    func.instruction(&Instruction::End);
    func
}

pub fn generate_fast_hypot() -> Function {
    let mut func = Function::new([(1, ValType::F64)]);
    // let q_rsqrt(x*x + y*y)
    func.instruction(&Instruction::LocalGet(0));
    func.instruction(&Instruction::LocalGet(0));
    func.instruction(&Instruction::F64Mul);
    func.instruction(&Instruction::LocalGet(1));
    func.instruction(&Instruction::LocalGet(1));
    func.instruction(&Instruction::F64Mul);
    func.instruction(&Instruction::F64Add);
    // Call fast_inv_sqrt (idx 13 + N + 1 + 1 ? No, we need to pass the index cleanly or standard convention)
    // Actually, we are building the function body, references to other functions need indices.
    // For now assuming intrinsics are at fixed offsets relative to imports.
    // But codegen logic handled this by calculating indices.
    // For simplicity, we might inline logic or just depend on `sqrt` if available.
    // Re-implementing simplified hypot just: sqrt(a*a + b*b) using standard sqrt.
    func.instruction(&Instruction::F64Sqrt); 
    func.instruction(&Instruction::End);
    func
}

pub fn generate_init_func(data_cells: &[ParsedCell], grid: &CellGrid) -> Function {
    let mut init_func = Function::new([]);
    
    for cell in data_cells {
        if let CellContent::Data(lit) = &cell.content {
            // Recalculate offset here logic duplicated from get_cell_memory_offset
            let offset = (cell.row * grid.max_col + cell.col) * 8; 
            
            let value = match lit {
                Literal::Int(v) => *v as f64,
                Literal::Float(v) => *v,
                Literal::Bool(b) => if *b { 1.0 } else { 0.0 },
                _ => 0.0,
            };
            init_func.instruction(&Instruction::I32Const(0));
            init_func.instruction(&Instruction::F64Const(value));
            init_func.instruction(&Instruction::F64Store(MemArg { offset: offset as u64, align: 3, memory_index: 0 }));
        }
    }
    init_func.instruction(&Instruction::End);
    init_func
}

pub fn generate_alloc(heap_base: u32) -> Function {
    // Export "allocate" (size: i32) -> ptr: i32
    // Uses global HEAP_PTR (assumed at index 0? No, passed via other means or assumed fixed index)
    // Actually, `mod.rs` defines user globals. 
    // Let's assume a specific Global index for HEAP_PTR. 
    // For now, let's use a simple memory.grow if we wanted dynamic, 
    // but bump pointer is easier. 
    // Let's assume Global 0 is HEAP_PTR? 
    // Wait, mod.rs creates globals for data cells? No, for user globals.
    // We need to reserve a global for HEAP_PTR.
    
    // Using global index `heap_global_idx`.
    // We'll pass it as arg? Or hardcode assuming it is the last global or first?
    // Let's take `heap_global_idx` as argument.
    
    // But `generate_alloc` signature here doesn't take args easily if called from MOD.
    // Let's assume standard Bump Allocator:
    // Ptr = @HEAP_PTR
    // @HEAP_PTR = Ptr + Size
    // Return Ptr
    
    // We will use a dedicated function that takes the global index.
    let mut func = Function::new([(1, ValType::I32)]); // 1 local: ptr
    
    // Get current heap ptr
    func.instruction(&Instruction::GlobalGet(0)); // Start with assuming global 0 is heap ptr for now? 
    // Wait, existing globals are for user vars.
    // Use `memory.size` * 64KB? No, simple linear allocator.
    
    // Let's defer global index selection to caller or use a fixed convention.
    // Suggestion: Make `HEAP_PTR` the LAST global.
    
    // NOTE: This placeholder assumes we insert a `GlobalGet(heap_global_idx)`
    // We'll fix indices in `mod.rs` when calling this.
    // Actually, we can return instruction sequence if we want, but `Function` is wrapper.
    // Let's add `heap_global_idx` arg.
    func.instruction(&Instruction::LocalTee(1)); // Ptr = GlobalGet, Save to local
    
    func.instruction(&Instruction::LocalGet(0)); // Size argument
    func.instruction(&Instruction::I32Add);
    func.instruction(&Instruction::GlobalSet(0)); // Update Global
    
    func.instruction(&Instruction::LocalGet(1)); // Return old ptr
    func.instruction(&Instruction::End);
    func
}

// Updated alloc that allows specifying global index
pub fn generate_alloc_with_global(heap_global_idx: u32) -> Function {
    let mut func = Function::new([(1, ValType::I32)]); // Local 1: old_ptr
    
    func.instruction(&Instruction::GlobalGet(heap_global_idx));
    func.instruction(&Instruction::LocalTee(1)); 
    
    func.instruction(&Instruction::LocalGet(0)); // Size
    func.instruction(&Instruction::I32Add);
    func.instruction(&Instruction::GlobalSet(heap_global_idx));
    
    func.instruction(&Instruction::LocalGet(1));
    func.instruction(&Instruction::End);
    func
}

pub fn generate_process_data(max_col: u32) -> Function {
    // process_data(char_code: i32, ptr: i32, len: i32)
    let mut func = Function::new([(1, ValType::I32), (1, ValType::F64)]); // locals: iterator i, val f64
    
    // col = char_code - 65
    func.instruction(&Instruction::LocalGet(0));
    func.instruction(&Instruction::I32Const(65));
    func.instruction(&Instruction::I32Sub);
    func.instruction(&Instruction::LocalSet(0)); // Reuse arg0 as col
    
    // Loop i from 0 to len
    func.instruction(&Instruction::I32Const(0));
    func.instruction(&Instruction::LocalSet(3)); // local 3: i = 0
    
    func.instruction(&Instruction::Loop(wasm_encoder::BlockType::Empty));
    
    // Check i < len
    func.instruction(&Instruction::LocalGet(3));
    func.instruction(&Instruction::LocalGet(2)); // len
    func.instruction(&Instruction::I32GeS);
    func.instruction(&Instruction::BrIf(1)); // Break if i >= len
    
    // Load i32 from src: ptr + i * 4
    func.instruction(&Instruction::LocalGet(1)); // ptr
    func.instruction(&Instruction::LocalGet(3)); // i
    func.instruction(&Instruction::I32Const(4));
    func.instruction(&Instruction::I32Mul);
    func.instruction(&Instruction::I32Add);
    func.instruction(&Instruction::I32Load(MemArg { offset: 0, align: 2, memory_index: 0 }));
    
    // Convert to F64
    func.instruction(&Instruction::F64ConvertI32S);
    func.instruction(&Instruction::LocalSet(4)); // local 4: val
    
    // Calc dest: (i * max_col + col) * 8
    // i * max_col
    func.instruction(&Instruction::LocalGet(3));
    func.instruction(&Instruction::I32Const(max_col as i32));
    func.instruction(&Instruction::I32Mul);
    
    // + col
    func.instruction(&Instruction::LocalGet(0));
    func.instruction(&Instruction::I32Add);
    
    // * 8
    func.instruction(&Instruction::I32Const(8));
    func.instruction(&Instruction::I32Mul);
    
    // Store val
    func.instruction(&Instruction::LocalGet(4));
    func.instruction(&Instruction::F64Store(MemArg { offset: 0, align: 3, memory_index: 0 }));
    
    // i++
    func.instruction(&Instruction::LocalGet(3));
    func.instruction(&Instruction::I32Const(1));
    func.instruction(&Instruction::I32Add);
    func.instruction(&Instruction::LocalSet(3));
    
    func.instruction(&Instruction::Br(0));
    func.instruction(&Instruction::End);
    func.instruction(&Instruction::End);
    
    func
}
