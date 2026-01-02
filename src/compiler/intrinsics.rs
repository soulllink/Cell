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
