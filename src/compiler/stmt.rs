use anyhow::Result;
use std::collections::HashMap;
use wasm_encoder::{Function, Instruction, ValType};
use crate::parser::{Block, Stmt, FunctionDef, Expr};
use super::types::CompilerContext;
use super::expr::compile_expr;

pub fn compile_cell(
    col: u32,
    row: u32,
    func_def: &FunctionDef,
    globals: &HashMap<String, u32>,
    cell_func_map: &HashMap<(u32, u32), u32>,
    func_name_map: &HashMap<String, u32>,
    func_arity_map: &HashMap<String, u32>,
    arity_to_type_idx: &HashMap<usize, u32>,
    my_table_idx: u32,
    reg_val_idx: u32,
    string_literals: &HashMap<String, (u32, u32)>,
    grid: &crate::loader::CellGrid,
) -> Result<Function> {
    let stmts = &func_def.body.stmts;
    let mut locals_map = HashMap::new();
    let mut locals_types = Vec::new();
    let mut next_local_idx = 0;

    // 1. Map Arguments first
    for arg in &func_def.args {
        locals_map.insert(arg.clone(), next_local_idx);
        locals_types.push(ValType::F64); // All user args are F64 for now
        next_local_idx += 1;
    }

    // 2. Scan for locals
    scan_locals(stmts, &mut locals_map, &mut locals_types, &mut next_local_idx);
    
    // Sum iterators for generator stubs (keeping logical consistency with original)
    // let sum_i_idx = next_local_idx; locals_types.push(ValType::I32); next_local_idx += 1;
    // let sum_len_idx = next_local_idx; locals_types.push(ValType::I32); next_local_idx += 1;
    // let sum_acc_idx = next_local_idx; locals_types.push(ValType::F64); 

    let mut func = Function::new(locals_types.iter().map(|n| (1, *n)));
    
    // Context creation
    let ctx = CompilerContext {
        col, row,
        locals: &locals_map,
        globals,
        cell_func_map,
        func_name_map,
        func_arity_map,
        arity_to_type_idx,
        reg_val_idx,
        loop_var_idx: None, // Will set in loops
        my_table_idx,
        string_literals,
        grid
    };

    compile_block_body(&ctx, &func_def.body, &mut func, false)?;
    
    // Default return logic
    // if function body didn't explicitly return, returns index of self in table
    func.instruction(&Instruction::I32Const((my_table_idx + 1) as i32));
    func.instruction(&Instruction::End);

    Ok(func)
}

fn scan_locals(stmts: &[Stmt], map: &mut HashMap<String, u32>, types: &mut Vec<ValType>, next_idx: &mut u32) {
    for stmt in stmts {
        match stmt {
            Stmt::Assign(name, _) => {
                if !map.contains_key(name) {
                    map.insert(name.clone(), *next_idx);
                    types.push(ValType::F64);
                    *next_idx += 1;
                }
            },
            Stmt::While(_, body) => scan_locals(&body.stmts, map, types, next_idx),
            Stmt::For(var, _, body) => {
                if !map.contains_key(var) {
                     map.insert(var.clone(), *next_idx);
                     types.push(ValType::F64); // loop var
                     *next_idx += 1;
                }
                scan_locals(&body.stmts, map, types, next_idx)
            },
            _ => {}
        }
    }
}

pub fn compile_block_body(ctx: &CompilerContext, block: &Block, instrs: &mut Function, needs_result: bool) -> Result<()> {
    let len = block.stmts.len();
    if len == 0 && needs_result { instrs.instruction(&Instruction::F64Const(0.0)); return Ok(()); }
    
    for (i, stmt) in block.stmts.iter().enumerate() {
        let is_last = i == len - 1;
        match stmt {
            Stmt::Assign(name, expr) => {
                compile_expr(ctx, expr, instrs)?;
                if let Some(idx) = ctx.locals.get(name) { instrs.instruction(&Instruction::LocalSet(*idx)); }
                else if let Some(idx) = ctx.globals.get(name) { instrs.instruction(&Instruction::GlobalSet(*idx)); }
                if is_last && needs_result { instrs.instruction(&Instruction::F64Const(0.0)); }
            },
            Stmt::Return(expr) => {
                 if let Expr::Reference(r) = expr {
                     let target_idx = ctx.cell_func_map.get(&(r.col, r.row));
                     if let Some(target_idx) = target_idx {
                         instrs.instruction(&Instruction::I32Const(*target_idx as i32));
                     } else {
                         instrs.instruction(&Instruction::I32Const((ctx.my_table_idx + 1) as i32));
                     }
                 } else {
                     compile_expr(ctx, expr, instrs)?;
                     instrs.instruction(&Instruction::Drop);
                     instrs.instruction(&Instruction::I32Const((ctx.my_table_idx + 1) as i32)); 
                 }
                 instrs.instruction(&Instruction::Return);
            },
            Stmt::Yield(expr) => {
                 compile_expr(ctx, expr, instrs)?;
                 instrs.instruction(&Instruction::GlobalSet(ctx.reg_val_idx));
                 instrs.instruction(&Instruction::Return);
            },
            Stmt::Expr(expr) => {
                compile_expr(ctx, expr, instrs)?;
                if !is_last || !needs_result { instrs.instruction(&Instruction::Drop); }
            },
            Stmt::While(cond, body) => {
                 // While Loop Logic
                 instrs.instruction(&Instruction::Block(wasm_encoder::BlockType::Empty));
                 instrs.instruction(&Instruction::Loop(wasm_encoder::BlockType::Empty));
                 
                 compile_expr(ctx, cond, instrs)?;
                 instrs.instruction(&Instruction::F64Const(0.0));
                 instrs.instruction(&Instruction::F64Eq);
                 instrs.instruction(&Instruction::BrIf(1)); // Break if cond == 0
                 
                 compile_block_body(ctx, body, instrs, false)?;
                 
                 instrs.instruction(&Instruction::Br(0));
                 instrs.instruction(&Instruction::End);
                 instrs.instruction(&Instruction::End);
                 
                 if is_last && needs_result { instrs.instruction(&Instruction::F64Const(0.0)); }
            },
             _ => { if is_last && needs_result { instrs.instruction(&Instruction::F64Const(0.0)); } }
        }
    }
    Ok(())
}
