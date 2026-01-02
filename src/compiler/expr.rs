use anyhow::Result;
use wasm_encoder::{Function, Instruction, ValType, BlockType, MemArg};
use crate::parser::{Expr, Literal, Op};
use super::types::CompilerContext;

pub fn compile_expr(
    ctx: &CompilerContext,
    expr: &Expr,
    instrs: &mut Function,
) -> Result<()> {
    match expr {
        Expr::Literal(lit) => {
            match lit {
                Literal::Int(v) => { instrs.instruction(&Instruction::F64Const(*v as f64)); },
                Literal::Float(v) => { instrs.instruction(&Instruction::F64Const(*v)); },
                Literal::Bool(b) => { instrs.instruction(&Instruction::F64Const(if *b { 1.0 } else { 0.0 })); },
                Literal::Nil => { instrs.instruction(&Instruction::F64Const(0.0)); },
                Literal::String(_) => { instrs.instruction(&Instruction::F64Const(0.0)); },
            }
        },
        Expr::RelativeRef(dir) => {
            // Re-implement resolve_relative logic here or move utility to ctx? 
            // For now, let's implement simple resolution.
            let (dx, dy) = match dir {
                 crate::parser::Direction::Left => (-1, 0),
                 crate::parser::Direction::Right => (1, 0),
                 crate::parser::Direction::Top => (0, -1),
                 crate::parser::Direction::Bottom => (0, 1),
            };
            
            let target_col = ctx.col as i32 + dx;
            let target_row = ctx.row as i32 + dy;
            
            if target_col >= 0 && target_col <= ctx.grid.max_col as i32 && 
               target_row >= 0 && target_row <= ctx.grid.max_row as i32 {
                   let offset = (target_row as u32 * ctx.grid.max_col + target_col as u32) * 8;
                   instrs.instruction(&Instruction::I32Const(0));
                   instrs.instruction(&Instruction::F64Load(MemArg { offset: offset as u64, align: 3, memory_index: 0 }));
            } else {
                instrs.instruction(&Instruction::F64Const(0.0));
            }
        },
        Expr::If(cond, then_b, else_b) => {
             compile_expr(ctx, cond, instrs)?;
             instrs.instruction(&Instruction::F64Const(0.0));
             instrs.instruction(&Instruction::F64Ne); 
             instrs.instruction(&Instruction::If(BlockType::Result(ValType::F64)));
              // We need compile_block here. But circular dependency?
              // Let's assume compile_block is passed or we break recursion.
              // Actually, compile_block delegates to compile_stmt which delegates here.
              // We might need to move compile_block to a shared utility or pass a closure?
              // For simplicity, let's look at how to handle this.
             // instrs.instruction(&Instruction::Else);
             // ...
             // Solving circular dependency: compile_block logic usually iterates statements.
             // Let's implement block compilation inline OR make a `compile_block_expr` in this file.
             
             // Since we don't have cyclic imports easily, we will duplicate simple block logic or link it up higher.
             // BETTER: Have `stmt::compile_block` which is public, and use it here?
             // Rust allows circular module deps usually if structured right.
             // But for now, let's implement the block logic locally or call back.
             // Let's use `super::stmt::compile_block` if possible.
             
             super::stmt::compile_block_body(ctx, then_b, instrs, true)?;
             
             instrs.instruction(&Instruction::Else);
             if let Some(eb) = else_b {
                   super::stmt::compile_block_body(ctx, eb, instrs, true)?;
             } else {
                 instrs.instruction(&Instruction::F64Const(0.0));
             }
             instrs.instruction(&Instruction::End);
        },
        Expr::Generator(col) => {
            if let Some(idx) = ctx.cell_func_map.get(&(*col, 0)) {
                instrs.instruction(&Instruction::I32Const(*idx as i32));
                instrs.instruction(&Instruction::CallIndirect { ty: 6, table: 0 }); 
                instrs.instruction(&Instruction::Drop); 
                instrs.instruction(&Instruction::GlobalGet(ctx.reg_val_idx)); 
            } else {
                instrs.instruction(&Instruction::F64Const(0.0));
            }
        },
        Expr::Input => { instrs.instruction(&Instruction::Call(3)); },
        Expr::Call(name, args) => {
            if name == "put" {
               for (i, arg) in args.iter().enumerate() {
                   if let Expr::Literal(Literal::String(s)) = arg {
                        if let Some(&(offset, len)) = ctx.string_literals.get(s) {
                            instrs.instruction(&Instruction::I32Const(offset as i32));
                            instrs.instruction(&Instruction::I32Const(len as i32));
                            instrs.instruction(&Instruction::Call(5)); 
                            instrs.instruction(&Instruction::F64Const(0.0));
                        } else {
                            instrs.instruction(&Instruction::F64Const(0.0));
                            instrs.instruction(&Instruction::Call(0)); 
                        }
                   } else {
                        compile_expr(ctx, arg, instrs)?;
                       instrs.instruction(&Instruction::Call(0)); 
                   }
                   if i < args.len() - 1 { instrs.instruction(&Instruction::Drop); }
               }
            } else if ["rand", "min", "max", "hypot", "q_hypot", "mod", "pow"].contains(&name.as_str()) {
                 compile_expr_binary_builtin(ctx, name, &args[0], &args[1], instrs)?;
            } else if ["floor", "ceil", "round", "q_rsqrt"].contains(&name.as_str()) {
                 compile_expr_unary_builtin(ctx, name, &args[0], instrs)?;
            } else if ["sin", "cos", "tan", "asin", "acos", "atan"].contains(&name.as_str()) {
                compile_expr(ctx, &args[0], instrs)?;
                let import_idx = match name.as_str() {
                    "sin" => 6, "cos" => 7, "tan" => 8,
                    "asin" => 9, "acos" => 10, "atan" => 11,
                    _ => 0
                };
                instrs.instruction(&Instruction::Call(import_idx));
            } else if let Some(&func_idx) = ctx.func_name_map.get(name) {
                // Call Value
                let arity = *ctx.func_arity_map.get(name).unwrap_or(&0) as usize;
                // Validation skipped for now
                let type_idx = *ctx.arity_to_type_idx.get(&arity).unwrap();
                
                instrs.instruction(&Instruction::I32Const(func_idx as i32));
                instrs.instruction(&Instruction::CallIndirect { ty: type_idx, table: 0 });
                instrs.instruction(&Instruction::Drop);
                instrs.instruction(&Instruction::GlobalGet(ctx.reg_val_idx));
            } else {
                instrs.instruction(&Instruction::F64Const(0.0));
            }
        },
        Expr::Reference(r) => {
            let offset = (r.row * ctx.grid.max_col + r.col) * 8; // Simplified offset calc
            instrs.instruction(&Instruction::I32Const(0));
            instrs.instruction(&Instruction::F64Load(MemArg { offset: offset as u64, align: 3, memory_index: 0 }));
        },
        Expr::RangeReference(start, _end) => {
            if let Some(loop_idx) = ctx.loop_var_idx {
                let start_offset = (start.row * ctx.grid.max_col + start.col) * 8;
                instrs.instruction(&Instruction::I32Const(start_offset as i32));
                instrs.instruction(&Instruction::LocalGet(loop_idx));
                instrs.instruction(&Instruction::I32Const(8));
                instrs.instruction(&Instruction::I32Mul);
                instrs.instruction(&Instruction::I32Add);
                instrs.instruction(&Instruction::F64Load(MemArg { offset: 0, align: 3, memory_index: 0 }));
            } else {
                let offset = (start.row * ctx.grid.max_col + start.col) * 8;
                instrs.instruction(&Instruction::F64Const(offset as f64));
            }
        },
        Expr::BinaryOp(lhs, op, rhs) => {
            compile_expr(ctx, lhs, instrs)?;
            compile_expr(ctx, rhs, instrs)?;
            match op {
                Op::Add => { instrs.instruction(&Instruction::F64Add); },
                Op::Sub => { instrs.instruction(&Instruction::F64Sub); },
                Op::Mul => { instrs.instruction(&Instruction::F64Mul); },
                Op::Div => { instrs.instruction(&Instruction::F64Div); },
                 Op::Power => {
                    instrs.instruction(&Instruction::Call(1)); // pow import
                },
                Op::Gt => { 
                    instrs.instruction(&Instruction::F64Gt); 
                    instrs.instruction(&Instruction::F64ConvertI32S); 
                },
                Op::Lt => { 
                     instrs.instruction(&Instruction::F64Lt);
                     instrs.instruction(&Instruction::F64ConvertI32S);
                },
                Op::Eq => { 
                     instrs.instruction(&Instruction::F64Eq);
                     instrs.instruction(&Instruction::F64ConvertI32S);
                },
                Op::Neq => { 
                     instrs.instruction(&Instruction::F64Ne);
                     instrs.instruction(&Instruction::F64ConvertI32S);
                },
                 Op::And => { instrs.instruction(&Instruction::F64Mul); },
                 Op::Or => { instrs.instruction(&Instruction::F64Max); },
                 Op::Mod => {
                      instrs.instruction(&Instruction::Call(13)); 
                 },
                 _ => {
                    instrs.instruction(&Instruction::Drop);
                    instrs.instruction(&Instruction::Drop);
                    instrs.instruction(&Instruction::F64Const(0.0));
                }
            }
        },
        Expr::UnaryOp(op, expr) => {
            compile_expr(ctx, expr, instrs)?;
            match op {
                crate::parser::UnaryOp::Negate => { instrs.instruction(&Instruction::F64Neg); },
                crate::parser::UnaryOp::Not => { 
                    instrs.instruction(&Instruction::F64Const(0.0));
                    instrs.instruction(&Instruction::F64Eq); 
                    instrs.instruction(&Instruction::F64ConvertI32S);
                },
                _ => {}
            }
        },
        Expr::Ident(name) => {
            if let Some(idx) = ctx.locals.get(name) { instrs.instruction(&Instruction::LocalGet(*idx)); }
            else if let Some(idx) = ctx.globals.get(name) { instrs.instruction(&Instruction::GlobalGet(*idx)); }
            else { instrs.instruction(&Instruction::F64Const(0.0)); }
        },
        _ => { instrs.instruction(&Instruction::F64Const(0.0)); },
    }
    Ok(())
}

fn compile_expr_binary_builtin(ctx: &CompilerContext, name: &str, arg1: &Expr, arg2: &Expr, instrs: &mut Function) -> Result<()> {
    compile_expr(ctx, arg1, instrs)?;
    compile_expr(ctx, arg2, instrs)?;
    match name {
        "rand" => { instrs.instruction(&Instruction::Call(4)); },
        "min" => { instrs.instruction(&Instruction::F64Min); },
        "max" => { instrs.instruction(&Instruction::F64Max); },
        "hypot" => { instrs.instruction(&Instruction::Call(12)); },
        "q_hypot" => {
             // 14 imports + N cells. Start instrinsics at 14+N.
             // Init: 0, Main: 1, InvSqrt: 2, FastHypot: 3
             let fast_hypot_idx = 14 + (ctx.cell_func_map.len() as u32) + 3;
             instrs.instruction(&Instruction::Call(fast_hypot_idx));
        },
        "mod" => {
             // mod is import 13
             instrs.instruction(&Instruction::Call(13)); 
        },
        "pow" => { instrs.instruction(&Instruction::Call(1)); } // pow import is 1? Yes check imports
        _ => {}
    }
    Ok(())
}

fn compile_expr_unary_builtin(ctx: &CompilerContext, name: &str, arg: &Expr, instrs: &mut Function) -> Result<()> {
    compile_expr(ctx, arg, instrs)?;
    match name {
        "floor" => { instrs.instruction(&Instruction::F64Floor); },
        "ceil" => { instrs.instruction(&Instruction::F64Ceil); },
        "round" => { instrs.instruction(&Instruction::F64Nearest); },
        "q_rsqrt" => {
             // InvSqrt is intrinsic 2
             let inv_sqrt_idx = 14 + (ctx.cell_func_map.len() as u32) + 2;
             instrs.instruction(&Instruction::Call(inv_sqrt_idx));
        },
        _ => {}
    }
    Ok(())
}
