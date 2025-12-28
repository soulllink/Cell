use anyhow::{Result, anyhow};
use wasm_encoder::{
    CodeSection, ExportKind, ExportSection, Function, FunctionSection, Instruction, MemArg, MemoryType,
    MemorySection, Module, TypeSection, ValType, ImportSection, EntityType, BlockType,
};
use crate::loader::CellGrid;
use crate::parser::{Expr, Stmt, Op, Literal, CellRef};
use std::collections::HashMap;

pub struct WasmCompiler<'a> {
    grid: &'a CellGrid,
    cell_offsets: HashMap<(u32, u32), u32>,
    next_func_index: u32,
}

impl<'a> WasmCompiler<'a> {
    pub fn new(grid: &'a CellGrid) -> Self {
        Self {
            grid,
            cell_offsets: HashMap::new(),
            next_func_index: 0,
        }
    }

    fn get_cell_memory_offset(&self, col: u32, row: u32) -> u32 {
        // Simple linear layout: (Col * MaxRows + Row) * 8 bytes (i64)
        // Assume MaxRows = 1000 for safety if not strict, or use grid.max_row
        // Using grid.max_row which is dynamic.
        // Safety padding?
        let max_rows = self.grid.max_row.max(100); 
        (col * max_rows + row) * 8
    }

    fn scan_locals(stmts: &[Stmt], locals_map: &mut HashMap<String, u32>, locals_types: &mut Vec<ValType>, next_idx: &mut u32) {
        for stmt in stmts {
            match stmt {
                Stmt::Assign(name, _) => {
                    if !locals_map.contains_key(name) {
                        locals_map.insert(name.clone(), *next_idx);
                        locals_types.push(ValType::F64);
                        *next_idx += 1;
                    }
                },
                Stmt::For(name, _, body) => {
                     // Check loop var
                     if !locals_map.contains_key(name) {
                        locals_map.insert(name.clone(), *next_idx);
                        locals_types.push(ValType::F64); 
                        *next_idx += 1;
                    }
                    Self::scan_locals(&body.stmts, locals_map, locals_types, next_idx);
                },
                Stmt::While(_, body) => {
                    Self::scan_locals(&body.stmts, locals_map, locals_types, next_idx);
                },
                Stmt::Expr(expr) => {
                     Self::scan_expr_locals(expr, locals_map, locals_types, next_idx);
                },
                 _ => {}
            }
        }
    }

    fn scan_expr_locals(expr: &Expr, locals_map: &mut HashMap<String, u32>, locals_types: &mut Vec<ValType>, next_idx: &mut u32) {
        match expr {
            Expr::If(_, then_b, else_b) => {
                 Self::scan_locals(&then_b.stmts, locals_map, locals_types, next_idx);
                 if let Some(eb) = else_b {
                     Self::scan_locals(&eb.stmts, locals_map, locals_types, next_idx);
                 }
            },
            Expr::Block(b) => {
                 Self::scan_locals(&b.stmts, locals_map, locals_types, next_idx);
            },
            _ => {}
        }
    }

    fn scan_globals(&mut self, execution_order: &Vec<((u32, u32), &String)>) -> Result<HashMap<String, u32>> {
        let mut globals = HashMap::new();
        let mut next_global_idx = 0;
        
        for (_, content) in execution_order {
             let stmts = crate::parser::parse_cell_content(content)?;
             self.scan_stmts_for_globals(&stmts, &mut globals, &mut next_global_idx);
        }
        Ok(globals)
    }
    
    fn scan_stmts_for_globals(&self, stmts: &[Stmt], globals: &mut HashMap<String, u32>, next_idx: &mut u32) {
        for stmt in stmts {
            match stmt {
                Stmt::Assign(name, _) => {
                    if !globals.contains_key(name) {
                        globals.insert(name.clone(), *next_idx);
                        *next_idx += 1;
                    }
                },
                Stmt::While(_, body) => self.scan_stmts_for_globals(&body.stmts, globals, next_idx),
                Stmt::For(_, _, body) => self.scan_stmts_for_globals(&body.stmts, globals, next_idx),
                Stmt::Expr(expr) => self.scan_expr_for_blocks(expr, globals, next_idx),
                _ => {}
            }
        }
    }
    
    fn scan_expr_for_blocks(&self, expr: &Expr, globals: &mut HashMap<String, u32>, next_idx: &mut u32) {
         match expr {
            Expr::If(_, then_b, else_b) => {
                 self.scan_stmts_for_globals(&then_b.stmts, globals, next_idx);
                 if let Some(eb) = else_b {
                     self.scan_stmts_for_globals(&eb.stmts, globals, next_idx);
                 }
            },
            Expr::Block(b) => {
                 self.scan_stmts_for_globals(&b.stmts, globals, next_idx);
            },
            _ => {}
         }
    }

    fn compile_cell(&self, col: u32, row: u32, stmts: &[Stmt], globals: &HashMap<String, u32>) -> Result<Function> {
        // Collect locals including those nested in blocks
        let mut locals_map = HashMap::new();
        let mut locals_types = Vec::new();
        let mut next_local_idx = 0;

        Self::scan_locals(stmts, &mut locals_map, &mut locals_types, &mut next_local_idx);
        
        // Reserve extra local for loop index (I32) if needed?
        // Let's just append it blindly for simplicity or use a fixed index if logic allows.
        // Or manage it dynamically.
        // We need 1 I32 local for `yield` loop counter.
        // We need 1 I32 local for `yield` loop counter.
        let loop_counter_idx = next_local_idx;
        locals_types.push(ValType::I32);
        next_local_idx += 1;
        
        // Also one for Loop Limit (I32)
        let loop_limit_idx = next_local_idx;
        locals_types.push(ValType::I32);
        next_local_idx += 1;
        
        // Locals for 'sum' intrinsic
        let sum_i_idx = next_local_idx;
        locals_types.push(ValType::I32);
        next_local_idx += 1;
        
        let sum_len_idx = next_local_idx;
        locals_types.push(ValType::I32);
        next_local_idx += 1;
        
        // Sum Accumulator (F64)
        let sum_acc_idx = next_local_idx;
        locals_types.push(ValType::F64);
        // next_local_idx += 1;

        let mut func = Function::new(locals_types.iter().map(|n| (1, *n)));

        let sum_locals_tuple = (sum_i_idx, sum_len_idx, sum_acc_idx);

        for stmt in stmts {
             match stmt {
                Stmt::Assign(name, expr) => {
                    self.compile_expr(expr, &locals_map, globals, &mut func, None, sum_locals_tuple)?;
                    if let Some(idx) = locals_map.get(name) {
                        func.instruction(&Instruction::LocalSet(*idx));
                    } else if let Some(idx) = globals.get(name) {
                        func.instruction(&Instruction::GlobalSet(*idx));
                    } else {
                         return Err(anyhow!("Unknown variable for assignment: {}", name));
                    }
                },
                Stmt::Return(expr) => {
                    let offset = self.get_cell_memory_offset(col, row);
                    func.instruction(&Instruction::I32Const(offset as i32));
                    self.compile_expr(expr, &locals_map, globals, &mut func, None, sum_locals_tuple)?;
                    func.instruction(&Instruction::F64Store(MemArg { offset: 0, align: 3, memory_index: 0 }));
                },
                Stmt::Yield(expr) => {
                    let size = self.grid.get_region_size(col, row);
                    
                    func.instruction(&Instruction::I32Const(0));
                    func.instruction(&Instruction::LocalSet(loop_counter_idx));
                    
                    func.instruction(&Instruction::I32Const(size as i32));
                    func.instruction(&Instruction::LocalSet(loop_limit_idx));
                    
                    func.instruction(&Instruction::Loop(wasm_encoder::BlockType::Empty)); 
                    func.instruction(&Instruction::Block(wasm_encoder::BlockType::Empty)); 
                    
                    func.instruction(&Instruction::LocalGet(loop_counter_idx));
                    func.instruction(&Instruction::LocalGet(loop_limit_idx));
                    func.instruction(&Instruction::I32GeS); 
                    func.instruction(&Instruction::BrIf(0)); 
                    
                    let base_offset = self.get_cell_memory_offset(col, row);
                    
                    func.instruction(&Instruction::I32Const(base_offset as i32));
                    func.instruction(&Instruction::LocalGet(loop_counter_idx));
                    func.instruction(&Instruction::I32Const(8));
                    func.instruction(&Instruction::I32Mul);
                    func.instruction(&Instruction::I32Add); 
                    
                    self.compile_expr(expr, &locals_map, globals, &mut func, Some(loop_counter_idx), sum_locals_tuple)?;
                    
                    func.instruction(&Instruction::F64Store(MemArg { offset: 0, align: 3, memory_index: 0 }));
                    
                    func.instruction(&Instruction::LocalGet(loop_counter_idx));
                    func.instruction(&Instruction::I32Const(1));
                    func.instruction(&Instruction::I32Add);
                    func.instruction(&Instruction::LocalSet(loop_counter_idx));
                    
                    func.instruction(&Instruction::Br(1)); 
                    func.instruction(&Instruction::End); 
                    func.instruction(&Instruction::End); 
                },
                Stmt::Expr(expr) => {
                    self.compile_expr(expr, &locals_map, globals, &mut func, None, sum_locals_tuple)?;
                    func.instruction(&Instruction::Drop);
                },
                Stmt::While(cond, body) => {
                     func.instruction(&Instruction::Loop(wasm_encoder::BlockType::Empty)); 
                     func.instruction(&Instruction::Block(wasm_encoder::BlockType::Empty)); 

                     self.compile_expr(cond, &locals_map, globals, &mut func, None, sum_locals_tuple)?;
                     func.instruction(&Instruction::F64Const(0.0));
                     func.instruction(&Instruction::F64Eq); 
                     func.instruction(&Instruction::BrIf(0)); 
                    
                     self.compile_block(body, &locals_map, globals, &mut func, sum_locals_tuple, false)?;
                     
                     func.instruction(&Instruction::Br(1)); 
                     func.instruction(&Instruction::End); 
                     func.instruction(&Instruction::End); 
                },
                Stmt::For(_, _, _) => { },
                Stmt::FnDef(_, _, _) => {},
            }
        }
        
        func.instruction(&Instruction::End);
        Ok(func)
    }

    fn compile_block(&self, block: &crate::parser::Block, locals: &HashMap<String, u32>, globals: &HashMap<String, u32>, instrs: &mut Function, sum_locals: (u32, u32, u32), needs_result: bool) -> Result<()> {
        let len = block.stmts.len();
        if len == 0 {
            if needs_result {
                instrs.instruction(&Instruction::F64Const(0.0));
            }
            return Ok(());
        }

        for (i, stmt) in block.stmts.iter().enumerate() {
            let is_last = i == len - 1;
             match stmt {
                Stmt::Assign(name, expr) => {
                    self.compile_expr(expr, locals, globals, instrs, None, sum_locals)?;
                    if let Some(idx) = locals.get(name) {
                        instrs.instruction(&Instruction::LocalSet(*idx));
                    } else if let Some(idx) = globals.get(name) {
                        instrs.instruction(&Instruction::GlobalSet(*idx));
                    } else {
                         return Err(anyhow!("Unknown variable: {}", name));
                    }
                    if is_last && needs_result {
                         instrs.instruction(&Instruction::F64Const(0.0));
                    }
                },
                Stmt::Return(expr) => { 
                    self.compile_expr(expr, locals, globals, instrs, None, sum_locals)?;
                    instrs.instruction(&Instruction::Return);
                },
                Stmt::Yield(expr) => {
                    self.compile_expr(expr, locals, globals, instrs, None, sum_locals)?;
                    instrs.instruction(&Instruction::Return);
                },
                Stmt::Expr(expr) => {
                    self.compile_expr(expr, locals, globals, instrs, None, sum_locals)?;
                    if is_last && needs_result {
                        // Keep result on stack
                    } else {
                        instrs.instruction(&Instruction::Drop);
                        if is_last && needs_result {
                             // SNH: if we dropped, we need to push back? No, above branch handles it.
                             // Wait, logic: if NOT (is_last && needs_result) -> Drop.
                        }
                    }
                },
                Stmt::While(cond, body) => {
                    instrs.instruction(&Instruction::Loop(wasm_encoder::BlockType::Empty)); 
                    instrs.instruction(&Instruction::Block(wasm_encoder::BlockType::Empty)); 

                    self.compile_expr(cond, locals, globals, instrs, None, sum_locals)?;
                    instrs.instruction(&Instruction::F64Const(0.0));
                    instrs.instruction(&Instruction::F64Eq); 
                    instrs.instruction(&Instruction::BrIf(0)); 
                    
                    self.compile_block(body, locals, globals, instrs, sum_locals, false)?; 
                    
                    instrs.instruction(&Instruction::Br(1)); 
                    instrs.instruction(&Instruction::End); 
                    instrs.instruction(&Instruction::End); 
                    
                    if is_last && needs_result {
                        instrs.instruction(&Instruction::F64Const(0.0));
                    }
                },
                 _ => {
                     if is_last && needs_result {
                         instrs.instruction(&Instruction::F64Const(0.0));
                     }
                 }
             }
        }
        Ok(())
    }

    fn compile_expr(&self, expr: &Expr, locals: &HashMap<String, u32>, globals: &HashMap<String, u32>, instrs: &mut Function, loop_var_idx: Option<u32>, sum_locals: (u32, u32, u32)) -> Result<()> {
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
            Expr::Input => {
                instrs.instruction(&Instruction::Call(3)); // Call env.input
            },
            Expr::If(cond, then_block, else_block) => {
                  self.compile_expr(cond, locals, globals, instrs, loop_var_idx, sum_locals)?; 
                  
                  instrs.instruction(&Instruction::F64Const(0.0));
                  instrs.instruction(&Instruction::F64Ne); 
                  
                  instrs.instruction(&Instruction::If(wasm_encoder::BlockType::Result(ValType::F64))); 
                  
                  self.compile_block(then_block, locals, globals, instrs, sum_locals, true)?;
                  
                  instrs.instruction(&Instruction::Else);
                  if let Some(else_b) = else_block {
                      self.compile_block(else_b, locals, globals, instrs, sum_locals, true)?;
                  } else {
                      instrs.instruction(&Instruction::F64Const(0.0));
                  }
                  instrs.instruction(&Instruction::End);
            },
            Expr::Reference(r) => {
                let offset = self.get_cell_memory_offset(r.col, r.row);
                instrs.instruction(&Instruction::I32Const(0));
                instrs.instruction(&Instruction::F64Load(MemArg { offset: offset as u64, align: 3, memory_index: 0 }));
            },
            Expr::RangeReference(start, _end) => {
                if let Some(loop_idx) = loop_var_idx {
                    let start_offset = self.get_cell_memory_offset(start.col, start.row);
                    instrs.instruction(&Instruction::I32Const(start_offset as i32));
                    instrs.instruction(&Instruction::LocalGet(loop_idx));
                    instrs.instruction(&Instruction::I32Const(8)); 
                    instrs.instruction(&Instruction::I32Mul);
                    instrs.instruction(&Instruction::I32Add);
                    instrs.instruction(&Instruction::F64Load(MemArg { offset: 0, align: 3, memory_index: 0 }));
                } else {
                    let offset = self.get_cell_memory_offset(start.col, start.row);
                    instrs.instruction(&Instruction::F64Const(offset as f64));
                }
            },
            Expr::BinaryOp(lhs, op, rhs) => {
                match op {
                    Op::Log => {
                        self.compile_expr(rhs, locals, globals, instrs, loop_var_idx, sum_locals)?;
                        instrs.instruction(&Instruction::Call(2)); // log
                        self.compile_expr(lhs, locals, globals, instrs, loop_var_idx, sum_locals)?;
                        instrs.instruction(&Instruction::Call(2)); // log
                        instrs.instruction(&Instruction::F64Div);
                    },
                    Op::Stile => {
                         self.compile_expr(rhs, locals, globals, instrs, loop_var_idx, sum_locals)?;
                         self.compile_expr(lhs, locals, globals, instrs, loop_var_idx, sum_locals)?;
                         instrs.instruction(&Instruction::Drop);
                         instrs.instruction(&Instruction::Drop);
                         instrs.instruction(&Instruction::F64Const(0.0));
                    },
                    _ => {
                        self.compile_expr(lhs, locals, globals, instrs, loop_var_idx, sum_locals)?;
                        self.compile_expr(rhs, locals, globals, instrs, loop_var_idx, sum_locals)?;
                        match op {
                            Op::Add => { instrs.instruction(&Instruction::F64Add); },
                            Op::Sub => { instrs.instruction(&Instruction::F64Sub); },
                            Op::Mul => { instrs.instruction(&Instruction::F64Mul); },
                            Op::Div => { instrs.instruction(&Instruction::F64Div); },
                            Op::Power => { instrs.instruction(&Instruction::Call(1)); }, 
                            Op::Ceil => { instrs.instruction(&Instruction::F64Max); }, 
                            Op::Floor => { instrs.instruction(&Instruction::F64Min); }, 
                            Op::Rho | Op::Iota | Op::Circular | Op::Radix | 
                            Op::Eq | Op::Neq | Op::Lt | Op::Gt | Op::Stile | Op::Log => {
                                // TODO
                                instrs.instruction(&Instruction::Drop); // RHS
                                instrs.instruction(&Instruction::Drop); // LHS
                                instrs.instruction(&Instruction::F64Const(0.0));
                            },
                        }
                    }
                }
            },
            Expr::Ident(name) => {
                if let Some(idx) = locals.get(name) {
                    instrs.instruction(&Instruction::LocalGet(*idx));
                } else if let Some(idx) = globals.get(name) {
                    instrs.instruction(&Instruction::GlobalGet(*idx));
                } else {
                    eprintln!("Warning: Unknown variable '{}', treating as 0.0", name);
                    instrs.instruction(&Instruction::F64Const(0.0));
                }
            },
            Expr::Call(name, args) => {
                if name == "put" {
                   if args.is_empty() { return Err(anyhow!("put takes at least 1 arg")); }
                   for (i, arg) in args.iter().enumerate() {
                       self.compile_expr(arg, locals, globals, instrs, loop_var_idx, sum_locals)?;
                       instrs.instruction(&Instruction::Call(0)); // put
                       if i < args.len() - 1 {
                           instrs.instruction(&Instruction::Drop);
                       }
                   }
                } else if name == "sum" {
                     if args.len() != 1 { return Err(anyhow!("sum checks 1 range arg")); }
                     let (sum_i, sum_len, sum_acc) = sum_locals;
                     if let Expr::RangeReference(start, end) = &args[0] {
                         let len = end.row as i32 - start.row as i32 + 1;
                         let size = len.max(1);
                         
                         instrs.instruction(&Instruction::I32Const(0));
                         instrs.instruction(&Instruction::LocalSet(sum_i));
                         instrs.instruction(&Instruction::I32Const(size));
                         instrs.instruction(&Instruction::LocalSet(sum_len));
                         instrs.instruction(&Instruction::F64Const(0.0));
                         instrs.instruction(&Instruction::LocalSet(sum_acc));
                         
                         instrs.instruction(&Instruction::Loop(wasm_encoder::BlockType::Empty)); 
                         instrs.instruction(&Instruction::Block(wasm_encoder::BlockType::Empty)); 
                         
                         instrs.instruction(&Instruction::LocalGet(sum_i));
                         instrs.instruction(&Instruction::LocalGet(sum_len));
                         instrs.instruction(&Instruction::I32GeS);
                         instrs.instruction(&Instruction::BrIf(0));
                         
                         instrs.instruction(&Instruction::LocalGet(sum_acc));
                         let start_offset = self.get_cell_memory_offset(start.col, start.row);
                         instrs.instruction(&Instruction::I32Const(start_offset as i32));
                         instrs.instruction(&Instruction::LocalGet(sum_i));
                         instrs.instruction(&Instruction::I32Const(8));
                         instrs.instruction(&Instruction::I32Mul);
                         instrs.instruction(&Instruction::I32Add);
                         instrs.instruction(&Instruction::F64Load(MemArg { offset: 0, align: 3, memory_index: 0 }));
                         instrs.instruction(&Instruction::F64Add);
                         instrs.instruction(&Instruction::LocalSet(sum_acc));
                         
                         instrs.instruction(&Instruction::LocalGet(sum_i));
                         instrs.instruction(&Instruction::I32Const(1));
                         instrs.instruction(&Instruction::I32Add);
                         instrs.instruction(&Instruction::LocalSet(sum_i));
                         
                         instrs.instruction(&Instruction::Br(1));
                         instrs.instruction(&Instruction::End);
                         instrs.instruction(&Instruction::End);
                         
                         instrs.instruction(&Instruction::LocalGet(sum_acc));
                     } else {
                         return Err(anyhow!("sum() only supports ranges in MVP"));
                     }
                } else if name == "rand" {
                    if args.len() != 2 { return Err(anyhow!("rand takes 2 args (min, max)")); }
                    self.compile_expr(&args[0], locals, globals, instrs, loop_var_idx, sum_locals)?;
                    self.compile_expr(&args[1], locals, globals, instrs, loop_var_idx, sum_locals)?;
                    instrs.instruction(&Instruction::Call(4)); // rand
                } else {
                    return Err(anyhow!("Unknown function: {}", name));
                }
            },
            Expr::Block(block) => {
                self.compile_block(block, locals, globals, instrs, sum_locals, true)?;
            },
            Expr::Array(_) => {
                instrs.instruction(&Instruction::F64Const(0.0));
            },
            _ => { },
        }
        Ok(())
    }

    pub fn generate(&mut self) -> Result<Vec<u8>> {
        let mut module = Module::new();
        
        let execution_order = self.grid.iter_execution_order();
        let globals_map = self.scan_globals(&execution_order)?;

        // 1. Types Section
        let mut types = TypeSection::new();
        types.function([], []);
        types.function([ValType::F64], [ValType::F64]);
        types.function([ValType::F64, ValType::F64], [ValType::F64]);
        types.function([ValType::F64], [ValType::F64]);
        types.function([], [ValType::F64]);
        module.section(&types);
        
        // 2. Import Section
        let mut imports = ImportSection::new();
        imports.import("env", "put", EntityType::Function(1));
        imports.import("env", "pow", EntityType::Function(2));
        imports.import("env", "log", EntityType::Function(3));
        imports.import("env", "input", EntityType::Function(4));
        imports.import("env", "rand", EntityType::Function(2)); // Type 2: (f64, f64) -> f64
        module.section(&imports);

        // 3. Function Section
        let mut functions = FunctionSection::new();
        for _ in &execution_order {
            functions.function(0); 
        }
        functions.function(0); 
        module.section(&functions);
        
        // 4. Memory Section
        let mut memory = MemorySection::new();
        memory.memory(MemoryType {
            minimum: 1, 
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        });
        module.section(&memory);

        // 5. Global Section
        let mut global_section = wasm_encoder::GlobalSection::new();
        // Create globals for each entry in map
        let mut sorted_globals: Vec<_> = globals_map.iter().collect();
        sorted_globals.sort_by_key(|k| k.1);
        
        for _ in sorted_globals {
             global_section.global(
                wasm_encoder::GlobalType { val_type: ValType::F64, mutable: true, shared: false },
                &wasm_encoder::ConstExpr::f64_const(0.0)
            );
        }
        module.section(&global_section);

        // 6. Export Section
        let mut exports = ExportSection::new();
        exports.export("memory", ExportKind::Memory, 0);
        // Imports: 0=put, 1=pow, 2=log, 3=input, 4=rand. Cells start at 5.
        exports.export("_start", ExportKind::Func, (execution_order.len() as u32) + 5);
        module.section(&exports);

        // 7. Code Section
        let mut codes = CodeSection::new();
        let mut cell_func_indices = Vec::new();
        // Index 0=put, 1=pow, 2=log. 3=input, 4=rand. Index 5 is first cell.
        let mut next_func_idx = 5;

        for ((col, row), content) in &execution_order {
            let stmts = crate::parser::parse_cell_content(content)?;
            let func = self.compile_cell(*col, *row, &stmts, &globals_map)?;
            codes.function(&func);
            cell_func_indices.push(next_func_idx);
            next_func_idx += 1;
        }

        let mut main_func = Function::new([]);
        for func_idx in cell_func_indices {
            main_func.instruction(&Instruction::Call(func_idx));
        }
        main_func.instruction(&Instruction::End);
        codes.function(&main_func);

        module.section(&codes);

        Ok(module.finish())
    }
}
