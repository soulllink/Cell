use anyhow::{Result, anyhow};
use wasm_encoder::{
    CodeSection, ExportKind, ExportSection, Function, FunctionSection, Instruction, MemArg, MemoryType,
    MemorySection, Module, TypeSection, ValType, ImportSection, EntityType, BlockType, DataSection, ConstExpr,
    TableSection, TableType, RefType, ElementSection, Elements
};
use crate::loader::CellGrid;
use crate::parser::{Expr, Stmt, Op, Literal, CellRef};
use std::collections::HashMap;

pub struct WasmCompiler<'a> {
    grid: &'a CellGrid,
    cell_offsets: HashMap<(u32, u32), u32>,
    next_func_index: u32,
    string_literals: HashMap<String, (u32, u32)>, // String -> (offset, len)
}

impl<'a> WasmCompiler<'a> {
    pub fn new(grid: &'a CellGrid) -> Self {
        Self {
            grid,
            cell_offsets: HashMap::new(),
            next_func_index: 0,
            string_literals: HashMap::new(),
        }
    }

    fn get_cell_memory_offset(&self, col: u32, row: u32) -> u32 {
        let max_rows = self.grid.max_row.max(100); 
        (col * max_rows + row) * 8
    }

    fn scan_locals(stmts: &[Stmt], locals_map: &mut HashMap<String, u32>, locals_types: &mut Vec<ValType>, next_idx: &mut u32) {
        for stmt in stmts {
            match stmt {
                Stmt::For(name, _, body) => {
                    if !locals_map.contains_key(name) {
                        locals_map.insert(name.clone(), *next_idx);
                        locals_types.push(ValType::F64);
                        *next_idx += 1;
                    }
                    Self::scan_locals(&body.stmts, locals_map, locals_types, next_idx);
                },
                Stmt::While(_, body) => Self::scan_locals(&body.stmts, locals_map, locals_types, next_idx),
                Stmt::Expr(expr) | Stmt::Return(expr) | Stmt::Yield(expr) => Self::scan_expr_locals(expr, locals_map, locals_types, next_idx),
                 _ => {}
            }
        }
    }

    fn scan_expr_locals(expr: &Expr, locals_map: &mut HashMap<String, u32>, locals_types: &mut Vec<ValType>, next_idx: &mut u32) {
        match expr {
            Expr::If(_, then_b, else_b) => {
                 Self::scan_locals(&then_b.stmts, locals_map, locals_types, next_idx);
                 if let Some(eb) = else_b { Self::scan_locals(&eb.stmts, locals_map, locals_types, next_idx); }
            },
            Expr::Block(b) => Self::scan_locals(&b.stmts, locals_map, locals_types, next_idx),
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
                Stmt::While(_, body) | Stmt::For(_, _, body) => self.scan_stmts_for_globals(&body.stmts, globals, next_idx),
                Stmt::Expr(expr) | Stmt::Return(expr) | Stmt::Yield(expr) => self.scan_expr_for_blocks(expr, globals, next_idx),
                _ => {}
            }
        }
    }
    
    fn scan_expr_for_blocks(&self, expr: &Expr, globals: &mut HashMap<String, u32>, next_idx: &mut u32) {
         match expr {
            Expr::If(_, then_b, else_b) => {
                 self.scan_stmts_for_globals(&then_b.stmts, globals, next_idx);
                 if let Some(eb) = else_b { self.scan_stmts_for_globals(&eb.stmts, globals, next_idx); }
            },
            Expr::Block(b) => self.scan_stmts_for_globals(&b.stmts, globals, next_idx),
            _ => {}
         }
    }

    fn scan_all_strings(&mut self, execution_order: &Vec<((u32, u32), &String)>) -> Result<()> {
         let mut current_offset = 1048576;
         for (_, content) in execution_order {
             let stmts = crate::parser::parse_cell_content(content)?;
             self.scan_stmts_for_strings(&stmts, &mut current_offset);
         }
         Ok(())
    }

    fn scan_stmts_for_strings(&mut self, stmts: &[Stmt], offset: &mut u32) {
        for stmt in stmts {
            match stmt {
                Stmt::Assign(_, expr) | Stmt::Return(expr) | Stmt::Yield(expr) | Stmt::Expr(expr) => {
                    self.scan_expr_for_strings(expr, offset);
                },
                Stmt::While(cond, body) => {
                    self.scan_expr_for_strings(cond, offset);
                    self.scan_stmts_for_strings(&body.stmts, offset);
                },
                Stmt::For(_, iter, body) => {
                    self.scan_expr_for_strings(iter, offset);
                    self.scan_stmts_for_strings(&body.stmts, offset);
                },
                _ => {}
            }
        }
    }

    fn scan_expr_for_strings(&mut self, expr: &Expr, offset: &mut u32) {
        match expr {
             Expr::Literal(Literal::String(s)) => {
                 if !self.string_literals.contains_key(s) {
                     let len = s.len() as u32;
                     self.string_literals.insert(s.clone(), (*offset, len));
                     *offset += len; 
                 }
             },
             Expr::If(cond, then_b, else_b) => {
                 self.scan_expr_for_strings(cond, offset);
                 self.scan_stmts_for_strings(&then_b.stmts, offset);
                 if let Some(eb) = else_b { self.scan_stmts_for_strings(&eb.stmts, offset); }
             },
             Expr::Call(_, args) | Expr::Array(args) => {
                 for arg in args { self.scan_expr_for_strings(arg, offset); }
             },
             Expr::BinaryOp(lhs, _, rhs) => {
                 self.scan_expr_for_strings(lhs, offset);
                 self.scan_expr_for_strings(rhs, offset);
             },
             Expr::Block(b) => self.scan_stmts_for_strings(&b.stmts, offset),
             _ => {}
        }
    }

    fn compile_cell(&self, col: u32, row: u32, stmts: &[Stmt], globals: &HashMap<String, u32>, cell_func_map: &HashMap<(u32, u32), u32>, my_table_idx: u32, reg_val_global_idx: u32) -> Result<Function> {
        let mut locals_map = HashMap::new();
        let mut locals_types = Vec::new();
        let mut next_local_idx = 0;

        Self::scan_locals(stmts, &mut locals_map, &mut locals_types, &mut next_local_idx);
        
        let sum_i_idx = next_local_idx; locals_types.push(ValType::I32); next_local_idx += 1;
        let sum_len_idx = next_local_idx; locals_types.push(ValType::I32); next_local_idx += 1;
        let sum_acc_idx = next_local_idx; locals_types.push(ValType::F64); 

        let mut func = Function::new(locals_types.iter().map(|n| (1, *n)));
        let sum_locals_tuple = (sum_i_idx, sum_len_idx, sum_acc_idx);

        let mut explicit_return = false;

        for stmt in stmts {
             match stmt {
                Stmt::Assign(name, expr) => {
                    self.compile_expr(expr, &locals_map, globals, &mut func, sum_locals_tuple, cell_func_map, reg_val_global_idx, None, my_table_idx)?;
                    if let Some(idx) = locals_map.get(name) {
                        func.instruction(&Instruction::LocalSet(*idx));
                    } else if let Some(idx) = globals.get(name) {
                        func.instruction(&Instruction::GlobalSet(*idx));
                    }
                },
                Stmt::Return(expr) => {
                     if let Expr::Reference(r) = expr {
                         if let Some(target_idx) = cell_func_map.get(&(r.col, r.row)) {
                             func.instruction(&Instruction::I32Const(*target_idx as i32));
                             func.instruction(&Instruction::Return);
                             explicit_return = true;
                         } else {
                             func.instruction(&Instruction::I32Const((my_table_idx + 1) as i32));
                             func.instruction(&Instruction::Return);
                             explicit_return = true;
                         }
                     } else {
                         self.compile_expr(expr, &locals_map, globals, &mut func, sum_locals_tuple, cell_func_map, reg_val_global_idx, None, my_table_idx)?;
                         func.instruction(&Instruction::Drop);
                         func.instruction(&Instruction::I32Const((my_table_idx + 1) as i32));
                         func.instruction(&Instruction::Return);
                         explicit_return = true;
                     }
                },
                 Stmt::Yield(expr) => {
                    self.compile_expr(expr, &locals_map, globals, &mut func, sum_locals_tuple, cell_func_map, reg_val_global_idx, None, my_table_idx)?;
                    func.instruction(&Instruction::GlobalSet(reg_val_global_idx)); 
                    func.instruction(&Instruction::I32Const(my_table_idx as i32)); 
                    func.instruction(&Instruction::Return);
                    explicit_return = true;
                },
                Stmt::Expr(expr) => {
                    self.compile_expr(expr, &locals_map, globals, &mut func, sum_locals_tuple, cell_func_map, reg_val_global_idx, None, my_table_idx)?;
                    func.instruction(&Instruction::Drop);
                },
                Stmt::While(cond, body) => {
                     func.instruction(&Instruction::Loop(wasm_encoder::BlockType::Empty)); 
                     func.instruction(&Instruction::Block(wasm_encoder::BlockType::Empty)); 
                     self.compile_expr(cond, &locals_map, globals, &mut func, sum_locals_tuple, cell_func_map, reg_val_global_idx, None, my_table_idx)?;
                     func.instruction(&Instruction::F64Const(0.0));
                     func.instruction(&Instruction::F64Eq); 
                     func.instruction(&Instruction::BrIf(0)); 
                     self.compile_block(body, &locals_map, globals, &mut func, sum_locals_tuple, false, cell_func_map, reg_val_global_idx, None, my_table_idx)?; 
                     func.instruction(&Instruction::Br(1)); 
                     func.instruction(&Instruction::End); 
                     func.instruction(&Instruction::End); 
                },
                _ => {}
            }
        }
        
        if !explicit_return {
            func.instruction(&Instruction::I32Const((my_table_idx + 1) as i32));
        }
        func.instruction(&Instruction::End);
        Ok(func)
    }

    fn compile_block(&self, block: &crate::parser::Block, locals: &HashMap<String, u32>, globals: &HashMap<String, u32>, instrs: &mut Function, sum_locals: (u32, u32, u32), needs_result: bool, cell_func_map: &HashMap<(u32, u32), u32>, reg_val_idx: u32, loop_var_idx: Option<u32>, my_table_idx: u32) -> Result<()> {
        let len = block.stmts.len();
        if len == 0 && needs_result { instrs.instruction(&Instruction::F64Const(0.0)); return Ok(()); }
        for (i, stmt) in block.stmts.iter().enumerate() {
            let is_last = i == len - 1;
             match stmt {
                Stmt::Assign(name, expr) => {
                    self.compile_expr(expr, locals, globals, instrs, sum_locals, cell_func_map, reg_val_idx, loop_var_idx, my_table_idx)?;
                    if let Some(idx) = locals.get(name) { instrs.instruction(&Instruction::LocalSet(*idx)); }
                    else if let Some(idx) = globals.get(name) { instrs.instruction(&Instruction::GlobalSet(*idx)); }
                    if is_last && needs_result { instrs.instruction(&Instruction::F64Const(0.0)); }
                },
                Stmt::Return(expr) => {
                     if let Expr::Reference(r) = expr {
                         if let Some(target_idx) = cell_func_map.get(&(r.col, r.row)) {
                             instrs.instruction(&Instruction::I32Const(*target_idx as i32));
                         } else {
                             instrs.instruction(&Instruction::I32Const((my_table_idx + 1) as i32));
                         }
                     } else {
                         self.compile_expr(expr, locals, globals, instrs, sum_locals, cell_func_map, reg_val_idx, loop_var_idx, my_table_idx)?;
                         instrs.instruction(&Instruction::Drop);
                         instrs.instruction(&Instruction::I32Const((my_table_idx + 1) as i32)); 
                     }
                     instrs.instruction(&Instruction::Return);
                },
                Stmt::Yield(expr) => { 
                     self.compile_expr(expr, locals, globals, instrs, sum_locals, cell_func_map, reg_val_idx, loop_var_idx, my_table_idx)?;
                     instrs.instruction(&Instruction::GlobalSet(reg_val_idx));
                     instrs.instruction(&Instruction::Return);
                },
                Stmt::Expr(expr) => {
                    self.compile_expr(expr, locals, globals, instrs, sum_locals, cell_func_map, reg_val_idx, loop_var_idx, my_table_idx)?;
                    if !is_last || !needs_result { instrs.instruction(&Instruction::Drop); }
                },
                 _ => { if is_last && needs_result { instrs.instruction(&Instruction::F64Const(0.0)); } }
             }
        }
        Ok(())
    }

    fn compile_expr(&self, expr: &Expr, locals: &HashMap<String, u32>, globals: &HashMap<String, u32>, instrs: &mut Function, sum_locals: (u32, u32, u32), cell_func_map: &HashMap<(u32, u32), u32>, reg_val_idx: u32, loop_var_idx: Option<u32>, my_table_idx: u32) -> Result<()> {
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
            Expr::If(cond, then_b, else_b) => {
                 self.compile_expr(cond, locals, globals, instrs, sum_locals, cell_func_map, reg_val_idx, loop_var_idx, my_table_idx)?;
                 instrs.instruction(&Instruction::F64Const(0.0));
                 instrs.instruction(&Instruction::F64Ne); 
                 instrs.instruction(&Instruction::If(BlockType::Result(ValType::F64)));
                 self.compile_block(then_b, locals, globals, instrs, sum_locals, true, cell_func_map, reg_val_idx, loop_var_idx, my_table_idx)?;
                 instrs.instruction(&Instruction::Else);
                 if let Some(eb) = else_b {
                     self.compile_block(eb, locals, globals, instrs, sum_locals, true, cell_func_map, reg_val_idx, loop_var_idx, my_table_idx)?;
                 } else {
                     instrs.instruction(&Instruction::F64Const(0.0));
                 }
                 instrs.instruction(&Instruction::End);
            },
            Expr::Generator(col) => {
                if let Some(idx) = cell_func_map.get(&(*col, 0)) {
                    instrs.instruction(&Instruction::I32Const(*idx as i32));
                    instrs.instruction(&Instruction::CallIndirect { ty: 6, table: 0 }); 
                    instrs.instruction(&Instruction::Drop); 
                    instrs.instruction(&Instruction::GlobalGet(reg_val_idx)); 
                } else {
                    instrs.instruction(&Instruction::F64Const(0.0));
                }
            },
            Expr::Input => { instrs.instruction(&Instruction::Call(3)); },
            Expr::Call(name, args) => {
                if name == "put" {
                   for (i, arg) in args.iter().enumerate() {
                       if let Expr::Literal(Literal::String(s)) = arg {
                            if let Some(&(offset, len)) = self.string_literals.get(s) {
                                instrs.instruction(&Instruction::I32Const(offset as i32));
                                instrs.instruction(&Instruction::I32Const(len as i32));
                                instrs.instruction(&Instruction::Call(5)); 
                                instrs.instruction(&Instruction::F64Const(0.0));
                            } else {
                                instrs.instruction(&Instruction::F64Const(0.0));
                                instrs.instruction(&Instruction::Call(0)); 
                            }
                       } else {
                           self.compile_expr(arg, locals, globals, instrs, sum_locals, cell_func_map, reg_val_idx, loop_var_idx, my_table_idx)?;
                           instrs.instruction(&Instruction::Call(0)); 
                       }
                       if i < args.len() - 1 { instrs.instruction(&Instruction::Drop); }
                   }
                } else if name == "rand" {
                    self.compile_expr(&args[0], locals, globals, instrs, sum_locals, cell_func_map, reg_val_idx, loop_var_idx, my_table_idx)?;
                    self.compile_expr(&args[1], locals, globals, instrs, sum_locals, cell_func_map, reg_val_idx, loop_var_idx, my_table_idx)?;
                    instrs.instruction(&Instruction::Call(4)); 
                }
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
                self.compile_expr(lhs, locals, globals, instrs, sum_locals, cell_func_map, reg_val_idx, loop_var_idx, my_table_idx)?;
                self.compile_expr(rhs, locals, globals, instrs, sum_locals, cell_func_map, reg_val_idx, loop_var_idx, my_table_idx)?;
                match op {
                    Op::Add => { instrs.instruction(&Instruction::F64Add); },
                    Op::Sub => { instrs.instruction(&Instruction::F64Sub); },
                    Op::Mul => { instrs.instruction(&Instruction::F64Mul); },
                    Op::Div => { instrs.instruction(&Instruction::F64Div); },
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
                    _ => { instrs.instruction(&Instruction::Drop); instrs.instruction(&Instruction::Drop); instrs.instruction(&Instruction::F64Const(0.0)); }
                }
            },
            Expr::Ident(name) => {
                if let Some(idx) = locals.get(name) { instrs.instruction(&Instruction::LocalGet(*idx)); }
                else if let Some(idx) = globals.get(name) { instrs.instruction(&Instruction::GlobalGet(*idx)); }
                else { instrs.instruction(&Instruction::F64Const(0.0)); }
            },
            _ => { instrs.instruction(&Instruction::F64Const(0.0)); },
        }
        Ok(())
    }

    pub fn generate(&mut self) -> Result<Vec<u8>> {
        let mut module = Module::new();
        let execution_order = self.grid.iter_execution_order();
        let globals_map = self.scan_globals(&execution_order)?;
        self.scan_all_strings(&execution_order)?;

        // 1. Types
        let mut types = TypeSection::new();
        types.function([], []); // 0
        types.function([ValType::F64], [ValType::F64]); // 1
        types.function([ValType::F64, ValType::F64], [ValType::F64]); // 2
        types.function([ValType::F64], [ValType::F64]); // 3
        types.function([], [ValType::F64]); // 4
        types.function([ValType::I32, ValType::I32], []); // 5
        types.function([], [ValType::I32]); // 6: Cell Function () -> NextIdx (I32)
        module.section(&types);
        
        // 2. Imports
        let mut imports = ImportSection::new();
        imports.import("env", "put", EntityType::Function(1));
        imports.import("env", "pow", EntityType::Function(2));
        imports.import("env", "log", EntityType::Function(3));
        imports.import("env", "input", EntityType::Function(4));
        imports.import("env", "rand", EntityType::Function(2));
        imports.import("env", "print", EntityType::Function(5));
        module.section(&imports);

        // 3. Functions (Signatures)
        let mut functions = FunctionSection::new();
        let mut cell_func_map = HashMap::new();
        let mut func_indices = Vec::new();
        let start_func_idx = 6; 
        for (i, ((col, row), _)) in execution_order.iter().enumerate() {
            functions.function(6); 
            cell_func_map.insert((*col, *row), i as u32);
            func_indices.push(start_func_idx + i as u32);
        }
        functions.function(0); // Main loop
        module.section(&functions);

        // 4. Tables
        let num_cells = execution_order.len() as u32;
        let mut tables = TableSection::new();
        tables.table(TableType { 
            element_type: RefType::FUNCREF, 
            minimum: num_cells as u64, 
            maximum: Some(num_cells as u64),
            table64: false 
        });
        module.section(&tables);

        // 5. Memories
        let mut memory = MemorySection::new();
        memory.memory(MemoryType { minimum: 20, maximum: None, memory64: false, shared: false, page_size_log2: None });
        module.section(&memory);

        // 6. Globals
        let mut global_section = wasm_encoder::GlobalSection::new();
        let mut sorted_globals: Vec<_> = globals_map.iter().collect();
        sorted_globals.sort_by_key(|k| k.1);
        for _ in &sorted_globals {
             global_section.global(wasm_encoder::GlobalType { val_type: ValType::F64, mutable: true, shared: false }, &wasm_encoder::ConstExpr::f64_const(0.0));
        }
        // REG_VAL
        let reg_val_idx = sorted_globals.len() as u32;
        global_section.global(wasm_encoder::GlobalType { val_type: ValType::F64, mutable: true, shared: false }, &wasm_encoder::ConstExpr::f64_const(0.0));
        module.section(&global_section);

        // 7. Exports
        let mut exports = ExportSection::new();
        exports.export("memory", ExportKind::Memory, 0);
        exports.export("_start", ExportKind::Func, start_func_idx + num_cells);
        module.section(&exports);

        // 8. Elements
        let mut elements = ElementSection::new();
        elements.active(Some(0), &ConstExpr::i32_const(0), Elements::Functions(&func_indices));
        module.section(&elements);

         // 9. Code
        let mut codes = CodeSection::new();
        for (i, ((col, row), content)) in execution_order.iter().enumerate() {
            let stmts = crate::parser::parse_cell_content(content)?;
            let func = self.compile_cell(*col, *row, &stmts, &globals_map, &cell_func_map, i as u32, reg_val_idx)?;
            codes.function(&func);
        }

        // Main Loop
        let mut main_func = Function::new([(1, ValType::I32)]); 
        main_func.instruction(&Instruction::I32Const(0));
        main_func.instruction(&Instruction::LocalSet(0));
        main_func.instruction(&Instruction::Loop(BlockType::Empty));
        main_func.instruction(&Instruction::LocalGet(0));
        main_func.instruction(&Instruction::I32Const(num_cells as i32));
        main_func.instruction(&Instruction::I32GeU);
        main_func.instruction(&Instruction::BrIf(1)); 
        main_func.instruction(&Instruction::LocalGet(0)); 
        main_func.instruction(&Instruction::CallIndirect { ty: 6, table: 0 });
        main_func.instruction(&Instruction::LocalSet(0));
        main_func.instruction(&Instruction::Br(0));
        main_func.instruction(&Instruction::End);
        main_func.instruction(&Instruction::End); 
        codes.function(&main_func);

        module.section(&codes);
        
        // 10. Data
        if !self.string_literals.is_empty() {
             let mut data_section = DataSection::new();
             for (s, (offset, _)) in &self.string_literals {
                 let bytes = s.as_bytes().to_vec();
                 data_section.active(0, &ConstExpr::i32_const(*offset as i32), bytes);
             }
             module.section(&data_section);
        }

        Ok(module.finish())
    }
}
