use anyhow::{Result, anyhow};
use wasm_encoder::{
    CodeSection, ExportKind, ExportSection, Function, FunctionSection, Instruction, MemArg, MemoryType,
    MemorySection, Module, TypeSection, ValType, ImportSection, EntityType, BlockType, DataSection, ConstExpr,
    TableSection, TableType, RefType, ElementSection, Elements, GlobalSection, GlobalType
};
use crate::loader::CellGrid;
use crate::parser::{Expr, Stmt, Op, Literal, CellRef, CellContent, FunctionDef, Direction, Block};
use std::collections::HashMap;

mod types;
mod expr;
mod stmt;
mod intrinsics;

use types::ParsedCell;

pub struct WasmCompiler<'a> {
    grid: &'a CellGrid,
    parsed_cells: Vec<ParsedCell>,
    string_literals: HashMap<String, (u32, u32)>, // content -> (offset, len)
    next_string_offset: u32,
    all_functions: Vec<(FunctionDef, Option<(u32, u32)>)>, // (func, opt_cell_pos)
}

impl<'a> WasmCompiler<'a> {
    pub fn new(grid: &'a CellGrid) -> Self {
        Self {
            grid,
            parsed_cells: Vec::new(),
            string_literals: HashMap::new(),
            next_string_offset: 1024,
            all_functions: Vec::new(),
        }
    }
    
    fn scan_globals(&self) -> Result<HashMap<String, u32>> {
        let globals = HashMap::new();
        // User globals from A1..ZZ99 (Data cells only?)
        Ok(globals)
    }

    fn scan_all_strings(&mut self) -> Result<()> {
         // Scan data cells
         for row in 0..self.grid.max_row {
            for col in 0..self.grid.max_col {
                if let Some(content) = self.grid.get(col, row) {
                     if content.starts_with('"') && content.ends_with('"') {
                         let s = &content[1..content.len()-1];
                         Self::add_string_literal(&mut self.string_literals, &mut self.next_string_offset, s);
                     }
                }
            }
        }
        
        let string_literals = &mut self.string_literals;
        let next_offset = &mut self.next_string_offset;

        // Scan code cells
        for cell in &self.parsed_cells {
            if let CellContent::Code(f) = &cell.content {
                Self::scan_block_strings(&f.body, string_literals, next_offset);
            }
        }
        Ok(())
    }
    
    fn add_string_literal(map: &mut HashMap<String, (u32, u32)>, next_offset: &mut u32, s: &str) {
         if !map.contains_key(s) {
             map.insert(s.to_string(), (*next_offset, s.len() as u32));
             *next_offset += s.len() as u32 + 1; // null term
         }
    }

    fn scan_block_strings(block: &Block, map: &mut HashMap<String, (u32, u32)>, next_offset: &mut u32) {
        for stmt in &block.stmts {
           Self::scan_stmt_strings(stmt, map, next_offset);
        }
    }

    fn scan_stmt_strings(stmt: &Stmt, map: &mut HashMap<String, (u32, u32)>, next_offset: &mut u32) {
        match stmt {
            Stmt::Assign(_, expr) => Self::scan_expr_strings(expr, map, next_offset),
            Stmt::Return(expr) => Self::scan_expr_strings(expr, map, next_offset),
            Stmt::Yield(expr) => Self::scan_expr_strings(expr, map, next_offset),
            Stmt::Expr(expr) => Self::scan_expr_strings(expr, map, next_offset),
            Stmt::While(expr, body) => {
                Self::scan_expr_strings(expr, map, next_offset);
                Self::scan_block_strings(body, map, next_offset);
            },
            Stmt::For(_, expr, body) => {
                Self::scan_expr_strings(expr, map, next_offset);
                Self::scan_block_strings(body, map, next_offset);
            },
            Stmt::FnDef(_, _, body) => Self::scan_block_strings(body, map, next_offset),
            Stmt::Unpack(_, expr) => Self::scan_expr_strings(expr, map, next_offset),
        }
    }

    fn scan_expr_strings(expr: &Expr, map: &mut HashMap<String, (u32, u32)>, next_offset: &mut u32) {
        match expr {
            Expr::Literal(Literal::String(s)) => Self::add_string_literal(map, next_offset, s),
            Expr::BinaryOp(lhs, _, rhs) => { Self::scan_expr_strings(lhs, map, next_offset); Self::scan_expr_strings(rhs, map, next_offset); },
            Expr::UnaryOp(_, expr) => Self::scan_expr_strings(expr, map, next_offset),
            Expr::Call(_, args) => { for arg in args { Self::scan_expr_strings(arg, map, next_offset); } },
            Expr::If(cond, then_b, else_b) => {
                Self::scan_expr_strings(cond, map, next_offset);
                Self::scan_block_strings(then_b, map, next_offset);
                if let Some(eb) = else_b { Self::scan_block_strings(eb, map, next_offset); }
            },
            Expr::Cond(branches, else_b) => {
                for (cond, body) in branches {
                    Self::scan_expr_strings(cond, map, next_offset);
                    Self::scan_block_strings(body, map, next_offset);
                }
                if let Some(eb) = else_b { Self::scan_block_strings(eb, map, next_offset); }
            },
            Expr::Recur(args) => { for arg in args { Self::scan_expr_strings(arg, map, next_offset); } },
            Expr::Amend(v, i, f) | Expr::Drill(v, i, f) => {
                Self::scan_expr_strings(v, map, next_offset);
                Self::scan_expr_strings(i, map, next_offset);
                Self::scan_expr_strings(f, map, next_offset);
            },
            Expr::Array(items) => { for item in items { Self::scan_expr_strings(item, map, next_offset); } },
            Expr::InstanceApply(_, _, _, _, body, _) => {
                if let Some(b) = body { Self::scan_block_strings(b, map, next_offset); }
            },
            _ => {}
        }
    }

    /// Scan all blocks for nested functions
    fn scan_all_functions(&mut self) -> Result<()> {
        for i in 0..self.parsed_cells.len() {
            let cell = &self.parsed_cells[i];
            let content = match &cell.content {
                CellContent::Code(f) => Some((f.clone(), cell.col, cell.row)),
                _ => None,
            };
            
            if let Some((f, col, row)) = content {
                self.all_functions.push((f.clone(), Some((col, row))));
                self.scan_block_functions(&f.body);
            }
        }
        Ok(())
    }

    fn scan_block_functions(&mut self, block: &Block) {
        for stmt in &block.stmts {
            self.scan_stmt_functions(stmt);
        }
    }

    fn scan_stmt_functions(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Assign(_, expr) => self.scan_expr_functions(expr),
            Stmt::Return(expr) => self.scan_expr_functions(expr),
            Stmt::Yield(expr) => self.scan_expr_functions(expr),
            Stmt::Expr(expr) => self.scan_expr_functions(expr),
            Stmt::While(expr, body) => {
                self.scan_expr_functions(expr);
                self.scan_block_functions(body);
            },
            Stmt::For(_, expr, body) => {
                self.scan_expr_functions(expr);
                self.scan_block_functions(body);
            },
            Stmt::FnDef(name, args, body) => {
                self.all_functions.push((FunctionDef {
                    name: name.clone(),
                    args: args.clone(),
                    body: body.clone(),
                    is_active: false,
                }, None));
                self.scan_block_functions(body);
            },
            Stmt::Unpack(_, expr) => self.scan_expr_functions(expr),
            _ => {}
        }
    }

    fn scan_expr_functions(&mut self, expr: &Expr) {
        match expr {
            Expr::BinaryOp(lhs, _, rhs) => { self.scan_expr_functions(lhs); self.scan_expr_functions(rhs); },
            Expr::UnaryOp(_, e) => self.scan_expr_functions(e),
            Expr::Call(_, args) => { for arg in args { self.scan_expr_functions(arg); } },
            Expr::If(cond, then_b, else_b) => {
                self.scan_expr_functions(cond);
                self.scan_block_functions(then_b);
                if let Some(eb) = else_b { self.scan_block_functions(eb); }
            },
            Expr::Cond(branches, else_b) => {
                for (cond, body) in branches {
                    self.scan_expr_functions(cond);
                    self.scan_block_functions(body);
                }
                if let Some(eb) = else_b { self.scan_block_functions(eb); }
            },
            Expr::Recur(args) => { for arg in args { self.scan_expr_functions(arg); } },
            Expr::InstanceApply(_, _, name, args, body, is_active) => {
                if let Some(b) = body {
                    self.all_functions.push((FunctionDef {
                        name: name.clone(),
                        args: args.clone().unwrap_or_default(),
                        body: b.clone(),
                        is_active: *is_active,
                    }, None));
                    self.scan_block_functions(b);
                }
            },
            Expr::Array(items) => { for item in items { self.scan_expr_functions(item); } },
            _ => {}
        }
    }

    /// Parse all cells and store in parsed_cells
    fn parse_all_cells(&mut self) -> Result<()> {
        for row in 0..self.grid.max_row {
            for col in 0..self.grid.max_col {
                if let Some(content) = self.grid.get(col, row) {
                    let parsed = crate::parser::parse_cell(content)?;
                    self.parsed_cells.push(ParsedCell { col, row, content: parsed });
                }
            }
        }
        Ok(())
    }

    fn get_cell_memory_offset(&self, col: u32, row: u32) -> u32 {
        (row * self.grid.max_col + col) * 8
    }

    pub fn generate(&mut self) -> Result<Vec<u8>> {
        // 1. Parse all cells
        self.parse_all_cells()?;
        self.scan_all_functions()?;
        let globals_map = self.scan_globals()?; 
        self.scan_all_strings()?; // Populate string map

        // Separate data/code
        let _data_cells: Vec<_> = self.parsed_cells.iter()
            .filter(|c| matches!(c.content, CellContent::Data(_)))
            .collect();
        let code_cells: Vec<_> = self.parsed_cells.iter()
            .filter(|c| matches!(c.content, CellContent::Code(_)))
            .collect();
            
        let mut module = Module::new();

        // 1. Types
        let mut types = TypeSection::new();
        types.function([], []);                              // 0: void->void
        types.function([ValType::F64], [ValType::F64]);      // 1: f64->f64
        types.function([ValType::F64, ValType::F64], [ValType::F64]); // 2: (f64,f64)->f64
        types.function([ValType::F64], [ValType::F64]);      // 3: 
        types.function([], [ValType::F64]);                  // 4: ->f64
        types.function([ValType::I32, ValType::I32], []);    // 5: (i32,i32)->void (print)
        types.function([ValType::I32], [ValType::I32]);      // 6: i32->i32 (allocate)
        types.function([ValType::I32, ValType::I32, ValType::I32], []); // 7: (i32,i32,i32)->void (process_data)
        
        // Dynamic types for cell functions: [f64; N] -> [i32] (Return next func index)
        // Base type index for cell types starts at 8
        let mut arity_to_type_idx = HashMap::new();
        let mut next_type_idx = 8;
        
        let mut arities = std::collections::HashSet::new();
        arities.insert(0); 
        for (f, _) in &self.all_functions {
            arities.insert(f.args.len());
        }
        let mut sorted_arities: Vec<_> = arities.into_iter().collect();
        sorted_arities.sort();
        
        for arity in sorted_arities {
            let params: Vec<ValType> = (0..arity).map(|_| ValType::F64).collect();
            types.function(params, [ValType::I32]); 
            arity_to_type_idx.insert(arity, next_type_idx);
            next_type_idx += 1;
        }
        module.section(&types);

        // 2. Imports
        let mut imports = ImportSection::new();
        imports.import("env", "put", EntityType::Function(1));
        imports.import("env", "pow", EntityType::Function(2));
        imports.import("env", "log", EntityType::Function(3));
        imports.import("env", "input", EntityType::Function(4));
        imports.import("env", "rand", EntityType::Function(2));
        imports.import("env", "print", EntityType::Function(5));
        
        imports.import("env", "sin", EntityType::Function(1));
        imports.import("env", "cos", EntityType::Function(1));
        imports.import("env", "tan", EntityType::Function(1));
        imports.import("env", "asin", EntityType::Function(1));
        imports.import("env", "acos", EntityType::Function(1));
        imports.import("env", "atan", EntityType::Function(1));
        imports.import("env", "hypot", EntityType::Function(2));
        imports.import("env", "fmod", EntityType::Function(2)); // Import fmod as type 2
        module.section(&imports);

        // 3. Functions
        let mut functions = FunctionSection::new();
        let mut cell_func_map = HashMap::new();
        let mut func_name_map = HashMap::new();
        let mut func_arity_map = HashMap::new();
        let mut func_indices = Vec::new(); // indices in the table
        
        // Imports: 0-13 (14 imports). Start func idx = 14.
        let start_func_idx = 14;
        
        for (i, (f, cell_pos)) in self.all_functions.iter().enumerate() {
            let arity = f.args.len();
            let type_idx = *arity_to_type_idx.get(&arity).unwrap();
            functions.function(type_idx);
            
            let table_idx = start_func_idx + i as u32;
            if let Some(pos) = cell_pos {
                cell_func_map.insert(*pos, table_idx);
            }
            func_indices.push(table_idx);
            
            func_name_map.insert(f.name.clone(), table_idx);
            func_arity_map.insert(f.name.clone(), arity as u32);
        }
        
        // Init (type 0)
        functions.function(0);
        // Main Loop (type 0)
        functions.function(0);
        // Fast Inv Sqrt (type 1)
        functions.function(1); 
        // Fast Hypot (type 2)
        functions.function(2);
        // Allocate (type 6)
        functions.function(6);
        // ProcessData (type 7)
        functions.function(7);
        // Runtime Min (type 2) - f64,f64 -> f64 which matches Hypot signature (Type 2)
        functions.function(2); 
        
        module.section(&functions);

        // 4. Tables
        let mut tables = TableSection::new();
        let table_count = start_func_idx + self.all_functions.len() as u32;
        tables.table(TableType {
            element_type: RefType::FUNCREF,
            minimum: (table_count + 100) as u64,
            maximum: None,
            table64: false,
        });
        module.section(&tables);

        // 5. Memories
        let mut memories = MemorySection::new();
        memories.memory(MemoryType {
            minimum: 100, 
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        });
        module.section(&memories);

        // 6. Globals
        let mut global_section = GlobalSection::new();
        let num_user_globals = globals_map.len() as u32;
        for _ in 0..num_user_globals {
             global_section.global(
                GlobalType { val_type: ValType::F64, mutable: true, shared: false },
                &ConstExpr::f64_const(0.0)
            );
        }
        // Register value global
        let reg_val_idx = num_user_globals;
        global_section.global(
            GlobalType { val_type: ValType::F64, mutable: true, shared: false },
            &ConstExpr::f64_const(0.0)
        );
        // Heap Pointer Global
        let heap_ptr_idx = reg_val_idx + 1;
        global_section.global(
             GlobalType { val_type: ValType::I32, mutable: true, shared: false },
             &ConstExpr::i32_const((self.grid.max_col * self.grid.max_row * 8 + 1024) as i32) // Start heap after grid + strings
        );
        // CUR_COL
        let cur_col_idx = heap_ptr_idx + 1;
        global_section.global(
            GlobalType { val_type: ValType::I32, mutable: true, shared: false },
            &ConstExpr::i32_const(0)
        );
        // CUR_ROW
        let cur_row_idx = heap_ptr_idx + 2;
        global_section.global(
            GlobalType { val_type: ValType::I32, mutable: true, shared: false },
            &ConstExpr::i32_const(0)
        );
        module.section(&global_section);

        // 7. Exports
        let mut exports = ExportSection::new();
        exports.export("memory", ExportKind::Memory, 0);
        let init_func_idx = start_func_idx + self.all_functions.len() as u32;
        let main_func_idx = init_func_idx + 1;
        let alloc_idx = init_func_idx + 4;
        let process_idx = init_func_idx + 5;
        
        exports.export("run", ExportKind::Func, main_func_idx);
        exports.export("allocate", ExportKind::Func, alloc_idx);
        exports.export("process_data", ExportKind::Func, process_idx);
        module.section(&exports);
        
        // 8. Elements
        // Check inferred type index for call_indirect.
        // We defaulted to 6 in `main` loop. It should be types.len() - 2 + arity type..
        // Wait, arity types start at 8. 
        // 0-arity func is type 8.
        // We need to fix `main` loop type index.
        
        let mut elements = ElementSection::new();
        let func_refs: Vec<u32> = (start_func_idx .. (start_func_idx + self.all_functions.len() as u32)).collect();
        if !func_refs.is_empty() {
             elements.active(Some(0), &ConstExpr::i32_const(start_func_idx as i32), Elements::Functions(func_refs.as_slice()));
        }
        module.section(&elements);
        
         // 9. Code (Bodies)
        let mut codes = CodeSection::new();
        
        // Compile all functions
        for (i, (f_def, cell_pos)) in self.all_functions.iter().enumerate() {
            let (col, row) = cell_pos.unwrap_or((0, 0));
            let use_context_init = cell_pos.is_some();
            let table_idx = start_func_idx + i as u32;
            let func = stmt::compile_cell(
                col, row, f_def, &globals_map, &cell_func_map, 
                &func_name_map, &func_arity_map, &arity_to_type_idx, table_idx, 
                reg_val_idx, cur_col_idx, cur_row_idx,
                &self.string_literals,
                self.grid,
                use_context_init
            )?;
            codes.function(&func);
        }

        // Init
        // We need to pass data cells BUT struct ParsedCell is used. 
        // We can just filter again or pass slice.
        // Actually intrinsics::generate_init_func expects `&[ParsedCell]`.
        codes.function(&intrinsics::generate_init_func(&self.parsed_cells, self.grid));

        // Main Loop
        // We need active cells..
        let active_indices: Vec<u32> = self.all_functions.iter().enumerate()
            .filter(|(_, (f, _))| f.is_active)
            .map(|(i, _)| start_func_idx + i as u32)
            .collect();
        
        let num_total_functions = self.all_functions.len() as u32;
        let mut main_func = Function::new([(1, ValType::I32)]); 
        // Call init
        main_func.instruction(&Instruction::Call(init_func_idx));
        
        if !active_indices.is_empty() {
            main_func.instruction(&Instruction::I32Const(active_indices[0] as i32));
            main_func.instruction(&Instruction::LocalSet(0));
            main_func.instruction(&Instruction::Loop(BlockType::Empty));
            
            // Call Indirect
            main_func.instruction(&Instruction::LocalGet(0));
             // Type for 0-arity func is 0 (void->void) or type 4 (void->f64)??
             // Wait, user functions return i32 (table index).
             // Type for cells (arity 0) is type_idx from map.
             // arity 0 -> type 6 (assuming) which is []->[i32].
             // We need to look it up.
            let type_idx = *arity_to_type_idx.get(&0).unwrap_or(&8); // Default 8 (0-arity)
            main_func.instruction(&Instruction::CallIndirect { ty: type_idx, table: 0 });
            main_func.instruction(&Instruction::LocalSet(0)); // Update next func index
            
            // If index > num_code, stop? 
            // Our convention: return code_cells.len() + 1 to stop.
            main_func.instruction(&Instruction::LocalGet(0));
            main_func.instruction(&Instruction::I32Const(num_total_functions as i32));
            main_func.instruction(&Instruction::I32LeU);
            main_func.instruction(&Instruction::BrIf(0));
            main_func.instruction(&Instruction::End);
        }
        main_func.instruction(&Instruction::End);
        codes.function(&main_func);
        
        // Fast Inv Sqrt
        codes.function(&intrinsics::generate_fast_inv_sqrt());
        // Fast Hypot
        codes.function(&intrinsics::generate_fast_hypot());
        // Allocate
        codes.function(&intrinsics::generate_alloc_with_global(heap_ptr_idx));
        // Process Data
        codes.function(&intrinsics::generate_process_data(self.grid.max_col));
        // Runtime Min
        codes.function(&intrinsics::generate_runtime_min());
        
        module.section(&codes);
        
        // 10. Data (Strings)
        let mut data = DataSection::new();
        // Since we map string -> offset
        // We can just iterate map and create data segments?
        // Or create one big blob. Simplest is one segment per string.
        for (s, (offset, _len)) in &self.string_literals {
             let bytes = s.as_bytes();
             let mut data_vec = bytes.to_vec();
             data_vec.push(0); // Null terminator (optional if we pass len)
             
             data.active(0, &ConstExpr::i32_const(*offset as i32), data_vec);
        }
        module.section(&data);
        
        Ok(module.finish())
    }
}
