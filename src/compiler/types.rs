use std::collections::HashMap;
use crate::parser::{CellContent, FunctionDef};
use crate::loader::CellGrid;

/// Parsed cell with resolved position
#[derive(Debug)]
pub struct ParsedCell {
    pub col: u32,
    pub row: u32,
    pub content: CellContent,
}

/// Context passed down to compilation functions to avoid 10+ arguments
pub struct CompilerContext<'a> {
    pub col: u32,
    pub row: u32,
    pub locals: &'a HashMap<String, u32>,
    pub globals: &'a HashMap<String, u32>,
    pub cell_func_map: &'a HashMap<(u32, u32), u32>,
    pub func_name_map: &'a HashMap<String, u32>,
    pub func_arity_map: &'a HashMap<String, u32>,
    pub arity_to_type_idx: &'a HashMap<usize, u32>,
    pub reg_val_idx: u32,
    pub cur_col_idx: u32,
    pub cur_row_idx: u32,
    pub loop_var_idx: Option<u32>,
    pub my_table_idx: u32,
    pub scratch_i32_idx: u32,
    pub scratch_i32_idx_2: u32,
    pub string_literals: &'a HashMap<String, (u32, u32)>,
    pub grid: &'a CellGrid,
}
