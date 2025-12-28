use anyhow::{Result, Context};
use calamine::{Reader, open_workbook, Data, Ods};
use std::collections::HashMap;
use std::path::Path;
use std::fs;

#[derive(Debug)]
pub struct CellGrid {
    pub cells: HashMap<(u32, u32), String>,
    pub max_col: u32,
    pub max_row: u32,
    pub merged_regions: Vec<((u32, u32), (u32, u32))>, 
}

impl CellGrid {
    fn sanitize_latin(input: String) -> String {
        input.replace('С', "C").replace('с', "c")
             .replace('А', "A").replace('а', "a")
             .replace('В', "B").replace('в', "b")
             .replace('Е', "E").replace('е', "e")
             .replace('Т', "T").replace('т', "t")
             .replace('О', "O").replace('о', "o")
             .replace('Р', "P").replace('р', "p")
             .replace('К', "K").replace('к', "k")
             .replace('Х', "X").replace('х', "x")
             .replace('М', "M").replace('м', "m")
             .replace('Н', "H").replace('н', "h")
    }

    pub fn new() -> Self {
        Self {
            cells: HashMap::new(),
            max_col: 0,
            max_row: 0,
            merged_regions: Vec::new(),
        }
    }

    pub fn insert(&mut self, col: u32, row: u32, content: String) {
        if content.trim().is_empty() { return; }
        let sanitized = Self::sanitize_latin(content);
        self.cells.insert((col, row), sanitized);
        if col >= self.max_col { self.max_col = col + 1; }
        if row >= self.max_row { self.max_row = row + 1; }
    }

    pub fn get(&self, col: u32, row: u32) -> Option<&String> {
        self.cells.get(&(col, row))
    }
    
    #[allow(dead_code)]
    pub fn add_merge(&mut self, start: (u32, u32), end: (u32, u32)) {
        self.merged_regions.push((start, end));
    }
    
    pub fn get_region_size(&self, col: u32, row: u32) -> u32 {
        for ((sc, sr), (ec, er)) in &self.merged_regions {
            if *sc == col && *sr == row {
                return (er - sr + 1) * (ec - sc + 1);
            }
        }
        1
    }

    /// Iterates through cells in Row-Major order (A1, B1, C1... A2, B2...)
    pub fn iter_execution_order(&self) -> Vec<((u32, u32), &String)> {
        let mut ordered_cells = Vec::new();
        for row in 0..self.max_row {
            for col in 0..self.max_col {
                if let Some(content) = self.get(col, row) {
                    ordered_cells.push(((col, row), content));
                }
            }
        }
        ordered_cells
    }
}

pub fn load_grid(path: &str) -> Result<CellGrid> {
    if path.to_lowercase().ends_with(".csv") {
        load_csv(path)
    } else {
        load_ods(path)
    }
}

fn load_csv(path: &str) -> Result<CellGrid> {
    let content = fs::read_to_string(path).context("Failed to read CSV file")?;
    let mut grid = CellGrid::new();
    
    let mut chars = content.chars().peekable();
    let mut in_quote = false;
    let mut current_cell = String::new();
    let mut col = 0;
    let mut row = 0;

    while let Some(c) = chars.next() {
        if in_quote {
            if c == '"' {
                if let Some(&next) = chars.peek() {
                    if next == '"' {
                        current_cell.push('"');
                        chars.next(); // skip escaped quote
                    } else {
                        in_quote = false;
                    }
                } else {
                    in_quote = false; // End of file
                }
            } else {
                current_cell.push(c);
            }
        } else {
            match c {
                '"' => in_quote = true,
                ',' => {
                    grid.insert(col, row, current_cell.trim().to_string());
                    current_cell.clear();
                    col += 1;
                },
                '\n' | '\r' => {
                     // Handle CRLF?
                     // If \r, peek \n.
                     // But split logic: If char is \r or \n.
                     // If we have content or previous comma?
                     // "a,b\n" -> a, b.
                     // "a,b" (no newline at end).
                     
                     // Push last cell if not empty or if comma preceded?
                     // My logic: comma pushes.
                     // End of line pushes last cell.
                     
                     // Problem: Multi-char newline.
                     if c == '\r' {
                         if let Some(&'\n') = chars.peek() {
                             chars.next();
                         }
                     }
                     
                     grid.insert(col, row, current_cell.trim().to_string());
                     current_cell.clear();
                     row += 1;
                     col = 0;
                },
                _ => current_cell.push(c),
            }
        }
    }
    // Final cell if no newline at EOF
    if !current_cell.trim().is_empty() || col > 0 { // If col>0 implies we had commas.
         grid.insert(col, row, current_cell.trim().to_string());
    }

    Ok(grid)
}

fn load_ods(path: &str) -> Result<CellGrid> {
    let mut workbook: Ods<std::io::BufReader<std::fs::File>> = open_workbook(path)
        .context("Failed to open ODS file")?;
     
    let sheet_names = workbook.sheet_names().to_owned();
    if sheet_names.is_empty() {
        anyhow::bail!("No sheets found in ODS file");
    }
    let first_sheet = sheet_names[0].clone();
    
    let mut grid = CellGrid::new();

    if let Ok(range) = workbook.worksheet_range(&first_sheet) {
        for (row_idx, row) in range.rows().enumerate() {
            for (col_idx, cell) in row.iter().enumerate() {
                 let content = match cell {
                    Data::String(s) => s.clone(),
                    Data::Float(f) => f.to_string(),
                    Data::Int(i) => i.to_string(),
                    Data::Bool(b) => b.to_string(),
                    Data::Empty => String::new(),
                    Data::DateTime(d) => d.to_string(), 
                    Data::Error(e) => format!("Error: {:?}", e),
                    Data::DateTimeIso(d) => d.clone(),
                    Data::DurationIso(d) => d.to_string(),
                 };
                 grid.insert(col_idx as u32, row_idx as u32, content);
            }
        }
    }

    Ok(grid)
}
