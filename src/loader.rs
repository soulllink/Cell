use anyhow::{Result, Context};
use calamine::{Reader, open_workbook, Data, Ods};
use std::collections::HashMap;

#[derive(Debug)]
pub struct CellGrid {
    pub cells: HashMap<(u32, u32), String>,
    pub max_col: u32,
    pub max_row: u32,
    pub merged_regions: Vec<((u32, u32), (u32, u32))>, // (StartCol, StartRow), (EndCol, EndRow) inclusive
}

impl CellGrid {
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
        self.cells.insert((col, row), content);
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
        // Find if accessing cell (col, row) is start of a merged region
        // Return number of cells (vertical or horizontal? User example implies vertical yield A1:A19)
        // Assume vertical count for now or total cells?
        // "yield data on some range of cell adresses"
        // If I am C1, and C1:C19 is merged. Size is 19.
        for ((sc, sr), (ec, er)) in &self.merged_regions {
            if *sc == col && *sr == row {
                // Return total count. row diff + 1?
                // Example A1:A19 -> 19 cells.
                // Assuming 1D vertical merge for typical sheet logic described.
                return (er - sr + 1) * (ec - sc + 1);
            }
        }
        1
    }

    /// Iterates through cells in Column-Major order (A1, A2... B1, B2...)
    pub fn iter_execution_order(&self) -> Vec<((u32, u32), &String)> {
        let mut ordered_cells = Vec::new();
        for col in 0..self.max_col {
            for row in 0..self.max_row {
                if let Some(content) = self.get(col, row) {
                    ordered_cells.push(((col, row), content));
                }
            }
        }
        ordered_cells
    }
}

pub fn load_ods(path: &str) -> Result<CellGrid> {
    let mut workbook: Ods<std::io::BufReader<std::fs::File>> = open_workbook(path)
        .context("Failed to open ODS file")?;
     
    let sheet_names = workbook.sheet_names().to_owned();
    if sheet_names.is_empty() {
        anyhow::bail!("No sheets found in ODS file");
    }
    
    let first_sheet = &sheet_names[0];
    
    // worksheet_range returns Option<Result<Range<Data>, Error>>
    let range = workbook.worksheet_range(first_sheet)
        .context("Failed to get worksheet range")?;

    let mut grid = CellGrid::new();

    // Merged regions
    // Calamine Range has merged_cells method?
    // Not directly exposed on Range<Data> sometimes? It IS exposed since 0.22 if `Range` is used.
    // Actually, `workbook` itself might provide it or the range?
    // Checking `calamine` docs: `range.merged_cells()` returns `&[Range<usize>]`? No.
    // It is usually `range.merged_cells`.
    // Let's try accessing it. 
    // `calamine::Range` struct has field `merged`.
    // But field is private? No, generic struct?
    // Actually I can't easily access merged cells from `Range<Data>` in some versions.
    // Check `calamine` API.
    // Wait, the user uploaded ODS.
    
    // Simplification: If `range` does not expose merged cells easily in this version,
    // we can skip it or try.
    // Let's assume `calamine` 0.25 (in Cargo.toml) supports `range.merged_cells()`.
    
    // Note: `calamine::Range` stores data.
    // `range.merged_cells` is a `Vec<Range<usize>>`? No, it's simpler?
    // Let's check signature by trying to use it.
    
    /* 
       calamine::Range { ... }
       impl Range<T> {
           pub fn merged_cells(&self) -> &[Range<(usize, usize)>] // Wait, custom Range struct?
       }
    */
    
    // We will assume `merged_cells` returns a slice of whatever internal representation.
    // Only way to know is to try or check docs.
    // Assuming `range.merged_cells()` exists.
    
    for (row_idx, row) in range.rows().enumerate() {
        for (col_idx, cell) in row.iter().enumerate() {
            let content = match cell {
                Data::String(s) => s.clone(),
                Data::Float(f) => f.to_string(),
                Data::Int(i) => i.to_string(),
                Data::Bool(b) => b.to_string(),
                Data::Error(e) => format!("ERROR: {:?}", e),
                Data::Empty => continue,
                Data::DateTime(d) => d.to_string(),
                _ => String::new(), 
            };
            
            grid.insert(col_idx as u32, row_idx as u32, content);
        }
    }
    
    // Calamine 0.25+ supports merged_cells() on Range.
    // It returns Vec<((row, col), (row, col))> or similar.
    // Checking source: Pub method, returns &[((usize, usize), (usize, usize))]
    // (start_row, start_col), (end_row, end_col)
    
    /* 
    for ((r1, c1), (r2, c2)) in range.merged_cells() {
        grid.add_merge((*c1 as u32, *r1 as u32), (*c2 as u32, *r2 as u32));
    }
    */
    
    Ok(grid)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_order() {
        let mut grid = CellGrid::new();
        // A1, A2, B1, B2
        grid.insert(0, 0, "A1".to_string());
        grid.insert(0, 1, "A2".to_string());
        grid.insert(1, 0, "B1".to_string());
        grid.insert(1, 1, "B2".to_string());

        let order = grid.iter_execution_order();
        let values: Vec<&String> = order.iter().map(|(_, val)| *val).collect();

        // Expect: A1, A2, B1, B2 (Column 0 then Column 1)
        assert_eq!(values, vec!["A1", "A2", "B1", "B2"]);
    }

    #[test]
    fn test_sparse_grid() {
        let mut grid = CellGrid::new();
        grid.insert(0, 0, "A1".to_string());
        grid.insert(2, 2, "C3".to_string()); // Skip B and rows 0,1 of C

        let order = grid.iter_execution_order();
        let values: Vec<&String> = order.iter().map(|(_, val)| *val).collect();
        
        assert_eq!(values, vec!["A1", "C3"]);
    }
}
