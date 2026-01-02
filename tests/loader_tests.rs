use cell::loader::CellGrid;

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

    // Expect: A1, B1, A2, B2 (Row 0 then Row 1 - Row-Major order)
    assert_eq!(values, vec!["A1", "B1", "A2", "B2"]);
}

#[test]
fn test_sparse_grid() {
    let mut grid = CellGrid::new();
    grid.insert(0, 0, "A1".to_string());
    grid.insert(2, 2, "C3".to_string()); // Skip B and rows 0,1 of C

    let order = grid.iter_execution_order();
    let values: Vec<&String> = order.iter().map(|(_, val)| *val).collect();
    // A1 (0,0) -> ... (0, max_row) -> ... (1, ...) -> (2, 0).. (2,2)
    // Since we iterate 0..max_col and 0..max_row.
    // max_col=3 (index 0,1,2), max_row=3 (index 0,1,2).
    
    // Iteration: 
    // Col 0: (0,0)=A1, (0,1)=None, (0,2)=None
    // Col 1: (1,0)=None ...
    // Col 2: ... (2,2)=C3
    
    assert_eq!(values, vec!["A1", "C3"]);
}
