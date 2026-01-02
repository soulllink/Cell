
#[test]
fn test_parse_func_args() {
    let input = "(MyFunc a b c do Flip a end)";
    let (rem, cell) = parse_cell(input).unwrap();
    if let CellContent::Code(func) = cell {
        assert_eq!(func.name, "MyFunc");
        assert_eq!(func.args, vec!["a", "b", "c"]);
    } else {
        panic!("Parsed as data cell");
    }
}
