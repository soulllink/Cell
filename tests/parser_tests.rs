use cell::parser::{
    parse_cell, parse_cell_content, CellContent, Literal, Stmt, Expr, Op, Direction
};

#[test]
fn test_parse_data_cell_int() {
    let res = parse_cell("42").unwrap();
    assert_eq!(res, CellContent::Data(Literal::Int(42)));
}

#[test]
fn test_parse_data_cell_string() {
    let res = parse_cell("\"hello\"").unwrap();
    assert_eq!(res, CellContent::Data(Literal::String("hello".to_string())));
}

#[test]
fn test_parse_code_cell_active() {
    let res = parse_cell("(Main! do put(42) end)").unwrap();
    match res {
        CellContent::Code(f) => {
            assert_eq!(f.name, "Main");
            assert!(f.is_active);
            assert_eq!(f.args.len(), 0);
            assert_eq!(f.body.stmts.len(), 1);
        },
        _ => panic!("Expected Code"),
    }
}

#[test]
fn test_parse_code_cell_lazy() {
    let res = parse_cell("(Helper do return 10 end)").unwrap();
    match res {
        CellContent::Code(f) => {
            assert_eq!(f.name, "Helper");
            assert!(f.args.is_empty());
            assert!(!f.is_active);
        },
        _ => panic!("Expected Code"),
    }
}

#[test]
fn test_parse_relative_ref() {
    let res = parse_cell_content("x = @left + @right").unwrap();
    match &res[0] {
        Stmt::Assign(name, Expr::BinaryOp(lhs, Op::Add, rhs)) => {
            assert_eq!(name, "x");
            assert_eq!(**lhs, Expr::RelativeRef(Direction::Left));
            assert_eq!(**rhs, Expr::RelativeRef(Direction::Right));
        },
        _ => panic!("Expected assign with relative refs"),
    }
}

#[test]
fn test_parse_relative_top_bottom() {
    let res = parse_cell_content("y = @top - @bottom").unwrap();
    match &res[0] {
        Stmt::Assign(name, Expr::BinaryOp(lhs, Op::Sub, rhs)) => {
            assert_eq!(name, "y");
            assert_eq!(**lhs, Expr::RelativeRef(Direction::Top));
            assert_eq!(**rhs, Expr::RelativeRef(Direction::Bottom));
        },
        _ => panic!("Expected assign with @top, @bottom"),
    }
}

#[test]
fn test_parse_if_with_relative() {
    let res = parse_cell_content("if @left > 0: return @right").unwrap();
    match &res[0] {
        Stmt::Expr(Expr::If(cond, then_block, None)) => {
            match cond.as_ref() {
                Expr::BinaryOp(l, Op::Gt, _) => {
                    assert_eq!(**l, Expr::RelativeRef(Direction::Left));
                },
                _ => panic!("Expected comparison"),
            }
            assert_eq!(then_block.stmts.len(), 1);
        },
        _ => panic!("Expected If"),
    }
}

#[test]
fn test_parse_func_args() {
    let input = "(MyFunc a b c do Flip a end)";
    let cell = parse_cell(input).unwrap();
    if let CellContent::Code(func) = cell {
        assert_eq!(func.name, "MyFunc");
        assert_eq!(func.args, vec!["a", "b", "c"]);
    } else {
        panic!("Parsed as data cell");
    }
}

#[test]
fn test_parse_fizzbuzz_while() {
    let input = "
      i = 1
      while i < 16 do
        put(i)
        i = i + 1
      end
    ";
    let res = parse_cell_content(input);
    assert!(res.is_ok(), "Failed to parse while loop: {:?}", res.err());
}
