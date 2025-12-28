use anyhow::{Result, anyhow};
use nom::{
    branch::alt,
    bytes::complete::{tag, take_while1, take_until},
    character::complete::{alpha1, alphanumeric1, digit1, multispace0, char},
    combinator::{map, map_res, opt, recognize, value, verify},
    multi::{many0, separated_list0, separated_list1},
    sequence::{delimited, pair, preceded, tuple, terminated},
    IResult,
};

#[derive(Debug, PartialEq, Clone)]
pub enum Op {
    Add, Sub, Mul, Div, Power, 
    Eq, Neq, Lt, Gt,
    Rho, Iota, Ceil, Floor, Stile, Log, Circular, Radix,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Literal {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
    Nil,
}

#[derive(Debug, PartialEq, Clone)]
pub struct CellRef {
    pub col: u32,
    pub row: u32,
    pub col_abs: bool,
    pub row_abs: bool,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Expr {
    Literal(Literal),
    Reference(CellRef),
    RangeReference(CellRef, CellRef),
    Ident(String),
    BinaryOp(Box<Expr>, Op, Box<Expr>),
    Call(String, Vec<Expr>),
    Array(Vec<Expr>),
    Input, // New Input variant
    Generator(u32), // Column Generator
    If(Box<Expr>, Box<Block>, Option<Box<Block>>), // Condition, Then, Else
    Block(Block),
}

#[derive(Debug, PartialEq, Clone)]
pub struct Block {
    pub stmts: Vec<Stmt>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Stmt {
    Assign(String, Expr),
    Return(Expr),
    Yield(Expr),
    Expr(Expr),
    While(Expr, Block),
    For(String, Expr, Block), // var, iter_expr, body
    FnDef(String, Vec<String>, Block), // name, args, body
}

// --- Parsing Logic ---

fn ws<'a, F: 'a, O, E: nom::error::ParseError<&'a str>>(inner: F) -> impl FnMut(&'a str) -> IResult<&'a str, O, E>
where
    F: FnMut(&'a str) -> IResult<&'a str, O, E>,
{
    delimited(multispace0, inner, multispace0)
}

fn parse_ident(input: &str) -> IResult<&str, String> {
    let parser = recognize(pair(
        alt((alpha1, tag("_"))),
        many0(alt((alphanumeric1, tag("_"))))
    ));
    
    verify(parser, |s: &str| {
        let keywords = [
            "if", "else", "do", "end", "while", "for", "in", "fn", "return", "yield", 
            "true", "false", "nil", "input",
            "plus", "minus", "times", "div", "pow", "range", "ceil", "floor", "mod", "log", "trig", "sqrt"
        ];
        !keywords.contains(&s)
    })(input).map(|(i, s)| (i, s.to_string()))
}

fn parse_literal(input: &str) -> IResult<&str, Literal> {
    alt((
        map(tag("true"), |_| Literal::Bool(true)),
        map(tag("false"), |_| Literal::Bool(false)),
        map(tag("nil"), |_| Literal::Nil),
        map(digit1, |s: &str| Literal::Int(s.parse::<i64>().unwrap())),
        map(delimited(char('"'), take_while1(|c| c != '"'), char('"')), |s: &str| Literal::String(s.to_string())),
    ))(input)
}

fn parse_col_index(input: &str) -> IResult<&str, u32> {
    map_res(alpha1, |s: &str| {
        let mut col = 0;
        for c in s.chars() {
            if !c.is_ascii_uppercase() { return Err(anyhow!("Invalid column")); }
            col = col * 26 + (c as u32 - 'A' as u32) + 1;
        }
        Ok(col - 1)
    })(input)
}

fn parse_cell_ref(input: &str) -> IResult<&str, CellRef> {
    let (input, col_abs) = map(opt(char('$')), |o| o.is_some())(input)?;
    let (input, col) = parse_col_index(input)?;
    let (input, row_abs) = map(opt(char('$')), |o| o.is_some())(input)?;
    let (input, row) = map_res(digit1, |s: &str| s.parse::<u32>())(input)?;
    
    Ok((input, CellRef { col, row: row.saturating_sub(1), col_abs, row_abs }))
}

fn parse_block(input: &str) -> IResult<&str, Block> {
    let (input, _) = ws(tag("do"))(input)?;
    let (input, stmts) = many0(terminated(ws(parse_stmt), opt(char(';'))))(input)?;
    let (input, _) = ws(tag("end"))(input)?;
    Ok((input, Block { stmts }))
}

fn parse_if(input: &str) -> IResult<&str, Expr> {
    let (input, _) = tag("if")(input)?;
    let (input, cond) = ws(parse_expr)(input)?;
    let (input, _) = opt(ws(char(':')))(input)?;
    
    let (input, then_block) = alt((
        parse_block,
        map(parse_stmt, |s| Block { stmts: vec![s] })
    ))(input)?;
    
    let (input, else_block_res) = opt(preceded(
        ws(tag("else")),
        alt((
            parse_block,
            map(parse_stmt, |s| Block { stmts: vec![s] })
        ))
    ))(input)?;
    
    Ok((input, Expr::If(Box::new(cond), Box::new(then_block), else_block_res.map(Box::new))))
}

fn parse_atom(input: &str) -> IResult<&str, Expr> {
    alt((
        parse_if,
        map(parse_literal, Expr::Literal),
        map(terminated(tag("input"), opt(tuple((ws(char('(')), ws(char(')')))))), |_| Expr::Input), // Input keyword with optional ()
        delimited(char('('), ws(parse_expr), char(')')),
        map(tuple((parse_cell_ref, char(':'), parse_cell_ref)), |(start, _, end)| Expr::RangeReference(start, end)),
        // New: A:B syntax (Column Range) -> Treated as A1:B{MaxRow} internally? 
        // Or specific RangeReference variant. For now, let's keep it strictly parse_cell_ref based ranges for MVP
        // and handle logic in codegen or assume A:A is mapped to A1:A[MAX].
        // But parse_cell_ref requires digits.
        // We need a parse_col_ref logic.
        map(tuple((parse_col_index, char(':'), parse_col_index)), |(start_col, _, _)| {
            Expr::Generator(start_col)
        }),
        map(parse_cell_ref, Expr::Reference),
        map(parse_ident, Expr::Ident),
    ))(input)
}

fn parse_call_or_atom(input: &str) -> IResult<&str, Expr> {
    let parse_call = map(tuple((
        parse_ident,
        ws(char('(')),
        separated_list0(ws(char(',')), parse_expr),
        ws(char(')')),
    )), |(name, _, args, _)| Expr::Call(name, args));

    alt((
        parse_call,
        parse_atom,
    ))(input)
}

fn parse_factor(input: &str) -> IResult<&str, Expr> {
    let (input, lhs) = ws(parse_call_or_atom)(input)?;
    let (input, op) = opt(tuple((
        ws(alt((
            tag("times"), tag("div"), tag("pow"), 
            tag("reshape"), tag("range"), tag("ceil"), tag("floor"), 
            tag("mod"), tag("log"), tag("trig"), tag("sqrt"),
            tag("*"), tag("/")
        ))),
        parse_factor
    )))(input)?; 
    
    if let Some((op_str, rhs)) = op {
        let op = match op_str {
            "times" => Op::Mul,
            "div" => Op::Div,
            "pow" => Op::Power,
            "reshape" => Op::Rho,
            "range" => Op::Iota,
            "ceil" => Op::Ceil,
            "floor" => Op::Floor,
            "mod" => Op::Stile,
            "log" => Op::Log,
            "trig" => Op::Circular,
            "sqrt" => Op::Radix,
            "*" => Op::Power, // * is Power
            "/" => Op::Div,
            _ => unreachable!(),

        };
        Ok((input, Expr::BinaryOp(Box::new(lhs), op, Box::new(rhs))))
    } else {
        Ok((input, lhs))
    }
}


fn parse_term(input: &str) -> IResult<&str, Expr> {
    let (input, lhs) = parse_factor(input)?;
    let (input, op) = opt(tuple((
        ws(alt((
            tag("+"), tag("-"),
            tag("plus"), tag("minus")
        ))),
        parse_term
    )))(input)?;
    
    if let Some((op_var, rhs)) = op {
        // op_var is char? No, alt returns different types?
        // tag returns &str, char returns char.
        // alt requires same type.
        // Solution: Use tag for all.
        // And match string.
        let op = match op_var { 
            "+" | "plus" => Op::Add, 
            "-" | "minus" => Op::Sub, 
            _ => unreachable!() 
        };
        Ok((input, Expr::BinaryOp(Box::new(lhs), op, Box::new(rhs))))
    } else {
        Ok((input, lhs))
    }
}

fn parse_comparison(input: &str) -> IResult<&str, Expr> {
    let (input, lhs) = parse_term(input)?;
    let (input, op) = opt(tuple((
        ws(alt((
            tag("="), tag("!="), tag("<"), tag(">")
        ))),
        parse_comparison
    )))(input)?;
    
    if let Some((op_str, rhs)) = op {
        let op = match op_str {
            "=" => Op::Eq,
            "!=" => Op::Neq,
            "<" => Op::Lt,
            ">" => Op::Gt,
            _ => unreachable!(),
        };
        Ok((input, Expr::BinaryOp(Box::new(lhs), op, Box::new(rhs))))
    } else {
        Ok((input, lhs))
    }
}

fn parse_expr(input: &str) -> IResult<&str, Expr> {
    parse_comparison(input)
}

fn parse_stmt(input: &str) -> IResult<&str, Stmt> {
    alt((
        map(tuple((tag("while"), ws(parse_expr), ws(parse_block))), |(_, cond, body)| Stmt::While(cond, body)),
        
        map(tuple((
            tag("for"), ws(parse_ident), ws(tag("in")), ws(parse_expr), ws(parse_block)
        )), |(_, var, _, iter, body)| Stmt::For(var, iter, body)),
        
        map(tuple((
            tag("fn"), ws(parse_ident), 
            delimited(char('('), separated_list0(ws(char(',')), ws(parse_ident)), char(')')),
            ws(parse_block)
        )), |(_, name, args, body)| Stmt::FnDef(name, args, body)),
        
        map(tuple((tag("return"), ws(parse_expr))), |(_, e)| Stmt::Return(e)),
        map(tuple((tag("yield"), ws(parse_expr))), |(_, e)| Stmt::Yield(e)),
        map(tuple((tag("put"), ws(char('(')), take_until(")"), char(')'))), |(_, _, content, _): (&str, _, &str, _)| {
             // Try to parse content as expression list
             let mut parse_args = separated_list0(ws(char(',')), parse_expr);
             match parse_args(content) {
                 Ok((rem, args)) if rem.trim().is_empty() => Stmt::Expr(Expr::Call("put".to_string(), args)),
                 _ => {
                     // Fallback: treat as raw string
                     Stmt::Expr(Expr::Call("put".to_string(), vec![Expr::Literal(Literal::String(content.to_string()))]))
                 }
             }
        }),
        map(tuple((parse_ident, ws(char('=')), ws(parse_expr))), |(id, _, e)| Stmt::Assign(id, e)),
        map(parse_expr, Stmt::Expr),
    ))(input)
}

pub fn parse_cell_content(input: &str) -> Result<Vec<Stmt>> {
    let (input, stmts) = many0(terminated(ws(parse_stmt), opt(char(';'))))(input)
        .map_err(|e| anyhow!("Parse Error: {}", e))?;
        
    if !input.is_empty() {
        return Err(anyhow!("Unparsed input: {}", input));
    }
    
    Ok(stmts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_expr() {
        let input = "10";
        let res = parse_cell_content(input).unwrap();
        assert_eq!(res.len(), 1);
        match &res[0] {
            Stmt::Expr(Expr::Literal(Literal::Int(10))) => (),
            _ => panic!("Expected Expr(Lit(10))"),
        }
    }

    #[test]
    fn test_parse_ref() {
        let input = "A1";
        let res = parse_cell_content(input).unwrap();
        assert_eq!(res.len(), 1);
        match &res[0] {
            Stmt::Expr(Expr::Reference(r)) => {
                assert_eq!(r.col, 0);
                assert_eq!(r.row, 0); 
            },
            _ => panic!("Expected Expr(Ref)"),
        }
    }

    #[test]
    fn test_parse_assign() {
        let input = "x = A1 + 10";
        let res = parse_cell_content(input).unwrap();
        assert_eq!(res.len(), 1);
        match &res[0] {
            Stmt::Assign(id, _) => {
                assert_eq!(id, "x");
            },
            _ => panic!("Expected Assign"),
        }
    }

    #[test]
    fn test_parse_yield() {
        let input = "yield A1";
        let res = parse_cell_content(input).unwrap();
        match &res[0] {
            Stmt::Yield(_) => (),
            _ => panic!("Expected Yield"),
        }
    }

    #[test]
    fn test_parse_range() {
        let input = "A1:Z5";
        let res = parse_cell_content(input).unwrap();
        match &res[0] {
            Stmt::Expr(Expr::RangeReference(s, e)) => {
                assert_eq!(s.col, 0);
                assert_eq!(e.col, 25);
            },
            _ => panic!("Expected Range"),
        }
    }
    
    #[test]
    fn test_parse_loops() {
        let input = "for x in A1:A10 do put(x) end";
        let res = parse_cell_content(input).unwrap();
        match &res[0] {
            Stmt::For(var, _, _) => assert_eq!(var, "x"),
            _ => panic!("Expected For"),
        }
        
        let input_while = "while true do x = x + 1 end";
        let res = parse_cell_content(input_while).unwrap();
        match &res[0] {
            Stmt::While(_, _) => (),
            _ => panic!("Expected While"),
        }
    }
    
    #[test]
    fn test_parse_fn_def() {
        let input = "fn my_func(a, b) do return a + b end";
        let res = parse_cell_content(input).unwrap();
        assert_eq!(res.len(), 1);
        match &res[0] {
            Stmt::FnDef(name, args, _) => {
                assert_eq!(name, "my_func");
                assert_eq!(args, &vec!["a".to_string(), "b".to_string()]);
            },
            _ => panic!("Expected FnDef"),
        }
    }

    #[test]
    fn test_parse_if() {
        let input = "if true do return 1 else return 0 end";
        let res = parse_cell_content(input).unwrap();
         match &res[0] {
            Stmt::Expr(Expr::If(_, _, Some(_))) => (),
            _ => panic!("Expected If with Else"),
        }

        let input_no_else = "if false do return 1 end";
        let res_no_else = parse_cell_content(input_no_else).unwrap();
        match &res_no_else[0] {
            Stmt::Expr(Expr::If(_, _, None)) => (),
            _ => panic!("Expected If without Else"),
        }
    }


    #[test]
    fn test_parse_new_ops() {
        // Test Unicode multiplication and new ops
        let input = "x × y";
        let res = parse_cell_content(input).unwrap();
         match &res[0] {
            Stmt::Expr(Expr::BinaryOp(_, Op::Mul, _)) => (),
            _ => panic!("Expected Mul ×"),
        }
        
        // Power *
        let input_pow = "x * y";
        let res_pow = parse_cell_content(input_pow).unwrap();
         match &res_pow[0] {
            Stmt::Expr(Expr::BinaryOp(_, Op::Power, _)) => (),
            _ => panic!("Expected Power *"),
        }

        // Rho ⍴
        let input_rho = "3 ⍴ 1";
        let res_rho = parse_cell_content(input_rho).unwrap();
         match &res_rho[0] {
            Stmt::Expr(Expr::BinaryOp(_, Op::Rho, _)) => (),
            _ => panic!("Expected Rho ⍴"),
        }
    }
}
