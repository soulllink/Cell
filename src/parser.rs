use anyhow::{Result, anyhow};
use nom::{
    branch::alt,
    bytes::complete::{tag, take_while1},
    character::complete::{alpha1, alphanumeric1, digit1, multispace0, char},
    combinator::{map, map_res, opt, recognize, value, verify},
    multi::{many0, separated_list0},
    sequence::{delimited, pair, preceded, tuple, terminated},
    IResult,
};

// ============================================================================
// AST Definitions
// ============================================================================

#[derive(Debug, PartialEq, Clone)]
pub enum Op {
    // Arithmetic
    Add, Sub, Mul, Div, Power, 
    // Comparison
    Eq, Neq, Lt, Gt, Le, Ge,
    // Logical
    And, Or, Match,
    // List/Array
    Right, Concat, Take, Drop, Pad, Find, Apply,
    // Other
    Mod,
}

#[derive(Debug, PartialEq, Clone)]
pub enum UnaryOp {
    Identity, Flip, Negate, First, Sqrt, Odometer, Where, Reverse, 
    GradeUp, GradeDown, Group, Not, Enlist, Null, Length, Floor, 
    String, Unique, Type, Eval,
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

/// Relative addressing: @left, @right, @top, @bottom
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Direction {
    Left,
    Right,
    Top,
    Bottom,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Expr {
    Literal(Literal),
    Reference(CellRef),
    RelativeRef(Direction),
    RangeReference(CellRef, CellRef),
    Ident(String),
    UnaryOp(UnaryOp, Box<Expr>),
    BinaryOp(Box<Expr>, Op, Box<Expr>),
    Call(String, Vec<Expr>),
    Array(Vec<Expr>),
    Input,
    Generator(u32),
    If(Box<Expr>, Box<Block>, Option<Box<Block>>),
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
    For(String, Expr, Block),
    FnDef(String, Vec<String>, Block),
}

// ============================================================================
// New Cell-Level AST
// ============================================================================

/// Represents a parsed cell's content.
#[derive(Debug, PartialEq, Clone)]
pub enum CellContent {
    /// Data cell: just a value (number, string, etc.)
    Data(Literal),
    /// Code cell: a named function block
    Code(FunctionDef),
}

/// A function definition within a cell: `(Name do ... end)`
#[derive(Debug, PartialEq, Clone)]
pub struct FunctionDef {
    pub name: String,
    pub args: Vec<String>,
    pub is_active: bool, // true if name ends with `!`
    pub body: Block,
}

// ============================================================================
// Parsing Helpers
// ============================================================================

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
        ];
        !keywords.contains(&s)
    })(input).map(|(i, s)| (i, s.to_string()))
}

/// Parse a function name, possibly ending with `!` for self-invocation
fn parse_func_name(input: &str) -> IResult<&str, (String, bool)> {
    let (input, name) = parse_ident(input)?;
    let (input, bang) = opt(char('!'))(input)?;
    Ok((input, (name, bang.is_some())))
}

fn parse_literal(input: &str) -> IResult<&str, Literal> {
    alt((
        map(tag("true"), |_| Literal::Bool(true)),
        map(tag("false"), |_| Literal::Bool(false)),
        map(tag("nil"), |_| Literal::Nil),
        // Float with decimal
        map_res(
            recognize(tuple((opt(char('-')), digit1, char('.'), digit1))),
            |s: &str| s.parse::<f64>().map(Literal::Float)
        ),
        // Integer
        map_res(
            recognize(tuple((opt(char('-')), digit1))),
            |s: &str| s.parse::<i64>().map(Literal::Int)
        ),
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

/// Parse relative reference: @left, @right, @top, @bottom
fn parse_relative_ref(input: &str) -> IResult<&str, Direction> {
    preceded(
        char('@'),
        alt((
            value(Direction::Left, tag("left")),
            value(Direction::Right, tag("right")),
            value(Direction::Top, tag("top")),
            value(Direction::Bottom, tag("bottom")),
        ))
    )(input)
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
        // Relative reference (@left, @right, @top, @bottom)
        map(parse_relative_ref, Expr::RelativeRef),
        map(parse_literal, Expr::Literal),
        map(terminated(tag("input"), opt(tuple((ws(char('(')), ws(char(')')))))), |_| Expr::Input),
        delimited(char('('), ws(parse_expr), char(')')),
        map(tuple((parse_cell_ref, char(':'), parse_cell_ref)), |(start, _, end)| Expr::RangeReference(start, end)),
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
    let (input, lhs) = ws(parse_unary)(input)?;
    let (input, op) = opt(tuple((
        ws(alt((
            tag("×"), tag("*"), tag("/")
        ))),
        parse_factor
    )))(input)?; 
    
    if let Some((op_str, rhs)) = op {
        let op = match op_str {
            "×" => Op::Mul,
            "*" => Op::Power,
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
        ws(alt((tag("+"), tag("-")))),
        parse_term
    )))(input)?;
    
    if let Some((op_var, rhs)) = op {
        let op = match op_var { 
            "+" => Op::Add, 
            "-" => Op::Sub, 
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
            tag("!="), tag("="), tag("<="), tag(">="), tag("<"), tag(">"),
            // Add other comparison-like ops if needed, e.g. Match "Match"
            tag("Match"), tag("And"), tag("Or"), tag("Mod"),
            tag("Right"), tag("Concat"), tag("Take"), tag("Drop"), tag("Pad"), tag("Find"), tag("Apply")
        ))),
        parse_comparison
    )))(input)?;
    
    if let Some((op_str, rhs)) = op {
        let op = match op_str {
            "=" => Op::Eq,
            "!=" => Op::Neq,
            "<" => Op::Lt,
            ">" => Op::Gt,
            "<=" => Op::Le,
            ">=" => Op::Ge,
            "Match" => Op::Match,
            "And" => Op::And,
            "Or" => Op::Or,
            "Mod" => Op::Mod,
            "Right" => Op::Right,
            "Concat" => Op::Concat,
            "Take" => Op::Take,
            "Drop" => Op::Drop,
            "Pad" => Op::Pad,
            "Find" => Op::Find,
            "Apply" => Op::Apply,
            _ => unreachable!(),
        };
        Ok((input, Expr::BinaryOp(Box::new(lhs), op, Box::new(rhs))))
    } else {
        Ok((input, lhs))
    }
}

fn parse_unary(input: &str) -> IResult<&str, Expr> {
    let (input, op_str) = opt(ws(alt((
        alt((
            tag("Self"), tag("Flip"), tag("Negate"), tag("First"), tag("Sqrt"), 
            tag("Enum"), tag("Odometer"), tag("Where"), tag("Reverse"), 
            tag("Ascend"), tag("Open"), tag("Descend"), tag("Close"), 
            tag("Group"), tag("Unitmat"), tag("Not"), tag("Enlist"), 
            tag("Null"), tag("Length"), tag("Size"), tag("Floor")
        )),
        alt((
            tag("Lowercase"), tag("String"), tag("Uniq"), tag("Type"), 
            tag("Get"), tag("Eval"), tag("Values")
        ))
    ))))(input)?;

    if let Some(op_str) = op_str {
        let op = match op_str {
            "Self" => UnaryOp::Identity,
            "Flip" => UnaryOp::Flip,
            "Negate" => UnaryOp::Negate,
            "First" => UnaryOp::First,
            "Sqrt" => UnaryOp::Sqrt,
            "Enum" | "Odometer" => UnaryOp::Odometer,
            "Where" => UnaryOp::Where,
            "Reverse" => UnaryOp::Reverse,
            "Ascend" | "Open" => UnaryOp::GradeUp,
            "Descend" | "Close" => UnaryOp::GradeDown,
            "Group" | "Unitmat" => UnaryOp::Group,
            "Not" => UnaryOp::Not,
            "Enlist" => UnaryOp::Enlist,
            "Null" => UnaryOp::Null,
            "Length" | "Size" => UnaryOp::Length,
            "Floor" | "Lowercase" => UnaryOp::Floor,
            "String" => UnaryOp::String,
            "Uniq" => UnaryOp::Unique,
            "Type" => UnaryOp::Type,
            "Get" | "Eval" | "Values" => UnaryOp::Eval,
            _ => unreachable!(),
        };
        let (input, expr) = parse_unary(input)?; // Right-associative or recursive
        Ok((input, Expr::UnaryOp(op, Box::new(expr))))
    } else {
        // Fallback to atoms/calls
        parse_call_or_atom(input)
    }
}

fn parse_expr(input: &str) -> IResult<&str, Expr> {
    // Top level expression parsing
    // Hierarchy: 
    // Comparison (Binary) -> Term (+/-) -> Factor (*//) -> Unary -> Call/Atom
    // Wait, typical precedence order:
    // Comparison (lowest)
    // Add/Sub
    // Mul/Div
    // Unary (highest bind)
    
    // In current implementation:
    // parse_comparison calls parse_term
    // parse_term calls parse_factor
    // parse_factor CURRENTLY calls parse_call_or_atom directly
    // WE NEED TO INSERT parse_unary IN BETWEEN factor and atom
    
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
        // put() as a statement
        map(tuple((
            tag("put"),
            delimited(
                ws(char('(')),
                separated_list0(ws(char(',')), parse_expr),
                ws(char(')'))
            )
        )), |(_, args)| Stmt::Expr(Expr::Call("put".to_string(), args))),
        map(tuple((parse_ident, ws(char('=')), ws(parse_expr))), |(id, _, e)| Stmt::Assign(id, e)),
        map(parse_expr, Stmt::Expr),
    ))(input)
}

// ============================================================================
// Top-Level Cell Parsing (NEW SYNTAX)
// ============================================================================

/// Parse the new Cell syntax: `(FuncName do ... end)` or plain data
pub fn parse_cell(input: &str) -> Result<CellContent> {
    let trimmed = input.trim();
    
    // Check if it's a function definition: starts with `(` and ends with `)`
    if trimmed.starts_with('(') && trimmed.ends_with(')') {
        // Extract inner content
        let inner = &trimmed[1..trimmed.len()-1].trim();
        
        // Parse: Name[!] [args...] do ... end
        let (remaining, (name, is_active)) = parse_func_name(inner)
            .map_err(|e| anyhow!("Failed to parse function name: {}", e))?;
        
        let (remaining, args) = many0(ws(parse_ident))(remaining)
             .map_err(|e| anyhow!("Failed to parse function args: {}", e))?;
            
        let (remaining, body) = parse_block(remaining.trim())
            .map_err(|e| anyhow!("Failed to parse function body: {}", e))?;
        
        if !remaining.trim().is_empty() {
            return Err(anyhow!("Unparsed content in function: {}", remaining));
        }
        
        Ok(CellContent::Code(FunctionDef { name, args, is_active, body }))
    } else {
        // Try to parse as a literal (data cell)
        match parse_literal(trimmed) {
            Ok((remaining, lit)) if remaining.trim().is_empty() => {
                Ok(CellContent::Data(lit))
            },
            _ => {
                // Fallback: treat as string data
                Ok(CellContent::Data(Literal::String(trimmed.to_string())))
            }
        }
    }
}

// ============================================================================
// Legacy API (for backward compatibility during transition)
// ============================================================================

pub fn parse_cell_content(input: &str) -> Result<Vec<Stmt>> {
    let (input, stmts) = many0(terminated(ws(parse_stmt), opt(char(';'))))(input)
        .map_err(|e| anyhow!("Parse Error: {}", e))?;
        
    if !input.is_empty() {
        return Err(anyhow!("Unparsed input: {}", input));
    }
    
    Ok(stmts)
}

// ============================================================================
// Tests
// ============================================================================



