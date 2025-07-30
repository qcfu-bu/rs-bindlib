#[derive(Debug)]
pub enum Op1 {
    Neg,
    Not,
}

#[derive(Debug)]
pub enum Op2 {
    Add,
    Sub,
    Mul,
    Div,
    Lte,
    Gte,
    Lt,
    Gt,
    Eq,
    Neq,
    And,
    Or,
}

#[derive(Debug)]
pub enum Term {
    Int(i32),
    Bool(bool),
    Var(String),
    Op1(Op1, Box<Term>),
    Op2(Op2, Box<Term>, Box<Term>),
    Fun(String, String, Box<Term>),
    App(Box<Term>, Box<Term>),
    LetIn(String, Box<Term>, Box<Term>),
    Ifte(Box<Term>, Box<Term>, Box<Term>),
}

pub fn int(i: i32) -> Box<Term> {
    Box::new(Term::Int(i))
}

pub fn bool(b: bool) -> Box<Term> {
    Box::new(Term::Bool(b))
}

pub fn var(s: String) -> Box<Term> {
    Box::new(Term::Var(s))
}

pub fn op1(op: Op1, m: Box<Term>) -> Box<Term> {
    Box::new(Term::Op1(op, m))
}

pub fn op2(op: Op2, m: Box<Term>, n: Box<Term>) -> Box<Term> {
    Box::new(Term::Op2(op, m, n))
}

pub fn fun(s1: String, s2: String, m: Box<Term>) -> Box<Term> {
    Box::new(Term::Fun(s1, s2, m))
}

pub fn app(m: Box<Term>, n: Box<Term>) -> Box<Term> {
    Box::new(Term::App(m, n))
}

pub fn letin(s: String, m: Box<Term>, n: Box<Term>) -> Box<Term> {
    Box::new(Term::LetIn(s, m, n))
}

pub fn ifte(m: Box<Term>, n1: Box<Term>, n2: Box<Term>) -> Box<Term> {
    Box::new(Term::Ifte(m, n1, n2))
}
