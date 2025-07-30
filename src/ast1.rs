use rs_bindlib::*;
use std::rc::Rc;

#[derive(Debug, Clone, Copy)]
pub enum Op1 {
    Neg,
    Not,
}

#[derive(Debug, Clone, Copy)]
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

pub type Term = Rc<TermNode>;

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub enum TermNode {
    Int(i32),
    Bool(bool),
    Var(Var<Term>),
    Op1(Op1, Term),
    Op2(Op2, Term, Term),
    Fun(MBinder<Term, Term>),
    App(Term, Term),
    LetIn(Term, Binder<Term, Term>),
    Ifte(Term, Term, Term),
}

pub fn new_var(name: String) -> Var<Term> {
    Var::new(|x| Rc::new(TermNode::Var(x)), name)
}

pub fn int(i: i32) -> Boxed<Term> {
    boxed(Rc::new(TermNode::Int(i)))
}

pub fn bool(b: bool) -> Boxed<Term> {
    boxed(Rc::new(TermNode::Bool(b)))
}

pub fn var(x: Var<Term>) -> Boxed<Term> {
    x.into_box()
}

pub fn op1(op: Op1, m: Boxed<Term>) -> Boxed<Term> {
    apply1(move |m| Rc::new(TermNode::Op1(op, m)), m)
}

pub fn op2(op: Op2, m: Boxed<Term>, n: Boxed<Term>) -> Boxed<Term> {
    apply2(move |m, n| Rc::new(TermNode::Op2(op, m, n)), m, n)
}

pub fn fun(bnd: Boxed<MBinder<Term, Term>>) -> Boxed<Term> {
    apply1(move |bnd| Rc::new(TermNode::Fun(bnd)), bnd)
}

pub fn app(m: Boxed<Term>, n: Boxed<Term>) -> Boxed<Term> {
    apply2(move |m, n| Rc::new(TermNode::App(m, n)), m, n)
}

pub fn letin(m: Boxed<Term>, n: Boxed<Binder<Term, Term>>) -> Boxed<Term> {
    apply2(move |m, n| Rc::new(TermNode::LetIn(m, n)), m, n)
}

pub fn ifte(m: Boxed<Term>, n1: Boxed<Term>, n2: Boxed<Term>) -> Boxed<Term> {
    apply3(
        move |m, n1, n2| Rc::new(TermNode::Ifte(m, n1, n2)),
        m,
        n1,
        n2,
    )
}

impl IntoBoxed<Term> for &TermNode {
    fn into_box(self) -> Boxed<Term> {
        match self {
            TermNode::Int(i) => int(*i),
            TermNode::Bool(i) => bool(*i),
            TermNode::Var(x) => x.clone().into_box(),
            TermNode::Op1(op, m) => op1(*op, m.into_box()),
            TermNode::Op2(op, m, n) => op2(*op, m.into_box(), n.into_box()),
            TermNode::Fun(bnd) => fun(bnd.clone().compose(|m| m.into_box()).into_box()),
            TermNode::App(m, n) => app(m.into_box(), n.into_box()),
            TermNode::LetIn(m, bnd) => letin(
                m.into_box(),
                bnd.clone().compose(|m| m.into_box()).into_box(),
            ),
            TermNode::Ifte(m, n1, n2) => ifte(m.into_box(), n1.into_box(), n2.into_box()),
        }
    }
}
