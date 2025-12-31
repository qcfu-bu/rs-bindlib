use rs_bindlib::*;
use std::{any::Any, rc::Rc};

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

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct Term(pub Rc<TermNode>);

impl Into<Rc<dyn Any>> for Term {
    fn into(self) -> Rc<dyn Any> {
        self.0
    }
}

impl From<Rc<dyn Any>> for Term {
    fn from(r: Rc<dyn Any>) -> Self {
        Term(r.downcast::<TermNode>().unwrap())
    }
}

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
    Var::new(|x| Term(Rc::new(TermNode::Var(x))), name)
}

pub fn int(i: i32) -> Boxed<Term> {
    boxed(Term(Rc::new(TermNode::Int(i))))
}

pub fn bool(b: bool) -> Boxed<Term> {
    boxed(Term(Rc::new(TermNode::Bool(b))))
}

pub fn var(x: Var<Term>) -> Boxed<Term> {
    x.into()
}

pub fn op1(op: Op1, m: Boxed<Term>) -> Boxed<Term> {
    apply1(move |m| Term(Rc::new(TermNode::Op1(op, m))), m)
}

pub fn op2(op: Op2, m: Boxed<Term>, n: Boxed<Term>) -> Boxed<Term> {
    apply2(move |m, n| Term(Rc::new(TermNode::Op2(op, m, n))), m, n)
}

pub fn fun(bnd: Boxed<MBinder<Term, Term>>) -> Boxed<Term> {
    apply1(move |bnd| Term(Rc::new(TermNode::Fun(bnd))), bnd)
}

pub fn app(m: Boxed<Term>, n: Boxed<Term>) -> Boxed<Term> {
    apply2(move |m, n| Term(Rc::new(TermNode::App(m, n))), m, n)
}

pub fn letin(m: Boxed<Term>, n: Boxed<Binder<Term, Term>>) -> Boxed<Term> {
    apply2(move |m, n| Term(Rc::new(TermNode::LetIn(m, n))), m, n)
}

pub fn ifte(m: Boxed<Term>, n1: Boxed<Term>, n2: Boxed<Term>) -> Boxed<Term> {
    apply3(
        move |m, n1, n2| Term(Rc::new(TermNode::Ifte(m, n1, n2))),
        m,
        n1,
        n2,
    )
}

impl Into<Boxed<Term>> for &Term {
    fn into(self) -> Boxed<Term> {
        match &*self.0 {
            TermNode::Int(i) => int(*i),
            TermNode::Bool(i) => bool(*i),
            TermNode::Var(x) => x.clone().into(),
            TermNode::Op1(op, m) => op1(*op, m.into()),
            TermNode::Op2(op, m, n) => op2(*op, m.into(), n.into()),
            TermNode::Fun(bnd) => fun(bnd.clone().compose(|m| (&m).into()).into()),
            TermNode::App(m, n) => app(m.into(), n.into()),
            TermNode::LetIn(m, bnd) => letin(m.into(), bnd.clone().compose(|m| (&m).into()).into()),
            TermNode::Ifte(m, n1, n2) => ifte(m.into(), n1.into(), n2.into()),
        }
    }
}
