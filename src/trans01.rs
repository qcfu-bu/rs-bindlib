use crate::{ast0, ast1};
use ahash::HashMap;
use rs_bindlib::*;

pub type Ctx = HashMap<String, Var<ast1::Term>>;

pub fn trans(ctx: &mut Ctx, m: &ast0::Term) -> Boxed<ast1::Term> {
    use ast0::Term::*;
    match m {
        Int(i) => ast1::int(*i),
        Bool(b) => ast1::bool(*b),
        Var(s) => {
            let x = ctx.get(s).unwrap();
            ast1::var(x.clone())
        }
        Op1(op1, m) => {
            let op1 = trans_op1(op1);
            let m = trans(ctx, m);
            ast1::op1(op1, m)
        }
        Op2(op2, m, n) => {
            let op2 = trans_op2(op2);
            let m = trans(ctx, m);
            let n = trans(ctx, n);
            ast1::op2(op2, m, n)
        }
        Fun(f0, x0, m) => {
            let mut local = ctx.clone();
            let f = ast1::new_var(f0.clone());
            let x = ast1::new_var(x0.clone());
            local.insert(f0.clone(), f.clone());
            local.insert(x0.clone(), x.clone());
            let m = trans(&mut local, m);
            let bnd = MBinder::bind_mvar(vec![f, x], m);
            ast1::fun(bnd)
        }
        App(m, n) => {
            let m = trans(ctx, m);
            let n = trans(ctx, n);
            ast1::app(m, n)
        }
        LetIn(x0, m, n) => {
            let m = trans(ctx, m);
            let mut local = ctx.clone();
            let x = ast1::new_var(x0.clone());
            local.insert(x0.clone(), x.clone());
            let n = trans(&mut local, n);
            let bnd = Binder::bind_var(x, n);
            ast1::letin(m, bnd)
        }
        Ifte(m, n1, n2) => {
            let m = trans(ctx, m);
            let n1 = trans(ctx, n1);
            let n2 = trans(ctx, n2);
            ast1::ifte(m, n1, n2)
        }
    }
}

fn trans_op1(op1: &ast0::Op1) -> ast1::Op1 {
    use ast0::Op1::*;
    match op1 {
        Neg => ast1::Op1::Neg,
        Not => ast1::Op1::Not,
    }
}

fn trans_op2(op2: &ast0::Op2) -> ast1::Op2 {
    use ast0::Op2::*;
    match op2 {
        Add => ast1::Op2::Add,
        Sub => ast1::Op2::Sub,
        Mul => ast1::Op2::Mul,
        Div => ast1::Op2::Div,
        Lte => ast1::Op2::Lte,
        Gte => ast1::Op2::Gte,
        Lt => ast1::Op2::Lt,
        Gt => ast1::Op2::Gt,
        Eq => ast1::Op2::Eq,
        Neq => ast1::Op2::Neq,
        And => ast1::Op2::And,
        Or => ast1::Op2::Or,
    }
}
