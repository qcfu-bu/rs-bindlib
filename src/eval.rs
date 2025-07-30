use crate::ast1::*;
use std::rc::Rc;

pub fn eval(m0: Term) -> Term {
    use TermNode::*;
    match &*m0 {
        Int(_) | Bool(_) | Var(_) => m0.clone(),
        Fun(_) => m0,
        Op1(op1, m) => {
            let m = eval(m.clone());
            eval_op1(op1, m)
        }
        Op2(op2, m, n) => {
            let m = eval(m.clone());
            let n = eval(n.clone());
            eval_op2(op2, m, n)
        }
        App(m, n) => {
            let m0 = eval(m.clone());
            let n0 = eval(n.clone());
            if let Fun(bnd) = &*m0 {
                let expr = bnd.subst(vec![m0.clone(), n0]);
                return eval(expr);
            }
            panic!("eval_App({:?})", m0);
        }
        LetIn(m, bnd) => {
            let m = eval(m.clone());
            eval(bnd.subst(m))
        }
        Ifte(m, n1, n2) => {
            let m = eval(m.clone());
            if let Bool(b) = &*m {
                if *b {
                    return eval(n1.clone());
                } else {
                    return eval(n2.clone());
                }
            }
            panic!("eval_Ifte({:?})", m0);
        }
    }
}

fn eval_op1(op: &Op1, m: Term) -> Term {
    use self::Op1::*;
    use TermNode::*;
    match (op, &*m) {
        (Not, Bool(b)) => Rc::new(Bool(!b)),
        (Neg, Int(i)) => Rc::new(Int(-i)),
        (_, _) => panic!("eval_op1({:?}, {:?})", op, m),
    }
}

fn eval_op2(op: &Op2, m: Term, n: Term) -> Term {
    use self::Op2::*;
    use TermNode::*;
    match (op, &*m, &*n) {
        (Add, Int(i), Int(j)) => Rc::new(Int(i + j)),
        (Sub, Int(i), Int(j)) => Rc::new(Int(i - j)),
        (Mul, Int(i), Int(j)) => Rc::new(Int(i * j)),
        (Div, Int(i), Int(j)) => Rc::new(Int(i / j)),
        (Lte, Int(i), Int(j)) => Rc::new(Bool(i <= j)),
        (Gte, Int(i), Int(j)) => Rc::new(Bool(i >= j)),
        (Lt, Int(i), Int(j)) => Rc::new(Bool(i < j)),
        (Gt, Int(i), Int(j)) => Rc::new(Bool(i > j)),
        (Eq, Int(i), Int(j)) => Rc::new(Bool(i == j)),
        (Neq, Int(i), Int(j)) => Rc::new(Bool(i != j)),
        (And, Bool(i), Bool(j)) => Rc::new(Bool(*i && *j)),
        (Or, Bool(i), Bool(j)) => Rc::new(Bool(*i || *j)),
        (_, _, _) => panic!("eval_op2({:?}, {:?}, {:?})", op, m, n),
    }
}
