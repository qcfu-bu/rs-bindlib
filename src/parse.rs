use crate::ast0;
use ast0::*;
use pest::iterators::Pairs;
use pest::pratt_parser::PrattParser;

#[derive(pest_derive::Parser)]
#[grammar = "lam.pest"]
pub struct LamParser;

lazy_static::lazy_static! {
  static ref PRATT_PARSER: PrattParser<Rule> = {
    use pest::pratt_parser::{Assoc::*, Op};
    use Rule::*;
    PrattParser::new()
      .op(Op::infix(and, Left) | Op::infix(or, Left))
      .op(Op::infix(eq, Left) | Op::infix(neq, Left))
      .op(Op::infix(lte, Left) | Op::infix(gte, Left) | Op::infix(lt, Left) | Op::infix(gt, Left))
      .op(Op::infix(add, Left) | Op::infix(sub, Left))
      .op(Op::infix(mul, Left) | Op::infix(div, Left))
      .op(Op::prefix(neg) | Op::prefix(not))
      .op(Op::infix(app, Left))
  };
}

pub fn parse_term(pairs: Pairs<Rule>) -> Box<Term> {
    use self::Op1::*;
    use self::Op2::*;
    PRATT_PARSER
        .map_primary(|primary| match primary.as_rule() {
            Rule::bool => bool(primary.as_str().parse::<bool>().unwrap()),
            Rule::integer => int(primary.as_str().parse::<i32>().unwrap()),
            Rule::var => var(String::from(primary.as_str())),
            Rule::letin => {
                let outer = primary.into_inner().next().unwrap();
                match outer.as_rule() {
                    Rule::decl_rec => {
                        let mut inner = outer.into_inner();
                        let f = String::from(inner.next().unwrap().as_str());
                        let x = String::from(inner.next().unwrap().as_str());
                        let args = inner.next().unwrap().into_inner();
                        let mut body = parse_term(inner.next().unwrap().into_inner());
                        let m = parse_term(inner.next().unwrap().into_inner());
                        for arg in args.rev() {
                            body = fun(String::from(""), String::from(arg.as_str()), body)
                        }
                        letin(f.clone(), fun(f, x, body), m)
                    }
                    Rule::decl => {
                        let mut inner = outer.into_inner();
                        let x = String::from(inner.next().unwrap().as_str());
                        let args = inner.next().unwrap().into_inner();
                        let mut body = parse_term(inner.next().unwrap().into_inner());
                        let m = parse_term(inner.next().unwrap().into_inner());
                        for arg in args.rev() {
                            body = fun(String::from(""), String::from(arg.as_str()), body)
                        }
                        letin(x, body, m)
                    }
                    _ => panic!(),
                }
            }
            Rule::lambda => {
                let mut inner = primary.into_inner();
                let args = inner.next().unwrap().into_inner();
                let mut body = parse_term(inner.next().unwrap().into_inner());
                for arg in args.rev() {
                    body = fun(String::from(""), String::from(arg.as_str()), body)
                }
                body
            }
            Rule::ifte => {
                let mut inner = primary.into_inner();
                let cond = parse_term(inner.next().unwrap().into_inner());
                let m1 = parse_term(inner.next().unwrap().into_inner());
                let m2 = parse_term(inner.next().unwrap().into_inner());
                ifte(cond, m1, m2)
            }
            Rule::term => parse_term(primary.into_inner()),
            _ => panic!(),
        })
        .map_infix(|lhs, op, rhs| match op.as_rule() {
            Rule::add => op2(Add, lhs, rhs),
            Rule::sub => op2(Sub, lhs, rhs),
            Rule::mul => op2(Mul, lhs, rhs),
            Rule::div => op2(Div, lhs, rhs),
            Rule::lte => op2(Lte, lhs, rhs),
            Rule::gte => op2(Gte, lhs, rhs),
            Rule::lt => op2(Lt, lhs, rhs),
            Rule::gt => op2(Gt, lhs, rhs),
            Rule::eq => op2(Eq, lhs, rhs),
            Rule::neq => op2(Neq, lhs, rhs),
            Rule::and => op2(And, lhs, rhs),
            Rule::or => op2(Or, lhs, rhs),
            Rule::app => app(lhs, rhs),
            _ => panic!(),
        })
        .map_prefix(|op, rhs| match op.as_rule() {
            Rule::not => op1(Not, rhs),
            Rule::neg => op1(Neg, rhs),
            _ => panic!(),
        })
        .parse(pairs)
}
