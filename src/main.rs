mod ast0;
mod ast1;
mod eval;
mod parse;
mod trans01;

use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL_ALLOC: MiMalloc = MiMalloc;

use ahash::HashMap;
use parse::*;
use pest::Parser;
use std::fs;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let file = fs::read_to_string(&args[1]).expect("cannot read file");
    match LamParser::parse(Rule::prog, &file) {
        Ok(mut pairs) => {
            let tm = parse_term(pairs.next().unwrap().into_inner());
            let tm = trans01::trans(&mut HashMap::default(), tm.as_ref()).unbox();
            println!("result : {:#?}", tm);
            let val = eval::eval(tm);
            println!("result : {:#?}", val);
        }
        Err(e) => {
            eprintln!("Parse failed: {:?}", e)
        }
    }
}
