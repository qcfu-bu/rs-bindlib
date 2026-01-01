use ahash::HashMap;
use itertools::{Itertools, merge};
use std::{
    any::Any,
    cell::RefCell,
    cmp,
    fmt::Debug,
    rc::{Rc, Weak},
    sync::atomic::{AtomicUsize, Ordering::Relaxed},
};

static STAMP: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone)]
struct Env {
    tab: RefCell<Vec<Option<Rc<dyn Any>>>>,
}

impl Env {
    fn new(size: usize) -> Self {
        Self {
            tab: RefCell::new(vec![None; size]),
        }
    }

    fn set<A: Into<Rc<dyn Any>> + 'static>(&self, i: usize, a: A) {
        let any: Rc<dyn Any> = a.into();
        self.tab.borrow_mut()[i] = Some(any);
    }

    fn get<A: From<Rc<dyn Any>> + 'static>(&self, i: usize) -> A {
        let any: Rc<dyn Any> = self.tab.borrow()[i].clone().unwrap();
        A::from(any)
    }

    fn clone_from(&self, src: &Env, len: usize) {
        let mut dst_tab = self.tab.borrow_mut();
        let src_tab = src.tab.borrow();
        dst_tab[..len].clone_from_slice(&src_tab[..len]);
    }
}

#[derive(Clone)]
struct Pos {
    inner: Rc<RefCell<HashMap<usize, usize>>>,
}

impl Pos {
    fn new() -> Self {
        Self {
            inner: Rc::new(RefCell::new(HashMap::default())),
        }
    }
}

#[derive(Clone)]
struct Clo<T> {
    inner: Rc<dyn Fn(&Pos, &Env) -> T>,
}

impl<T> Clo<T> {
    fn new<F>(f: F) -> Self
    where
        F: Fn(&Pos, &Env) -> T + 'static,
    {
        Self { inner: Rc::new(f) }
    }

    fn map<B, F>(self, f: F) -> Clo<B>
    where
        T: Clone + 'static,
        B: Clone + 'static,
        F: Fn(T) -> B + 'static,
    {
        Clo::new(move |vs, env| f((self.inner)(vs, env)))
    }

    fn app<A, B>(self: Self, a: A) -> Clo<B>
    where
        A: Clone + 'static,
        B: Clone + 'static,
        T: Fn(A) -> B + 'static,
    {
        Clo::new(move |vs, env| (self.inner)(vs, env)(a.clone()))
    }

    fn compose<A, B>(self, cla: Clo<A>) -> Clo<B>
    where
        A: Clone + 'static,
        B: Clone + 'static,
        T: Fn(A) -> B + 'static,
    {
        Clo::new(move |vs, env| {
            let f = (self.inner)(vs, env);
            let a = (cla.inner)(vs, env);
            f(a)
        })
    }
}

#[derive(Clone)]
struct VarInner<A> {
    name: String,
    mk_free: Rc<dyn Fn(Var<A>) -> A>,
    boxed: Boxed<A>,
}

#[derive(Clone)]
pub struct Var<A> {
    key: usize,
    inner: Rc<VarInner<A>>,
}

impl<A> Debug for Var<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let key = self.key;
        let name = &self.inner.name;
        write!(f, "{:?}@{:?}", name, key)
    }
}

impl<A> PartialEq for Var<A> {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl<A> PartialOrd for Var<A> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.key.partial_cmp(&other.key)
    }
}

impl<A> Eq for Var<A> {}

impl<A> Ord for Var<A> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key.cmp(&other.key)
    }
}

impl<A: From<Rc<dyn Any>> + 'static> Var<A> {
    pub fn new<F>(mk_free: F, name: String) -> Var<A>
    where
        F: Fn(Var<A>) -> A + 'static,
    {
        let key = STAMP.fetch_add(1, Relaxed);
        Var::build(key, Rc::new(mk_free), name)
    }

    fn build(key: usize, mk_free: Rc<dyn Fn(Var<A>) -> A>, name: String) -> Self {
        let x = Rc::new_cyclic(|wk: &Weak<VarInner<A>>| {
            let clo = Clo::new(move |vp, env| {
                let i = vp.inner.borrow()[&key];
                env.get::<A>(i)
            });
            let wk = AnyVar::from(key, wk.clone());
            let boxed = Boxed::mk_env(vec![wk], 0, clo);
            VarInner {
                name,
                mk_free,
                boxed,
            }
        });
        Var { key, inner: x }
    }
}

#[derive(Clone)]
struct AnyVar {
    key: usize,
    inner: Weak<dyn Any>,
}

impl PartialEq for AnyVar {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl PartialOrd for AnyVar {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.key.partial_cmp(&other.key)
    }
}

impl Eq for AnyVar {}

impl Ord for AnyVar {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key.cmp(&other.key)
    }
}

impl AnyVar {
    fn from(key: usize, wk: Weak<dyn Any>) -> Self {
        AnyVar { key, inner: wk }
    }

    fn downcast<A: 'static>(&self) -> Var<A> {
        let wk: Weak<dyn Any> = self.inner.clone();
        Var {
            key: self.key,
            inner: wk.upgrade().unwrap().downcast().unwrap(),
        }
    }
}

#[derive(Clone)]
pub struct Boxed<A> {
    inner: BoxedInner<A>,
}

#[derive(Clone)]
enum BoxedInner<A> {
    Box(A),
    Env(Vec<AnyVar>, usize, Clo<A>),
}

impl<A: 'static> Boxed<A> {
    fn mk_box(value: A) -> Self {
        Boxed {
            inner: BoxedInner::Box(value),
        }
    }

    fn mk_env(vs: Vec<AnyVar>, n: usize, t: Clo<A>) -> Self {
        Boxed {
            inner: BoxedInner::Env(vs, n, t),
        }
    }

    pub fn unbox(self) -> A
    where
        A: Into<Rc<dyn Any>> + Clone + 'static,
    {
        match self.inner {
            BoxedInner::Box(t) => t,
            BoxedInner::Env(vs, nb, t) => {
                let nbvs = vs.len();
                let env = Env::new(nbvs + nb);
                let mut cur = 0;
                let pos = Pos::new();
                for wk in vs {
                    let x = wk.downcast::<A>();
                    let v = (x.inner.mk_free)(x.clone());
                    env.set(cur, v);
                    pos.inner.borrow_mut().insert(x.key, cur);
                    cur += 1;
                }
                (t.inner)(&pos, &env)
            }
        }
    }
}

fn merge_unique(l1: &Vec<AnyVar>, l2: &Vec<AnyVar>) -> Vec<AnyVar> {
    let l1 = l1.iter();
    let l2 = l2.iter();
    merge(l1, l2).dedup().cloned().collect()
}

fn remove<A>(x: &Var<A>, xs: &Vec<AnyVar>) -> Option<Vec<AnyVar>> {
    let mut acc: Vec<AnyVar> = vec![];
    let mut it = xs.iter();
    while let Some(wk) = it.next() {
        if wk.key < x.key {
            acc.push(wk.clone());
        } else if wk.key == x.key {
            acc.extend(it.cloned());
            break;
        } else {
            return None;
        }
    }
    return Some(acc);
}

fn minimize_aux_prefix(size: usize, n: usize, env: &Env) -> Env {
    let new_env = Env::new(size + n);
    new_env.clone_from(env, size);
    new_env
}

fn minimize_aux(tab: Vec<usize>, n: usize, env: &Env) -> Env {
    let size = tab.len();
    let new_env = Env::new(size + n);
    for (i, x) in tab.iter().enumerate() {
        let opt = &env.tab.borrow()[*x];
        new_env.tab.borrow_mut()[i] = opt.as_ref().cloned();
    }
    new_env
}

fn minimize<A: 'static>(xs: Vec<AnyVar>, n: usize, t: Clo<A>) -> Clo<A> {
    if n == 0 {
        return t;
    }
    Clo::new(move |vp, env| {
        let size = xs.len();
        let mut tab: Vec<usize> = vec![0; size];
        let mut prefix = true;
        let vp1 = Pos::new();
        for (i, wk) in xs.iter().enumerate() {
            let j = vp.inner.borrow()[&wk.key];
            if i != j {
                prefix = false;
            }
            tab[i] = j;
            vp1.inner.borrow_mut().insert(wk.key, i);
        }
        let env = if prefix {
            minimize_aux_prefix(size, n, &env)
        } else {
            minimize_aux(tab, n, &env)
        };
        (t.inner)(&vp1, &env)
    })
}

#[derive(Clone)]
pub struct Binder<A, B> {
    var: Rc<VarInner<A>>,
    bind: bool,
    rank: usize,
    value: Rc<dyn Fn(A) -> B>,
}

impl<A, B> Debug for Binder<A, B>
where
    A: Into<Rc<dyn Any>> + From<Rc<dyn Any>> + Debug + Clone + 'static,
    B: Debug + Clone + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (x, m) = self.unbind();
        f.debug_struct("Binder")
            .field("var", &x)
            .field("body", &m)
            .finish()
    }
}

impl<A, B> Binder<A, B>
where
    A: Into<Rc<dyn Any>> + From<Rc<dyn Any>> + Clone + 'static,
    B: Clone + 'static,
{
    pub fn name(&self) -> String {
        self.var.name.clone()
    }

    pub fn subst(&self, x: A) -> B {
        (self.value)(x)
    }

    pub fn occur(&self) -> bool {
        self.bind
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn is_closed(&self) -> bool {
        self.rank == 0
    }

    pub fn compose<C: 'static, F>(self, f: F) -> Binder<A, C>
    where
        F: Fn(B) -> C + 'static,
    {
        Binder::<A, C> {
            var: self.var,
            bind: self.bind,
            rank: self.rank,
            value: Rc::new(move |x| f((self.value)(x))),
        }
    }

    pub fn unbind(&self) -> (Var<A>, B)
    where
        A: Clone + 'static,
        B: Clone + 'static,
    {
        let key = STAMP.fetch_add(1, Relaxed);
        let x = Var::build(key, self.var.mk_free.clone(), self.var.name.clone());
        let v = (self.var.mk_free)(x.clone());
        let m = self.subst(v);
        return (x, m);
    }

    fn bind_var_aux1(t: Clo<B>, pos: Pos, env: Env) -> Rc<dyn Fn(A) -> B> {
        Rc::new(move |arg| {
            env.set(0, arg);
            (t.inner)(&pos, &env)
        })
    }

    fn bind_var_aux2(rank: usize, t: Clo<B>, pos: Pos, env: Env) -> Rc<dyn Fn(A) -> B> {
        Rc::new(move |arg| {
            env.set(rank, arg);
            (t.inner)(&pos, &env)
        })
    }

    fn bind_var_aux3(x: Var<A>, rank: usize, t: Clo<B>, pos: Pos, env: Env) -> Binder<A, B> {
        let value = Self::bind_var_aux2(rank, t, pos, env);
        Binder {
            var: x.inner,
            rank,
            bind: true,
            value,
        }
    }

    fn bind_var_aux4(t: Clo<B>, pos: Pos, env: Env) -> Rc<dyn Fn(A) -> B> {
        Rc::new(move |_| (t.inner)(&pos, &env))
    }

    fn bind_var_aux5(x: Var<A>, rank: usize, t: Clo<B>, pos: Pos, env: Env) -> Binder<A, B> {
        let value = Self::bind_var_aux4(t, pos, env);
        Binder {
            var: x.inner,
            rank,
            bind: false,
            value,
        }
    }

    pub fn bind_var(x: Var<A>, b: Boxed<B>) -> Boxed<Binder<A, B>> {
        match b.inner {
            BoxedInner::Box(t) => Boxed::mk_box(Binder {
                var: x.inner,
                bind: false,
                rank: 0,
                value: Rc::new(move |_| t.clone()),
            }),
            BoxedInner::Env(xs, n, t) => {
                if xs.len() == 1 && x.key == xs[0].key {
                    let vp = Pos::new();
                    vp.inner.borrow_mut().insert(x.key, 0);
                    let value = Self::bind_var_aux1(t, vp, Env::new(n + 1));
                    return Boxed::mk_box(Binder {
                        var: x.inner,
                        rank: 0,
                        bind: true,
                        value,
                    });
                }
                if let Some(xs) = remove(&x, &xs) {
                    let key = x.key;
                    let rank = xs.len();
                    let clo = Clo::new(move |vp, env| {
                        vp.inner.borrow_mut().insert(key, rank);
                        Self::bind_var_aux3(x.clone(), rank, t.clone(), vp.clone(), env.clone())
                    });
                    return Boxed::mk_env(xs, n + 1, clo);
                };
                let rank = xs.len();
                let clo = Clo::new(move |vp, env| {
                    Self::bind_var_aux5(x.clone(), rank, t.clone(), vp.clone(), env.clone())
                });
                Boxed::mk_env(xs, n, clo)
            }
        }
    }
}

#[derive(Clone)]
pub struct MBinder<A, B> {
    vars: Vec<Rc<VarInner<A>>>,
    binds: Vec<bool>,
    rank: usize,
    value: Rc<dyn Fn(Vec<A>) -> B>,
}

impl<A, B> Debug for MBinder<A, B>
where
    A: Into<Rc<dyn Any>> + From<Rc<dyn Any>> + Debug + Clone + 'static,
    B: Debug + Clone + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (xs, m) = self.unbind();
        f.debug_struct("MBinder")
            .field("vars", &xs)
            .field("body", &m)
            .finish()
    }
}

impl<A, B> MBinder<A, B>
where
    A: Into<Rc<dyn Any>> + From<Rc<dyn Any>> + Clone + 'static,
    B: Clone + 'static,
{
    pub fn arity(&self) -> usize {
        self.vars.len()
    }

    pub fn names(&self) -> Vec<String> {
        self.vars.iter().map(|v| v.name.clone()).collect()
    }

    pub fn subst(&self, x: Vec<A>) -> B {
        (self.value)(x)
    }

    pub fn occurs(&self) -> Vec<bool> {
        self.binds.clone()
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn is_closed(&self) -> bool {
        self.rank == 0
    }

    pub fn compose<C: 'static, F>(self, f: F) -> MBinder<A, C>
    where
        F: Fn(B) -> C + 'static,
    {
        MBinder::<A, C> {
            vars: self.vars,
            binds: self.binds,
            rank: self.rank,
            value: Rc::new(move |xs| f((self.value)(xs))),
        }
    }

    pub fn unbind(&self) -> (Vec<Var<A>>, B)
    where
        A: Into<Rc<dyn Any>> + From<Rc<dyn Any>> + Clone + 'static,
        B: Clone + 'static,
    {
        let mut xs = vec![];
        let mut ms = vec![];
        for v in self.vars.iter() {
            let key = STAMP.fetch_add(1, Relaxed);
            let x = Var::build(key, v.mk_free.clone(), v.name.clone());
            xs.push(x.clone());
            ms.push((v.mk_free)(x));
        }
        let m = self.subst(ms);
        return (xs, m);
    }

    fn bind_mvar_aux1(binds: Vec<bool>, t: Clo<B>, pos: Pos, env: Env) -> Rc<dyn Fn(Vec<A>) -> B> {
        let arity = binds.len();
        Rc::new(move |args| {
            assert_eq!(args.len(), arity);
            let mut n = 0;
            for i in 0..arity {
                if binds[i] {
                    env.set(n, args[i].clone());
                    n += 1;
                }
            }
            (t.inner)(&pos, &env)
        })
    }

    fn bind_mvar_aux2(arity: usize, t: Clo<B>, pos: Pos, env: Env) -> Rc<dyn Fn(Vec<A>) -> B> {
        Rc::new(move |args| {
            assert_eq!(args.len(), arity);
            (t.inner)(&pos, &env)
        })
    }

    fn bind_mvar_aux3(
        t: Clo<B>,
        vars: Vec<Rc<VarInner<A>>>,
        rank: usize,
        binds: Vec<bool>,
        pos: Pos,
        env: Env,
    ) -> MBinder<A, B> {
        let value = Self::bind_mvar_aux2(binds.len(), t, pos, env);
        MBinder {
            vars,
            binds,
            rank,
            value,
        }
    }

    fn bind_mvar_aux4(
        t: Clo<B>,
        rank: usize,
        binds: Vec<bool>,
        pos: Pos,
        env: Env,
    ) -> Rc<dyn Fn(Vec<A>) -> B> {
        let arity = binds.len();
        Rc::new(move |args| {
            assert_eq!(args.len(), arity);
            let mut cur_pos = rank;
            for i in 0..arity {
                if binds[i] {
                    env.set(cur_pos, args[i].clone());
                    cur_pos += 1;
                }
            }
            (t.inner)(&pos, &env)
        })
    }

    fn bind_mvar_aux5(
        t: Clo<B>,
        vars: Vec<Rc<VarInner<A>>>,
        rank: usize,
        binds: Vec<bool>,
        pos: Pos,
        env: Env,
    ) -> MBinder<A, B> {
        let value = Self::bind_mvar_aux4(t, rank, binds.clone(), pos, env);
        MBinder {
            vars,
            binds,
            rank,
            value,
        }
    }

    pub fn bind_mvar(xs: Vec<Var<A>>, b: Boxed<B>) -> Boxed<MBinder<A, B>> {
        match b.inner {
            BoxedInner::Box(t) => {
                let mut vars = vec![];
                let mut binds = vec![];
                for x in xs.iter() {
                    vars.push(x.inner.clone());
                    binds.push(false);
                }
                Boxed::mk_box(MBinder {
                    vars,
                    binds,
                    rank: 0,
                    value: Rc::new(move |args| {
                        assert_eq!(args.len(), xs.len());
                        t.clone()
                    }),
                })
            }
            BoxedInner::Env(mut vs, n, t) => {
                let mut keys = vec![];
                let mut m = n;
                for x in xs.iter() {
                    if let Some(vs1) = remove(x, &vs) {
                        vs = vs1;
                        m += 1;
                        keys.push(Some(x.key));
                    } else {
                        keys.push(None);
                    }
                }
                if vs.is_empty() {
                    let mut vars = vec![];
                    let mut binds = vec![];
                    let mut cur_pos = 0;
                    let vp = Pos::new();
                    for (i, key) in keys.iter().enumerate() {
                        vars.push(xs[i].inner.clone());
                        if let Some(k) = key {
                            vp.inner.borrow_mut().insert(*k, cur_pos);
                            binds.push(true);
                            cur_pos += 1;
                        } else {
                            binds.push(false);
                        }
                    }
                    let value = Self::bind_mvar_aux1(binds.clone(), t, vp, Env::new(m));
                    Boxed::mk_box(MBinder {
                        vars,
                        binds,
                        rank: 0,
                        value,
                    })
                } else if m == n {
                    let rank = vs.len();
                    let clo = Clo::new(move |vp, env| {
                        let mut vars = vec![];
                        let mut binds = vec![];
                        for x in xs.iter() {
                            vars.push(x.inner.clone());
                            binds.push(false);
                        }
                        Self::bind_mvar_aux3(t.clone(), vars, rank, binds, vp.clone(), env.clone())
                    });
                    Boxed::mk_env(vs, m, clo)
                } else {
                    let rank = vs.len();
                    let clo = Clo::new(move |vp, env| {
                        let mut vars = vec![];
                        let mut binds = vec![];
                        let mut cur_pos = rank;
                        for (i, key) in keys.iter().enumerate() {
                            vars.push(xs[i].inner.clone());
                            if let Some(k) = key {
                                vp.inner.borrow_mut().insert(*k, cur_pos);
                                binds.push(true);
                                cur_pos += 1;
                            } else {
                                binds.push(false);
                            }
                        }
                        MBinder::bind_mvar_aux5(
                            t.clone(),
                            vars,
                            rank,
                            binds,
                            vp.clone(),
                            env.clone(),
                        )
                    });
                    Boxed::mk_env(vs, m, clo)
                }
            }
        }
    }
}

fn boxed_apply<A, B, F>(f: Boxed<F>, a: Boxed<A>) -> Boxed<B>
where
    A: Clone + 'static,
    B: Clone + 'static,
    F: Fn(A) -> B + Clone + 'static,
{
    match (f.inner, a.inner) {
        (BoxedInner::Box(f), BoxedInner::Box(a)) => Boxed::mk_box(f(a)),
        (BoxedInner::Box(f), BoxedInner::Env(va, na, ta)) => Boxed::mk_env(va, na, ta.map(f)),
        (BoxedInner::Env(vf, nf, tf), BoxedInner::Box(a)) => Boxed::mk_env(vf, nf, tf.app(a)),
        (BoxedInner::Env(vf, nf, tf), BoxedInner::Env(va, na, ta)) => {
            let vs = merge_unique(&vf, &va);
            let clof = minimize(vf, nf, tf);
            let cloa = minimize(va, na, ta);
            Boxed::mk_env(vs, 0, Clo::compose(clof, cloa))
        }
    }
}

pub fn boxed<A>(a: A) -> Boxed<A>
where
    A: Clone + 'static,
{
    Boxed {
        inner: BoxedInner::Box(a),
    }
}

pub fn apply1<A, B, F>(f: F, ta: Boxed<A>) -> Boxed<B>
where
    A: Clone + 'static,
    B: Clone + 'static,
    F: Fn(A) -> B + Clone + 'static,
{
    match ta.inner {
        BoxedInner::Box(a) => Boxed::mk_box(f(a)),
        BoxedInner::Env(xs, na, ta) => Boxed::mk_env(xs, na, ta.map(f)),
    }
}

pub fn apply2<A, B, C, F>(f: F, ta: Boxed<A>, tb: Boxed<B>) -> Boxed<C>
where
    A: Clone + 'static,
    B: Clone + 'static,
    C: Clone + 'static,
    F: Fn(A, B) -> C + Clone + 'static,
{
    let g = move |a: A| {
        let f = f.clone();
        move |b: B| f(a.clone(), b)
    };
    boxed_apply(apply1(g, ta), tb)
}

pub fn apply3<A, B, C, D, F>(f: F, ta: Boxed<A>, tb: Boxed<B>, tc: Boxed<C>) -> Boxed<D>
where
    A: Clone + 'static,
    B: Clone + 'static,
    C: Clone + 'static,
    D: Clone + 'static,
    F: Fn(A, B, C) -> D + Clone + 'static,
{
    let g = move |a: A, b: B| {
        let f = f.clone();
        move |c: C| f(a.clone(), b.clone(), c)
    };
    boxed_apply(apply2(g, ta, tb), tc)
}

pub fn apply4<A, B, C, D, E, F>(
    f: F,
    ta: Boxed<A>,
    tb: Boxed<B>,
    tc: Boxed<C>,
    td: Boxed<D>,
) -> Boxed<E>
where
    A: Clone + 'static,
    B: Clone + 'static,
    C: Clone + 'static,
    D: Clone + 'static,
    E: Clone + 'static,
    F: Fn(A, B, C, D) -> E + Clone + 'static,
{
    let g = move |a: A, b: B, c: C| {
        let f = f.clone();
        move |d: D| f(a.clone(), b.clone(), c.clone(), d)
    };
    boxed_apply(apply3(g, ta, tb, tc), td)
}

impl<A: Clone + 'static> From<Var<A>> for Boxed<A> {
    fn from(var: Var<A>) -> Boxed<A> {
        var.inner.boxed.clone()
    }
}

impl<A, B> From<Binder<A, Boxed<B>>> for Boxed<Binder<A, B>>
where
    A: Into<Rc<dyn Any>> + From<Rc<dyn Any>> + Clone + 'static,
    B: Clone + 'static,
{
    fn from(binder: Binder<A, Boxed<B>>) -> Boxed<Binder<A, B>> {
        let (xs, t) = binder.unbind();
        Binder::bind_var(xs, t)
    }
}

impl<A, B> From<MBinder<A, Boxed<B>>> for Boxed<MBinder<A, B>>
where
    A: Into<Rc<dyn Any>> + From<Rc<dyn Any>> + Clone + 'static,
    B: Clone + 'static,
{
    fn from(mbinder: MBinder<A, Boxed<B>>) -> Boxed<MBinder<A, B>> {
        let (xs, t) = mbinder.unbind();
        MBinder::bind_mvar(xs, t)
    }
}

impl<A> From<Option<Boxed<A>>> for Boxed<Option<A>>
where
    A: Clone + 'static,
{
    fn from(opt: Option<Boxed<A>>) -> Boxed<Option<A>> {
        match opt {
            Some(a) => match a.inner {
                BoxedInner::Box(t) => boxed(Some(t)),
                BoxedInner::Env(vs, n, t) => {
                    let clo = Clo::new(move |vp, env| Some((t.inner)(vp, env)));
                    Boxed::mk_env(vs, n, clo)
                }
            },
            None => boxed(None),
        }
    }
}

impl<A, E> From<Result<Boxed<A>, E>> for Boxed<Result<A, E>>
where
    A: Clone + 'static,
    E: Clone + 'static,
{
    fn from(opt: Result<Boxed<A>, E>) -> Boxed<Result<A, E>> {
        match opt {
            Ok(a) => match a.inner {
                BoxedInner::Box(t) => boxed(Ok(t)),
                BoxedInner::Env(vs, n, t) => {
                    let clo = Clo::new(move |vp, env| Ok((t.inner)(vp, env)));
                    Boxed::mk_env(vs, n, clo)
                }
            },
            Err(e) => boxed(Err(e)),
        }
    }
}

impl<A, T1, T2> From<T1> for Boxed<T2>
where
    A: Clone + 'static,
    T1: IntoIterator<Item = Boxed<A>> + FromIterator<Boxed<A>> + 'static,
    T2: FromIterator<A> + 'static,
{
    fn from(val: T1) -> Boxed<T2> {
        let mut b = true;
        let mut vars = vec![];
        let mut clos = vec![];
        let mut n = 0;
        for a in val.into_iter() {
            match a.inner {
                BoxedInner::Box(t) => {
                    clos.push(Clo::new(move |_, _| t.clone()));
                }
                BoxedInner::Env(vs, na, ta) => {
                    b = false;
                    n = cmp::max(na, n);
                    vars = merge_unique(&vars, &vs);
                    clos.push(minimize(vs, na, ta));
                }
            }
        }
        let f = move |vp: &Pos, env: &Env| clos.iter().map(|c| (c.inner)(vp, env)).collect();
        if b {
            let vp = Pos::new();
            Boxed::mk_box(f(&vp, &Env::new(0)))
        } else {
            let clo = Clo::new(f);
            Boxed::mk_env(vars.clone(), n, minimize(vars, n, clo))
        }
    }
}
