"""Conservative typed Python AST to Rust. Unsupported semantics fall back.

Every native operation either produces exactly CPython's result or returns a
non-zero status, in which case the runtime replays the call in Python.

Kinds used for values:
  'i64' int, 'f64' float, 'bool' bool,
  'array' / 'iarray' 1-D float64 / int64 NumPy arrays,
  'array2' / 'iarray2' 2-D float64 / int64 NumPy arrays,
  'flist' / 'ilist' Python lists of floats / ints (copied in, copied back after writes),
  'series' / 'iseries' pandas Series of float64 / int64 with a default 0..n-1 index (read-only),
  'str' strings (read-only; elements are 'char', one character) and 'bytes' (read-only; elements are ints),
  'tuple(k1,k2,...)' tuples of scalars (return values and unpacking only),
  'none' for functions that return None.
"""

import ast
import builtins
import inspect
import math
import random as _random
import sys
import textwrap
import types


class Unsupported(ValueError):
    pass


class F64Array:
    """Annotation for a contiguous, aligned, native-endian 1-D NumPy float64 array.

    Read elements with a[i] or float(a[i]); write them with a[i] = value. If a native
    call fails halfway, its writes are undone before the call is replayed in Python.
    """


class I64Array:
    """Annotation for a contiguous, aligned, native-endian 1-D NumPy int64 array."""


class F64Array2:
    """Annotation for a C-contiguous 2-D NumPy float64 array: read and write a[i, j]."""


class I64Array2:
    """Annotation for a C-contiguous 2-D NumPy int64 array: read and write a[i, j]."""


def prange(*args):
    """Like range(), but PyRust may run the iterations on several CPU cores.

    In plain Python it is exactly range(). Use it only when iterations are independent:
    each one writes its own array element (out[i] = ...) and may add to int totals.
    """
    return range(*args)


# Every sequence kind is passed to Rust as (pointer, rows, cols) and read through Arr<T>.
ARRAY_INFO = {
    'array': ('f64', 1),
    'iarray': ('i64', 1),
    'array2': ('f64', 2),
    'iarray2': ('i64', 2),
    'flist': ('f64', 1),
    'ilist': ('i64', 1),
    'series': ('f64', 1),
    'iseries': ('i64', 1),
    'str': ('u32', 1),
    'bytes': ('u8', 1),
}
ARRAY_KINDS = tuple(ARRAY_INFO)
NUMPY_KINDS = ('array', 'iarray', 'array2', 'iarray2')
LIST_KINDS = ('flist', 'ilist')
READ_ONLY_KINDS = ('series', 'iseries', 'str', 'bytes')
ELEMENT = {k: v[0] for k, v in ARRAY_INFO.items()}
# What reading one element gives: str elements are one-character strings, bytes elements are ints.
READ_KIND = {'f64': 'f64', 'i64': 'i64', 'u32': 'char', 'u8': 'i64'}
SCALARS = ('i64', 'f64', 'bool')
VALUES = ('i64', 'f64', 'bool', 'char')  # scalars plus single characters
RUST_TYPE = {'i64': 'i64', 'f64': 'f64', 'bool': 'bool', 'char': 'u32', 'none': '()'}
ZERO = {'i64': '0', 'f64': '0.0', 'bool': 'false', 'char': '0'}
ANNOTATIONS = {
    int: 'i64',
    'int': 'i64',
    float: 'f64',
    'float': 'f64',
    bool: 'bool',
    'bool': 'bool',
    F64Array: 'array',
    'F64Array': 'array',
    I64Array: 'iarray',
    'I64Array': 'iarray',
    F64Array2: 'array2',
    'F64Array2': 'array2',
    I64Array2: 'iarray2',
    'I64Array2': 'iarray2',
    str: 'str',
    'str': 'str',
    bytes: 'bytes',
    'bytes': 'bytes',
    list: 'flist',
    'list': 'flist',
    list[float]: 'flist',
    'list[float]': 'flist',
    list[int]: 'ilist',
    'list[int]': 'ilist',
}
RETURN_ANNOTATIONS = {
    int: 'i64',
    'int': 'i64',
    float: 'f64',
    'float': 'f64',
    bool: 'bool',
    'bool': 'bool',
    None: 'none',
    'None': 'none',
    str: 'char',
    'str': 'char',
}
MAX_DEPTH = 300  # native call depth; deeper recursion replays in Python (keeps thread stacks safe)
PARALLEL_MIN = 4096  # prange loops shorter than this run on one core

# Status codes shared with runtime._STATUS.
#   1 integer overflow          2 zero divisor / zero range step
#   3 array index out of bounds 4 iteration budget exceeded
#   5 no scalar return          6 value outside the native fast path (math domain,
#                                  float overflow, negative exponent, huge int, ...)
#   7 interrupted (Ctrl-C)      8 internal error (Rust panic), replayed in Python
HELPERS = r"""
use std::sync::atomic::{AtomicU8, Ordering};
// rng: Python's Mersenne Twister state (624 words + position), gauss: [gauss_next, has_gauss_next].
pub struct Ctx { fuel: u64, cancel: *const AtomicU8, ticks: u32, rng: *mut u32, gauss: *mut f64 }
fn tick(c:&mut Ctx)->Result<(),i32>{
    if c.fuel==0 {return Err(4)}
    c.fuel-=1;
    c.ticks=c.ticks.wrapping_add(1);
    if c.ticks & 0xFFFF==0 && !c.cancel.is_null() && unsafe{(*c.cancel).load(Ordering::Relaxed)}!=0 {return Err(7)}
    Ok(())
}
struct SendPtr<T>(*mut T);
impl<T> Clone for SendPtr<T> { fn clone(&self) -> Self { *self } }
impl<T> Copy for SendPtr<T> {}
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> SendPtr<T> { fn get(self) -> *mut T { self.0 } }
pub struct Arr<T:Copy>{ ptr:*mut T, len:usize, rows:usize, cols:usize, track:bool,
                        log:Vec<(usize,T)>, orig:Option<Vec<T>>, owned:Vec<T> }
impl<T:Copy> Arr<T>{
    fn view(ptr:*mut T,rows:usize,cols:usize,track:bool)->Arr<T>{
        Arr{ptr,len:rows*cols,rows,cols,track,log:Vec::new(),orig:None,owned:Vec::new()}
    }
    fn empty()->Arr<T>{
        Arr{ptr:std::ptr::null_mut(),len:0,rows:0,cols:0,track:false,log:Vec::new(),orig:None,owned:Vec::new()}
    }
    fn owned(rows:i64,cols:i64,fill:T)->Result<Arr<T>,i32>{
        if rows<0 || cols<0 {return Err(6)}
        let len=(rows as u64).checked_mul(cols as u64).ok_or(6)?;
        if len>(1u64<<40) {return Err(6)}
        let mut v:Vec<T>=Vec::new();
        v.try_reserve_exact(len as usize).map_err(|_|6)?;
        v.resize(len as usize,fill);
        let ptr=v.as_mut_ptr();
        Ok(Arr{ptr,len:len as usize,rows:rows as usize,cols:cols as usize,track:false,log:Vec::new(),orig:None,owned:v})
    }
    fn snapshot(&mut self){
        // Keep a full copy of the original contents (used before parallel writes).
        if self.track && self.orig.is_none() {
            let mut o=unsafe{std::slice::from_raw_parts(self.ptr,self.len)}.to_vec();
            for &(k,old) in self.log.iter().rev() {o[k]=old;}
            self.orig=Some(o); self.log=Vec::new();
        }
    }
    fn write(&mut self,j:usize,v:T){
        if self.track && self.orig.is_none() {
            self.log.push((j,unsafe{*self.ptr.add(j)}));
            // Few writes: keep the small undo log. Many writes: one copy is cheaper.
            if self.log.len()>64 && self.log.len()*16>self.len {self.snapshot();}
        }
        unsafe{*self.ptr.add(j)=v;}
    }
    fn rollback(&mut self){
        if !self.track {return}
        unsafe{
            if let Some(o)=&self.orig {std::ptr::copy_nonoverlapping(o.as_ptr(),self.ptr,self.len);}
            else {for &(k,old) in self.log.iter().rev() {*self.ptr.add(k)=old;}}
        }
    }
}
fn index(len:usize,i:i64)->Result<usize,i32>{
    let j=if i<0 {(len as i128)+(i as i128)} else {i as i128};
    if j<0 || j>=len as i128 {Err(3)} else {Ok(j as usize)}
}
fn at<T:Copy>(a:&Arr<T>,i:i64)->Result<T,i32>{let j=index(a.len,i)?; Ok(unsafe{*a.ptr.add(j)})}
fn at2<T:Copy>(a:&Arr<T>,i:i64,k:i64)->Result<T,i32>{
    let r=index(a.rows,i)?; let c=index(a.cols,k)?; Ok(unsafe{*a.ptr.add(r*a.cols+c)})
}
fn put<T:Copy>(a:&mut Arr<T>,i:i64,v:T)->Result<(),i32>{let j=index(a.len,i)?; a.write(j,v); Ok(())}
// pandas Series s[i] looks up the label i: negative numbers are not positions there.
fn at_nn<T:Copy>(a:&Arr<T>,i:i64)->Result<T,i32>{if i<0 {return Err(3)} at(a,i)}
fn has<T:Copy+PartialEq>(a:&Arr<T>,x:T)->bool{
    for j in 0..a.len {if unsafe{*a.ptr.add(j)}==x {return true}}
    false
}
// One-character str methods: exact for ASCII, anything else replays in Python.
fn ascii(c:u32)->Result<u8,i32>{if c<128 {Ok(c as u8)} else {Err(6)}}
fn c_isdigit(c:u32)->Result<bool,i32>{Ok(ascii(c)?.is_ascii_digit())}
fn c_isalpha(c:u32)->Result<bool,i32>{Ok(ascii(c)?.is_ascii_alphabetic())}
fn c_isalnum(c:u32)->Result<bool,i32>{Ok(ascii(c)?.is_ascii_alphanumeric())}
fn c_isspace(c:u32)->Result<bool,i32>{let b=ascii(c)?; Ok(b==b' ' || (9..=13).contains(&b) || (28..=31).contains(&b))}
fn c_isupper(c:u32)->Result<bool,i32>{Ok(ascii(c)?.is_ascii_uppercase())}
fn c_islower(c:u32)->Result<bool,i32>{Ok(ascii(c)?.is_ascii_lowercase())}
fn c_lower(c:u32)->Result<u32,i32>{Ok(ascii(c)?.to_ascii_lowercase() as u32)}
fn c_upper(c:u32)->Result<u32,i32>{Ok(ascii(c)?.to_ascii_uppercase() as u32)}
fn to_char(i:i64)->Result<u32,i32>{if (0..=0x10FFFF).contains(&i) {Ok(i as u32)} else {Err(6)}}
// CPython's random module (_randommodule.c and random.py), bit for bit.
fn mt_next(c:&mut Ctx)->Result<u32,i32>{
    if c.rng.is_null() {return Err(6)}
    let mt=unsafe{std::slice::from_raw_parts_mut(c.rng,625)};
    let mut pos=mt[624] as usize;
    if pos>=624 {
        const N:usize=624; const M:usize=397;
        for k in 0..N {
            let y=(mt[k]&0x80000000)|(mt[(k+1)%N]&0x7fffffff);
            let mut v=mt[(k+M)%N]^(y>>1);
            if y&1!=0 {v^=0x9908b0df;}
            mt[k]=v;
        }
        pos=0;
    }
    let mut y=mt[pos];
    mt[624]=(pos+1) as u32;
    y^=y>>11; y^=(y<<7)&0x9d2c5680; y^=(y<<15)&0xefc60000; y^=y>>18;
    Ok(y)
}
fn rnd_random(c:&mut Ctx)->Result<f64,i32>{
    let a=(mt_next(c)?>>5) as f64; let b=(mt_next(c)?>>6) as f64;
    Ok((a*67108864.0+b)*(1.0/9007199254740992.0))
}
fn rnd_bits(c:&mut Ctx,k:u32)->Result<i64,i32>{
    if k==0 {return Ok(0)}
    if k<=32 {return Ok((mt_next(c)?>>(32-k)) as i64)}
    if k>63 {return Err(6)}
    let mut r:u64=0; let mut left=k as i64; let mut word=0;
    while left>0 {
        let mut w=mt_next(c)?;
        if left<32 {w>>=32-left;}
        r|=(w as u64)<<(32*word);
        word+=1; left-=32;
    }
    Ok(r as i64)
}
fn rnd_below(c:&mut Ctx,n:i64)->Result<i64,i32>{
    if n<=0 {return Err(6)}
    let k=64-(n as u64).leading_zeros();
    let mut r=rnd_bits(c,k)?;
    while r>=n {r=rnd_bits(c,k)?;}
    Ok(r)
}
fn rnd_range(c:&mut Ctx,start:i64,stop:i64)->Result<i64,i32>{
    let width=stop.checked_sub(start).ok_or(1)?;
    if width<=0 {return Err(6)}
    Ok(start+rnd_below(c,width)?)
}
fn rnd_gauss(c:&mut Ctx,mu:f64,sigma:f64)->Result<f64,i32>{
    let g=c.gauss;
    if g.is_null() {return Err(6)}
    let z;
    if unsafe{*g.add(1)}!=0.0 {
        z=unsafe{*g};
        unsafe{*g=0.0; *g.add(1)=0.0;}
    } else {
        let x2pi=rnd_random(c)?*(2.0*std::f64::consts::PI);
        let g2rad=(-2.0*(1.0-rnd_random(c)?).ln()).sqrt();
        z=x2pi.cos()*g2rad;
        unsafe{*g=x2pi.sin()*g2rad; *g.add(1)=1.0;}
    }
    Ok(mu+z*sigma)
}
// Parallel loops write distinct elements of a snapshotted array through a shared view.
unsafe impl<T:Copy> Sync for Arr<T> {}
unsafe impl<T:Copy> Send for Arr<T> {}
fn put_par<T:Copy>(a:&Arr<T>,i:i64,v:T)->Result<(),i32>{let j=index(a.len,i)?; unsafe{*a.ptr.add(j)=v;} Ok(())}
fn put2_par<T:Copy>(a:&Arr<T>,i:i64,k:i64,v:T)->Result<(),i32>{
    let r=index(a.rows,i)?; let c=index(a.cols,k)?; unsafe{*a.ptr.add(r*a.cols+c)=v;} Ok(())
}
fn put2<T:Copy>(a:&mut Arr<T>,i:i64,k:i64,v:T)->Result<(),i32>{
    let r=index(a.rows,i)?; let c=index(a.cols,k)?; a.write(r*a.cols+c,v); Ok(())
}
fn into_raw<T:Copy>(a:Arr<T>)->(*mut T,usize){
    let b=a.owned.into_boxed_slice(); let n=b.len(); (Box::into_raw(b) as *mut T,n)
}
#[no_mangle] pub unsafe extern "C" fn pyrust_free_f64(p:*mut f64,n:usize){
    if !p.is_null() {drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(p,n)));}
}
#[no_mangle] pub unsafe extern "C" fn pyrust_free_i64(p:*mut i64,n:usize){
    if !p.is_null() {drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(p,n)));}
}
fn floor_div(a:i64,b:i64)->Result<i64,i32>{
    if b==0 {return Err(2)}
    let a=a as i128; let b=b as i128;
    let q=a/b; let r=a%b;
    let q=if r!=0 && ((r<0)!=(b<0)) {q-1} else {q};
    i64::try_from(q).map_err(|_|1)
}
fn modulo(a:i64,b:i64)->Result<i64,i32>{
    if b==0 {return Err(2)}
    let a=a as i128; let b=b as i128; let r=a%b;
    Ok((if r!=0 && ((r<0)!=(b<0)) {r+b} else {r}) as i64)
}
fn divide(a:f64,b:f64)->Result<f64,i32>{if b==0.0 {Err(2)} else {Ok(a/b)}}
fn int_true_div(a:i64,b:i64)->Result<f64,i32>{
    // CPython divides exactly-representable ints as floats; larger ones need
    // its correctly rounded long division, so those calls replay.
    if b==0 {return Err(2)}
    const LIM:i64=1i64<<53;
    if a< -LIM || a>LIM || b< -LIM || b>LIM {return Err(6)}
    Ok((a as f64)/(b as f64))
}
fn float_floor_div(a:f64,b:f64)->Result<f64,i32>{
    // CPython _float_div_mod
    if b==0.0 {return Err(2)}
    let m=a%b;
    let mut d=(a-m)/b;
    if m!=0.0 && ((b<0.0)!=(m<0.0)) {d-=1.0;}
    if d!=0.0 {
        let f=d.floor();
        Ok(if d-f>0.5 {f+1.0} else {f})
    } else {
        Ok((0.0f64).copysign(a/b))
    }
}
fn float_mod(a:f64,b:f64)->Result<f64,i32>{
    // CPython float_rem
    if b==0.0 {return Err(2)}
    let mut m=a%b;
    if m!=0.0 {
        if (b<0.0)!=(m<0.0) {m+=b;}
    } else {
        m=(0.0f64).copysign(b);
    }
    Ok(m)
}
fn int_pow(a:i64,b:i64)->Result<i64,i32>{
    if b<0 {return Err(6)}
    if b>u32::MAX as i64 {
        return match a {0=>Ok(0),1=>Ok(1),-1=>Ok(if b%2==0 {1} else {-1}),_=>Err(1)};
    }
    a.checked_pow(b as u32).ok_or(1)
}
fn float_pow(a:f64,b:f64)->Result<f64,i32>{
    // CPython float_pow: special cases first, then C pow with error checks.
    if b==0.0 {return Ok(1.0)}
    if a.is_nan() {return Ok(a)}
    if b.is_nan() {return Ok(if a==1.0 {1.0} else {b})}
    if a.is_infinite() || b.is_infinite() {return Ok(a.powf(b))}
    if a==0.0 && b<0.0 {return Err(2)}
    if a<0.0 && b!=b.floor() {return Err(6)}
    let r=a.powf(b);
    if r.is_infinite() {return Err(6)}
    Ok(r)
}
fn shift_left(a:i64,b:i64)->Result<i64,i32>{
    if b<0 {return Err(6)}
    if a==0 {return Ok(0)}
    if b>=64 {return Err(1)}
    i64::try_from((a as i128)<<b).map_err(|_|1)
}
fn shift_right(a:i64,b:i64)->Result<i64,i32>{
    if b<0 {return Err(6)}
    Ok(if b>=64 {if a<0 {-1} else {0}} else {a>>b})
}
fn max_i(a:i64,b:i64)->i64{if b>a {b} else {a}}
fn min_i(a:i64,b:i64)->i64{if b<a {b} else {a}}
fn max_f(a:f64,b:f64)->f64{if b>a {b} else {a}}
fn min_f(a:f64,b:f64)->f64{if b<a {b} else {a}}
fn to_int(x:f64)->Result<i64,i32>{
    if !x.is_finite() {return Err(6)}
    let t=x.trunc();
    if t>=-9223372036854775808.0 && t<9223372036854775808.0 {Ok(t as i64)} else {Err(1)}
}
fn round_int(x:f64)->Result<i64,i32>{to_int(x.round_ties_even())}
fn math1(x:f64,r:f64)->Result<f64,i32>{
    // CPython math_1: NaN from a non-NaN input is a domain error, and an
    // infinite result from a finite input is a domain or overflow error.
    if r.is_nan() && !x.is_nan() {return Err(6)}
    if r.is_infinite() && x.is_finite() {return Err(6)}
    Ok(r)
}
fn m_atan2(y:f64,x:f64)->f64{
    // CPython m_atan2: C99 special cases fixed in Python itself, then libm.
    const PI:f64=std::f64::consts::PI;
    if x.is_nan() || y.is_nan() {return f64::NAN}
    if y.is_infinite() {
        if x.is_infinite() {
            return if (1.0f64).copysign(x)==1.0 {(0.25*PI).copysign(y)} else {(0.75*PI).copysign(y)};
        }
        return (0.5*PI).copysign(y);
    }
    if x.is_infinite() || y==0.0 {
        return if (1.0f64).copysign(x)==1.0 {(0.0f64).copysign(y)} else {PI.copysign(y)};
    }
    y.atan2(x)
}
fn m_fmod(x:f64,y:f64)->Result<f64,i32>{
    // CPython math.fmod
    if y.is_infinite() && x.is_finite() {return Ok(x)}
    let r=x%y;
    if r.is_nan() && !x.is_nan() && !y.is_nan() {return Err(6)}
    Ok(r)
}
fn cmp_if(i:i64,f:f64)->Result<i8,i32>{
    // int vs float, -1 less, 0 equal, 1 greater, 2 NaN. Exact for |i| <= 2**53, where
    // CPython (exact) and NumPy scalars (convert the int) agree; larger ints replay.
    const LIM:i64=1i64<<53;
    if i< -LIM || i>LIM {return Err(6)}
    if f.is_nan() {return Ok(2)}
    let g=i as f64;
    Ok(if g<f {-1} else if g>f {1} else {0})
}
fn sum_f(a:&Arr<f64>,c:&mut Ctx)->Result<f64,i32>{
    if a.len==0 {return Err(6)}  // Python's sum() of an empty array is the int 0
    let mut s=0.0f64;
    for j in 0..a.len {tick(c)?; s+=unsafe{*a.ptr.add(j)};}
    Ok(s)
}
fn sum_i(a:&Arr<i64>,c:&mut Ctx)->Result<i64,i32>{
    let mut s=0i64;
    for j in 0..a.len {tick(c)?; s=s.checked_add(unsafe{*a.ptr.add(j)}).ok_or(1)?;}
    Ok(s)
}
"""

# math functions with one float argument and a float result (CPython math_1 rules).
MATH_FLOAT = {
    'sqrt': 'sqrt',
    'exp': 'exp',
    'log': 'ln',
    'log2': 'log2',
    'log10': 'log10',
    'sin': 'sin',
    'cos': 'cos',
    'tan': 'tan',
    'asin': 'asin',
    'acos': 'acos',
    'atan': 'atan',
    'sinh': 'sinh',
    'cosh': 'cosh',
    'tanh': 'tanh',
}
MATH_OTHER = ('floor', 'ceil', 'trunc', 'fabs', 'isnan', 'isinf', 'isfinite', 'atan2', 'copysign', 'fmod', 'hypot')
MATH_CONSTANTS = {'pi', 'e', 'tau', 'inf', 'nan'}
# Builtins the compiler understands, checked again at call time in case they are rebound.
KNOWN_BUILTINS = {
    'range',
    'len',
    'float',
    'int',
    'abs',
    'min',
    'max',
    'pow',
    'sum',
    'round',
    'divmod',
    'enumerate',
    'bool',
    'ord',
    'chr',
}
NUMPY_MAKERS = {'zeros': 0, 'empty': 0, 'ones': 1, 'zeros_like': 0, 'empty_like': 0, 'ones_like': 1}
# random module functions reproduced exactly (same Mersenne Twister state as Python's random).
RANDOM_FUNCS = ('random', 'uniform', 'randint', 'randrange', 'gauss', 'choice')
# One-character str methods, exact for ASCII; other characters replay in Python.
CHAR_TESTS = {
    'isdigit': 'c_isdigit',
    'isalpha': 'c_isalpha',
    'isalnum': 'c_isalnum',
    'isspace': 'c_isspace',
    'isupper': 'c_isupper',
    'islower': 'c_islower',
}
CHAR_MAPS = {'lower': 'c_lower', 'upper': 'c_upper'}

HINTS = [
    (
        'List',
        'List literals are compiled only as [value] * n. Pass lists of numbers as arguments, '
        'or create one with [0.0] * n and fill it in a loop.',
    ),
    ('ListComp', "List comprehensions aren't compiled. Write a loop that fills a list made with [0.0] * n."),
    ('GeneratorExp', "Generator expressions aren't compiled. Write a loop that adds to a total."),
    ('Dict', "Dicts aren't compiled. Move dict work outside the decorated function."),
    ('Set', "Sets aren't compiled. Move set work outside the decorated function."),
    (
        'JoinedStr',
        'Building or formatting strings is not compiled. Strings can be read: len(s), s[i], for ch in s, '
        "ch == 'a', ch in 'aeiou', ord(ch), ch.isdigit().",
    ),
    ('Lambda', "Lambdas aren't compiled. Use a plain module-level function instead."),
    ('With', "`with` blocks aren't compiled. Keep I/O and context managers outside the decorated function."),
    ('Try', "`try` blocks aren't compiled. Errors in compiled code replay in Python, so handle them in the caller."),
]


def float_literal(value):
    if math.isnan(value):
        return '(f64::NAN)'
    if math.isinf(value):
        return '(f64::INFINITY)' if value > 0 else '(f64::NEG_INFINITY)'
    return f'({value!r}f64)'


def tuple_kind(kinds):
    return 'tuple(' + ','.join(kinds) + ')'


def tuple_parts(kind):
    return kind[6:-1].split(',') if kind.startswith('tuple(') else None


def rust_type(kind):
    if kind in ARRAY_INFO:
        return f'Arr<{ARRAY_INFO[kind][0]}>'
    parts = tuple_parts(kind)
    if parts:
        return '(' + ','.join(RUST_TYPE[p] for p in parts) + ')'
    return RUST_TYPE[kind]


def kind_of_value(value):
    """Native kind for a Python argument value, or None if it must run as Python."""
    t = type(value)
    if t is int:
        return 'i64' if -(2**63) <= value < 2**63 else None
    if t is float:
        return 'f64'
    if t is bool:
        return 'bool'
    if t is str:
        return 'str'
    if t is bytes:
        return 'bytes'
    if t is list:
        return list_kind(value)
    np = sys.modules.get('numpy')
    if np is not None and t is np.ndarray and value.ndim in (1, 2):
        if value.dtype == np.float64:
            return 'array' if value.ndim == 1 else 'array2'
        if value.dtype == np.int64:
            return 'iarray' if value.ndim == 1 else 'iarray2'
        return None
    return series_kind(value)


def list_kind(value):
    """'flist' for a list of floats, 'ilist' for a list of 64-bit ints, else None."""
    element_types = set(map(type, value))
    if not element_types or element_types == {float}:
        return 'flist'
    if element_types == {int} and min(value) >= -(2**63) and max(value) < 2**63:
        return 'ilist'
    return None


def series_kind(value):
    """'series' / 'iseries' for a NumPy-backed float64 / int64 pandas Series with index 0..n-1."""
    pd = sys.modules.get('pandas')
    np = sys.modules.get('numpy')
    if pd is None or np is None or not isinstance(value, pd.Series):
        return None
    index = value.index
    if type(index) is not pd.RangeIndex or index.start != 0 or index.step != 1:
        return None
    dtype = value.dtype
    if not isinstance(dtype, np.dtype):  # nullable and other extension dtypes
        return None
    if dtype == np.float64:
        return 'series'
    if dtype == np.int64:
        return 'iseries'
    return None


def default_kinds(function):
    """Kinds from annotations (unannotated parameters are int)."""
    kinds = {}
    for name, p in inspect.signature(function).parameters.items():
        annotation = p.annotation
        if annotation is inspect.Parameter.empty:
            kinds[name] = 'i64'
        elif (
            getattr(annotation, '__module__', '').startswith('pandas')
            and getattr(annotation, '__name__', '') == 'Series'
        ):
            kinds[name] = 'series'
        elif annotation == 'pd.Series' or annotation == 'pandas.Series':
            kinds[name] = 'series'
        else:
            try:
                kinds[name] = ANNOTATIONS[annotation]
            except (KeyError, TypeError):
                raise Unsupported(
                    f'{function.__name__}: unsupported annotation for {name}: {annotation!r} '
                    '(use int, float, bool, str, bytes, list[float], list[int], pandas.Series, '
                    'F64Array, I64Array, F64Array2 or I64Array2)'
                ) from None
    return kinds


def _resolve_callee(obj):
    """An @optimize wrapper or a plain function; returns the original function or None."""
    original = getattr(obj, 'python', None)
    if isinstance(original, types.FunctionType) and callable(getattr(obj, 'report', None)):
        return original
    if isinstance(obj, types.FunctionType):
        return obj
    return None


class Compiler:
    def __init__(self, function, max_iterations=10_000_000, _program=None, kinds=None):
        if hasattr(function, '__wrapped__'):
            raise Unsupported('Wrapped functions are not supported')
        self.function = function
        self.first_line = function.__code__.co_firstlineno
        self.tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        self.node = self.tree.body[0]
        if not isinstance(self.node, ast.FunctionDef) or function.__closure__:
            raise Unsupported('Only ordinary functions without closure capture are supported')
        self.signature = inspect.signature(function)
        for name, p in self.signature.parameters.items():
            if p.kind not in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                raise Unsupported(f'{function.__name__}: *args, **kwargs and keyword-only parameters are unsupported')
        self.params = list(self.signature.parameters)
        self.types = dict(kinds) if kinds is not None else default_kinds(function)
        self.globals = function.__globals__
        ns = function.__builtins__
        self.builtins_ns = ns if isinstance(ns, dict) else vars(ns)
        self.names = {p: f'v{i}' for i, p in enumerate(self.params)}
        self.assigned = set(self.names)
        self.local_names = set(self.names) | {
            n.id for n in ast.walk(self.node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        }
        targets = []
        for node in ast.walk(self.node):
            if isinstance(node, ast.Assign):
                targets += node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets.append(node.target)
        self.written = set()
        for t in targets:
            base = t.value if isinstance(t, ast.Subscript) else None
            if isinstance(base, ast.Subscript):  # a[i][j] = v
                base = base.value
            if isinstance(base, ast.Name) and self.types.get(base.id) in ARRAY_KINDS:
                self.written.add(base.id)
        returns = self.signature.return_annotation
        try:
            self.declared_return = RETURN_ANNOTATIONS.get(returns)
        except TypeError:
            self.declared_return = None
        self.local_arrays = set()
        self.used_builtins = set()
        self.counter = 0
        self.loop_depth = 0
        self.parallel = None  # info about the prange loop being compiled, if any
        self.return_type = None
        self.max_iterations = max_iterations
        # Shared by every function compiled into one library.
        self.program = (
            _program
            if _program is not None
            else {
                'functions': [],
                'index': {},
                'stack': [],
                'guards': {},
                'code_guards': {},
                'callees': [],
                'recursive': {},
                'next': 0,
                'parallel': False,
                'random': False,
            }
        )

    # ------------------------------------------------------------ diagnostics
    def fail(self, node, message, hint=None):
        line = self.first_line + getattr(node, 'lineno', 1) - 1
        try:
            snippet = ast.unparse(node)
        except Exception:
            snippet = type(node).__name__
        if len(snippet) > 60:
            snippet = snippet[:57] + '...'
        if hint is None:
            for sub in ast.walk(node):
                hint = next((h for name, h in HINTS if type(sub).__name__ == name), None)
                if hint:
                    break
            if hint is None and isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
                hint = dict(HINTS)['JoinedStr']
        text = f'{self.function.__name__}, line {line}: {message}: `{snippet}`'
        raise Unsupported(text + (f'\n  hint: {hint}' if hint else ''))

    # ---------------------------------------------------------------- helpers
    def name(self, name):
        if name not in self.names:
            self.names[name] = f'v{len(self.names)}'
        return self.names[name]

    def temp(self, prefix='t'):
        self.counter += 1
        return f'{prefix}{self.counter}'

    def guard(self, name, expected):
        """Record that global/builtin `name` must still be `expected` at call time."""
        key = (id(self.globals), name)
        self.program['guards'][key] = (self.globals, self.builtins_ns, name, expected)

    def lookup(self, name):
        """Current global/builtin value of a non-local name (or a sentinel)."""
        return self.globals.get(name, self.builtins_ns.get(name, _MISSING))

    def builtin(self, name):
        if name in self.local_names:
            raise Unsupported(f'{self.function.__name__}: local {name} shadowing is unsupported')
        self.used_builtins.add(name)
        self.guard(name, getattr(builtins, name))

    def is_builtin(self, name):
        return name in KNOWN_BUILTINS and name not in self.local_names and self.lookup(name) is getattr(builtins, name)

    def is_prange(self, func):
        if not isinstance(func, ast.Name) or func.id in self.local_names:
            return False
        obj = self.lookup(func.id)
        if obj is prange:
            self.guard(func.id, obj)
            return True
        return False

    def numpy_module(self, node):
        """The numpy module if `node` is a global name bound to it, else None."""
        np = sys.modules.get('numpy')
        if (
            np is not None
            and isinstance(node, ast.Name)
            and node.id not in self.local_names
            and self.lookup(node.id) is np
        ):
            self.guard(node.id, np)
            return np
        return None

    def math_name(self, func):
        """'sqrt' for math.sqrt(...) or a `from math import sqrt` name; else None."""
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id not in self.local_names
            and self.lookup(func.value.id) is math
        ):
            self.guard(func.value.id, math)
            return func.attr
        if isinstance(func, ast.Name) and func.id not in self.local_names:
            obj = self.lookup(func.id)
            for attr in (*MATH_FLOAT, *MATH_OTHER):
                if obj is getattr(math, attr, None):
                    self.guard(func.id, obj)
                    return attr
        return None

    def expect(self, node, kind):
        value, actual = self.expr(node)
        if actual == 'bool' and kind == 'i64':
            return f'({value} as i64)'
        if actual != kind:
            self.fail(node, f'expected {kind}, got {actual}')
        return value

    def scalar(self, node):
        """int or float value; bools are promoted to int like Python does in arithmetic."""
        value, kind = self.expr(node)
        if kind == 'bool':
            return f'({value} as i64)', 'i64'
        if kind not in ('i64', 'f64'):
            self.fail(node, 'expected an int or float value')
        return value, kind

    def as_float(self, node):
        value, kind = self.scalar(node)
        return f'({value} as f64)' if kind == 'i64' else value

    # ----------------------------------------------------------------- arrays
    def array(self, node):
        """(name, kind) of an array parameter or local array."""
        if isinstance(node, ast.Name) and node.id in self.assigned and self.types.get(node.id) in ARRAY_KINDS:
            return node.id, self.types[node.id]
        if isinstance(node, ast.List):
            self.fail(node, 'list literals are not compiled here')  # the List hint explains [value] * n
        self.fail(
            node,
            'expected an array, list, str, bytes or pandas Series argument, or a local array or list',
            'Create local arrays with np.zeros(n) and local lists with [0.0] * n.',
        )

    def aref(self, name):
        if self.parallel is not None and name in self.parallel['arrays']:
            return f'&*{self.name(name)}_r'
        return f'&*{self.name(name)}' if name in self.params else f'&{self.name(name)}'

    def amut(self, name, node):
        self.check_writable(name, node)
        if self.parallel is not None:
            self.fail(
                node,
                'inside prange, arrays can only be written with out[i] = ...',
                'Called functions cannot write arrays inside prange.',
            )
        if name in self.params:
            if name not in self.written:
                self.written.add(name)
            return f'&mut *{self.name(name)}'
        return f'&mut {self.name(name)}'

    def check_writable(self, name, node):
        kind = self.types.get(name)
        if kind in READ_ONLY_KINDS:
            what = {'series': 'pandas Series', 'iseries': 'pandas Series', 'str': 'str', 'bytes': 'bytes'}[kind]
            self.fail(
                node,
                f'{name!r} is a {what}, which compiled code can only read',
                'Write results into a NumPy array (np.zeros(n)) or a list ([0.0] * n) instead.',
            )

    def indices(self, sub):
        """(array name, kind, [index exprs]) for a[i], a[i, j] or a[i][j]."""
        node = sub.slice
        if isinstance(sub.value, ast.Subscript) and not isinstance(node, ast.Tuple):
            name, kind = self.array(sub.value.value)  # a[i][j]
            self.check_parallel_index(sub, name, sub.value.slice)
            if ARRAY_INFO[kind][1] != 2:
                self.fail(sub, 'a[i][j] needs a 2-D array')
            return name, kind, [self.expect(sub.value.slice, 'i64'), self.expect(node, 'i64')]
        name, kind = self.array(sub.value)
        self.check_parallel_index(sub, name, node)
        dims = ARRAY_INFO[kind][1]
        if isinstance(node, ast.Slice) or (
            isinstance(node, ast.Tuple) and any(isinstance(e, ast.Slice) for e in node.elts)
        ):
            self.fail(
                sub,
                'array slices are not compiled',
                'Loop over indices instead of slicing, e.g. for i in range(start, stop): ... a[i] ...',
            )
        parts = node.elts if isinstance(node, ast.Tuple) else [node]
        if len(parts) != dims:
            self.fail(sub, f'this array has {dims} dimension(s); index it with {dims} index(es)')
        return name, kind, [self.expect(p, 'i64') for p in parts]

    def check_parallel_index(self, sub, name, index):
        """Inside prange, arrays that the loop writes may only be touched at the loop index."""
        if self.parallel is None or name not in self.parallel['writes']:
            return
        first = index.elts[0] if isinstance(index, ast.Tuple) and index.elts else index
        if not (isinstance(first, ast.Name) and first.id == self.parallel['var']):
            self.fail(
                sub,
                f'inside prange, {name!r} is written, so it may only be used at the loop index '
                f'({name}[{self.parallel["var"]}])',
                'Other iterations run at the same time, so reading their elements would not match Python.',
            )

    def element(self, sub, positional=False):
        name, kind, idx = self.indices(sub)
        if len(idx) == 2:
            return f'at2({self.aref(name)},{",".join(idx)})?', READ_KIND[ELEMENT[kind]]
        # s[i] on a Series looks up the label i (index 0..n-1); s.iloc[i] is positional.
        fn = 'at_nn' if kind in ('series', 'iseries') and not positional else 'at'
        return self.read(f'{fn}({self.aref(name)},{idx[0]})?', kind)

    def read(self, value, kind):
        """An element read from a sequence of this kind, as (rust, value kind)."""
        element = ELEMENT[kind]
        if element == 'u8':
            return f'({value} as i64)', 'i64'
        return value, READ_KIND[element]

    def list_repeat(self, n):
        """[value] * n or n * [value]: a new list of ints or floats."""
        listed, count = (n.left, n.right) if isinstance(n.left, ast.List) else (n.right, n.left)
        if isinstance(count, ast.List) or len(listed.elts) != 1:
            self.fail(n, 'only [value] * n lists are compiled', 'Write [0.0] * n, then fill it in a loop.')
        fill, kind = self.expr(listed.elts[0])
        if kind not in ('i64', 'f64'):
            self.fail(n, 'lists made with [value] * n must hold ints or floats')
        size = self.expect(count, 'i64')
        if self.parallel is not None:
            self.fail(n, 'lists cannot be created inside prange', 'Create them before the loop.')
        element_kind = 'flist' if kind == 'f64' else 'ilist'
        return f'Arr::<{kind}>::owned({size},1i64,{fill})?', element_kind

    def shape(self, node):
        """a.shape -> tuple kind value."""
        name, kind = self.array(node.value)
        ref = self.aref(name)
        if ARRAY_INFO[kind][1] == 1:
            return f'(({ref}).len as i64,)', 'tuple(i64)'
        return f'(({ref}).rows as i64, ({ref}).cols as i64)', 'tuple(i64,i64)'

    def numpy_maker(self, n):
        """np.zeros(n) / np.empty((r, c)) / np.ones(n, dtype=np.int64) / np.zeros_like(a) -> new array."""
        np = self.numpy_module(n.func.value)
        attr = n.func.attr
        fill = NUMPY_MAKERS[attr]
        dtype = None
        for kw in n.keywords:
            if kw.arg != 'dtype':
                self.fail(n, f'only the dtype= keyword is supported for np.{attr}')
            target = kw.value
            if isinstance(target, ast.Name) and target.id in ('float', 'int') and self.is_builtin(target.id):
                dtype = 'f64' if target.id == 'float' else 'i64'
                self.builtin(target.id)
            elif (
                isinstance(target, ast.Attribute)
                and self.numpy_module(target.value) is np
                and target.attr in ('float64', 'int64', 'double')
            ):
                dtype = 'i64' if target.attr == 'int64' else 'f64'
            else:
                self.fail(kw.value, 'dtype must be float, int, np.float64 or np.int64')
        if len(n.args) != 1:
            self.fail(n, f'np.{attr} takes one shape argument here')
        if attr.endswith('_like'):
            name, kind = self.array(n.args[0])
            if kind not in NUMPY_KINDS + LIST_KINDS:
                self.fail(n, f'np.{attr}() needs a NumPy array or a list of numbers')
            element = dtype or ELEMENT[kind]
            ref = self.aref(name)
            rows, cols = f'(({ref}).rows as i64)', f'(({ref}).cols as i64)'
            dims = ARRAY_INFO[kind][1]
        else:
            element = dtype or 'f64'
            shape = n.args[0]
            if isinstance(shape, ast.Tuple) and len(shape.elts) == 2:
                rows, cols = self.expect(shape.elts[0], 'i64'), self.expect(shape.elts[1], 'i64')
                dims = 2
            elif isinstance(shape, ast.Tuple) and len(shape.elts) == 1:
                rows, cols, dims = self.expect(shape.elts[0], 'i64'), '1i64', 1
            elif isinstance(shape, ast.Tuple):
                self.fail(n, 'only 1-D and 2-D arrays are compiled')
            else:
                rows, cols, dims = self.expect(shape, 'i64'), '1i64', 1
        kind = {('f64', 1): 'array', ('i64', 1): 'iarray', ('f64', 2): 'array2', ('i64', 2): 'iarray2'}[element, dims]
        literal = ('1.0' if fill else '0.0') if element == 'f64' else ('1' if fill else '0')
        return f'Arr::<{element}>::owned({rows},{cols},{literal}{element})?', kind

    # ------------------------------------------------------------ expressions
    def expr(self, n):
        if isinstance(n, ast.Constant):
            if type(n.value) is bool:
                return ('true' if n.value else 'false'), 'bool'
            if type(n.value) is int and -(2**63) <= n.value < 2**63:
                return f'({n.value}i64)', 'i64'
            if type(n.value) is float and math.isfinite(n.value):
                return float_literal(n.value), 'f64'
            if type(n.value) is str and len(n.value) == 1:
                return f'({ord(n.value)}u32)', 'char'
            self.fail(n, 'only int, float, bool and one-character str constants are supported')
        if isinstance(n, ast.Name):
            if self.parallel is not None and n.id in self.parallel['reductions']:
                self.fail(n, f'inside prange, the total {n.id!r} can only be added to, not read')
            if n.id in self.assigned:
                return self.name(n.id), self.types[n.id]
            if n.id not in self.local_names:
                return self.global_constant(n)
            self.fail(
                n,
                'variable may be used before it is assigned',
                'Give it a value before the loop or if statement, for example total = 0.0.',
            )
        if isinstance(n, ast.Attribute):
            if (
                isinstance(n.value, ast.Name)
                and n.value.id not in self.local_names
                and self.lookup(n.value.id) is math
                and n.attr in MATH_CONSTANTS
            ):
                self.guard(n.value.id, math)
                return float_literal(getattr(math, n.attr)), 'f64'
            if n.attr == 'shape':
                return self.shape(n)
            if n.attr == 'size':
                name, _ = self.array(n.value)
                return f'(({self.aref(name)}).len as i64)', 'i64'
            self.fail(
                n,
                'unsupported attribute',
                'Only math.* constants and functions, array .shape and '
                '.size, and np.zeros/np.empty/np.ones are compiled.',
            )
        if isinstance(n, ast.Subscript):
            if isinstance(n.value, ast.Attribute) and n.value.attr == 'shape':
                value, kind = self.shape(n.value)
                if not (isinstance(n.slice, ast.Constant) and type(n.slice.value) is int):
                    self.fail(n, 'index .shape with a constant, like a.shape[0]')
                parts = tuple_parts(kind)
                i = n.slice.value
                if not -len(parts) <= i < len(parts):
                    self.fail(n, 'shape index out of range')
                return f'({value}.{i % len(parts)})', 'i64'
            if (
                isinstance(n.value, ast.Attribute)
                and n.value.attr == 'iloc'
                and isinstance(n.value.value, ast.Name)
                and self.types.get(n.value.value.id) in ('series', 'iseries')
            ):
                plain = ast.copy_location(ast.Subscript(value=n.value.value, slice=n.slice, ctx=ast.Load()), n)
                return self.element(plain, positional=True)
            return self.element(n)
        if isinstance(n, ast.UnaryOp):
            if isinstance(n.op, ast.Not):
                return f'(!({self.condition(n.operand)}))', 'bool'
            if isinstance(n.op, ast.USub):
                # AST represents -2**63 as unary minus applied to the positive literal.
                if isinstance(n.operand, ast.Constant) and type(n.operand.value) is int and n.operand.value == 2**63:
                    return 'i64::MIN', 'i64'
                value, kind = self.scalar(n.operand)
                if kind == 'i64':
                    return f'({value}).checked_neg().ok_or(1)?', kind
                return f'(-({value}))', kind
            if isinstance(n.op, ast.UAdd):
                value, kind = self.scalar(n.operand)
                return f'({value})', kind
            if isinstance(n.op, ast.Invert):
                return f'(!({self.expect(n.operand, "i64")}))', 'i64'
        if isinstance(n, ast.BinOp):
            if isinstance(n.op, ast.Mult) and (isinstance(n.left, ast.List) or isinstance(n.right, ast.List)):
                return self.list_repeat(n)
            return self.binop(n, n.left, n.op, n.right)
        if isinstance(n, ast.Compare):
            return f'({self.condition(n)})', 'bool'
        if isinstance(n, ast.BoolOp):
            values = [self.expr(v) for v in n.values]
            if any(k != 'bool' for _, k in values):
                self.fail(
                    n,
                    '`and` / `or` between non-bool values returns one of the values in Python',
                    'Use it inside an if/while condition, or compare first: (a > 0) and (b > 0).',
                )
            op = ' && ' if isinstance(n.op, ast.And) else ' || '
            return '(' + op.join(f'({v})' for v, _ in values) + ')', 'bool'
        if isinstance(n, ast.IfExp):
            cond = self.condition(n.test)
            a, ta = self.expr(n.body)
            b, tb = self.expr(n.orelse)
            if ta != tb or ta not in VALUES:
                self.fail(
                    n,
                    'both branches of `x if c else y` must have the same type',
                    'Use 0.0 instead of 0 (or the other way round) so both branches match.',
                )
            return f'(if {cond} {{ {a} }} else {{ {b} }})', ta
        if isinstance(n, ast.Tuple):
            values = [self.expr(e) for e in n.elts]
            if not values or any(k not in SCALARS for _, k in values):
                self.fail(n, 'only tuples of int, float and bool values are compiled')
            return '(' + ', '.join(v for v, _ in values) + (',' if len(values) == 1 else '') + ')', tuple_kind(
                [k for _, k in values]
            )
        if isinstance(n, ast.Call):
            return self.call(n)
        self.fail(n, 'unsupported expression')

    def global_constant(self, n):
        value = self.lookup(n.id)
        if type(value) is bool:
            self.guard(n.id, value)
            return ('true' if value else 'false'), 'bool'
        if type(value) is int and -(2**63) <= value < 2**63:
            self.guard(n.id, value)
            return f'({value}i64)', 'i64'
        if type(value) is float:
            self.guard(n.id, value)
            return float_literal(value), 'f64'
        if value is _MISSING:
            self.fail(n, f'name {n.id!r} is not defined')
        self.fail(n, f'global {n.id!r} must be an int, float or bool constant')

    def binop(self, n, left, op, right):
        a, ta = self.expr(left)
        b, tb = self.expr(right)
        for k, side in ((ta, left), (tb, right)):
            if k not in SCALARS:
                self.fail(
                    side,
                    'only int, float and bool values can be used in arithmetic',
                    "Read array elements with a[i]; arrays themselves aren't combined.",
                )
        if ta == tb == 'bool' and isinstance(op, (ast.BitAnd, ast.BitOr, ast.BitXor)):
            return f'(({a}) {({ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^"})[type(op)]} ({b}))', 'bool'
        if ta == 'bool':
            a, ta = f'({a} as i64)', 'i64'
        if tb == 'bool':
            b, tb = f'({b} as i64)', 'i64'
        if ta == tb == 'i64':
            checked = {ast.Add: 'checked_add', ast.Sub: 'checked_sub', ast.Mult: 'checked_mul'}.get(type(op))
            if checked:
                return f'({a}).{checked}({b}).ok_or(1)?', 'i64'
            helper = {
                ast.FloorDiv: 'floor_div',
                ast.Mod: 'modulo',
                ast.Pow: 'int_pow',
                ast.LShift: 'shift_left',
                ast.RShift: 'shift_right',
            }.get(type(op))
            if helper:
                return f'{helper}({a},{b})?', 'i64'
            bitwise = {ast.BitAnd: '&', ast.BitOr: '|', ast.BitXor: '^'}.get(type(op))
            if bitwise:
                return f'(({a}) {bitwise} ({b}))', 'i64'
            if isinstance(op, ast.Div):
                return f'int_true_div({a},{b})?', 'f64'
        else:
            a = f'({a} as f64)' if ta == 'i64' else a
            b = f'({b} as f64)' if tb == 'i64' else b
            symbol = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*'}.get(type(op))
            if symbol:
                return f'(({a}) {symbol} ({b}))', 'f64'
            helper = {
                ast.Div: 'divide',
                ast.FloorDiv: 'float_floor_div',
                ast.Mod: 'float_mod',
                ast.Pow: 'float_pow',
            }.get(type(op))
            if helper:
                return f'{helper}({a},{b})?', 'f64'
        self.fail(n, f'unsupported operator for {ta} and {tb}')

    def call(self, n):
        func, args = n.func, n.args
        if isinstance(func, ast.Attribute) and func.attr in NUMPY_MAKERS and self.numpy_module(func.value) is not None:
            return self.numpy_maker(n)
        if n.keywords and not (
            isinstance(func, ast.Name) and not self.is_builtin(func.id) and self.math_name(func) is None
        ):
            self.fail(
                n,
                'keyword arguments are only supported in calls to your own functions and np.zeros(..., dtype=...)',
                'Pass the arguments by position.',
            )
        math_attr = self.math_name(func)
        if math_attr is not None:
            return self.math_call(n, math_attr, args)
        random_attr = self.random_name(func)
        if random_attr is not None:
            if n.keywords:
                self.fail(n, 'pass the arguments of random functions by position')
            return self.random_call(n, random_attr, args)
        if isinstance(func, ast.Attribute) and (func.attr in CHAR_TESTS or func.attr in CHAR_MAPS):
            value, kind = self.expr(func.value)
            if kind != 'char' or args:
                self.fail(
                    n,
                    f'.{func.attr}() is compiled only on one character, like s[i].{func.attr}()',
                    'Loop over the string and test each character.',
                )
            if func.attr in CHAR_TESTS:
                return f'{CHAR_TESTS[func.attr]}({value})?', 'bool'
            return f'{CHAR_MAPS[func.attr]}({value})?', 'char'
        if not isinstance(func, ast.Name):
            self.fail(
                n,
                'only calls to builtins, math functions, np.zeros/empty/ones and plain functions are supported',
                "Method calls and calls through objects aren't compiled.",
            )
        name = func.id
        if name in self.local_names:
            self.fail(n, f'{name!r} is a local variable, not a function')
        if self.is_builtin(name):
            return self.builtin_call(n, name, args)
        if name == 'print' and self.lookup(name) is builtins.print:
            self.fail(
                n,
                'print() is not compiled',
                'Remove print() from the hot function, or print the returned value afterwards.',
            )
        callee = _resolve_callee(self.lookup(name))
        if callee is not None:
            return self.user_call(n, name, callee)
        self.fail(n, f'call to {name!r} is unsupported')

    def random_name(self, func):
        """'randint' for random.randint(...) or a `from random import randint` name; else None.

        Only the functions of Python's shared random generator are recognised, so the
        compiled code can continue exactly the same random sequence.
        """
        module = _random
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id not in self.local_names
            and self.lookup(func.value.id) is module
            and func.attr in RANDOM_FUNCS
        ):
            self.guard(func.value.id, module)
            attr = func.attr
        elif isinstance(func, ast.Name) and func.id not in self.local_names:
            obj = self.lookup(func.id)
            attr = next((a for a in RANDOM_FUNCS if getattr(module, a, None) is obj), None)
            if attr is None:
                return None
            self.guard(func.id, obj)
        else:
            return None
        current = getattr(module, attr, None)
        if getattr(current, '__self__', None) is not getattr(module, '_inst', object()):
            return None  # random.<attr> was replaced by something else
        # Replacing random.<attr> later sends calls back to Python.
        self.program['guards'][(id(module.__dict__), attr)] = (module.__dict__, {}, attr, current)
        return attr

    def random_call(self, n, attr, args):
        if self.parallel is not None:
            self.fail(
                n,
                'random numbers are not compiled inside prange',
                'Python draws them in order, one after another; use a normal range loop.',
            )
        self.program['random'] = True
        count = len(args)
        if attr == 'random' and count == 0:
            return 'rnd_random(fuel)?', 'f64'
        if attr == 'uniform' and count == 2:
            a, ta = self.scalar(args[0])
            b, tb = self.scalar(args[1])
            t = self.temp('r')
            if ta == tb == 'i64':
                # a + (b - a) * random(): the int difference is converted to float once.
                return (
                    f'({{ let {t}a: i64 = {a}; let {t}b: i64 = {b}; let {t}d = ({t}b).checked_sub({t}a).ok_or(1)?; '
                    f'({t}a as f64) + ({t}d as f64) * rnd_random(fuel)? }})'
                ), 'f64'
            a = f'({a} as f64)' if ta == 'i64' else a
            b = f'({b} as f64)' if tb == 'i64' else b
            return f'({{ let {t}a: f64 = {a}; let {t}b: f64 = {b}; {t}a + ({t}b - {t}a) * rnd_random(fuel)? }})', 'f64'
        if attr == 'randint' and count == 2:
            a, b = self.expect(args[0], 'i64'), self.expect(args[1], 'i64')
            t = self.temp('r')
            return (
                f'({{ let {t}a: i64 = {a}; let {t}b: i64 = {b}; '
                f'rnd_range(fuel,{t}a,({t}b).checked_add(1).ok_or(1)?)? }})'
            ), 'i64'
        if attr == 'randrange' and count == 1:
            return f'rnd_below(fuel,{self.expect(args[0], "i64")})?', 'i64'
        if attr == 'randrange' and count == 2:
            a, b = self.expect(args[0], 'i64'), self.expect(args[1], 'i64')
            t = self.temp('r')
            return f'({{ let {t}a: i64 = {a}; let {t}b: i64 = {b}; rnd_range(fuel,{t}a,{t}b)? }})', 'i64'
        if attr == 'gauss' and count == 2:
            mu, sigma = self.as_float(args[0]), self.as_float(args[1])
            t = self.temp('r')
            return f'({{ let {t}m: f64 = {mu}; let {t}s: f64 = {sigma}; rnd_gauss(fuel,{t}m,{t}s)? }})', 'f64'
        if attr == 'choice' and count == 1:
            name, kind = self.array(args[0])
            if ARRAY_INFO[kind][1] != 1:
                self.fail(n, 'random.choice() of a 2-D array picks a row; it is not compiled')
            ref = self.aref(name)
            t = self.temp('r')
            # Python reads seq[i] for a random i; for a Series that is the label i, which is position i here.
            return self.read(
                f'({{ let {t}n = ({ref}).len as i64; if {t}n == 0 {{ return Err(3) }} '
                f'at({ref}, rnd_below(fuel,{t}n)?)? }})',
                kind,
            )
        forms = {
            'random': 'random.random()',
            'uniform': 'random.uniform(a, b)',
            'randint': 'random.randint(a, b)',
            'randrange': 'random.randrange(stop) or random.randrange(start, stop)',
            'gauss': 'random.gauss(mu, sigma)',
            'choice': 'random.choice(seq)',
        }
        self.fail(n, f'only {forms[attr]} is compiled')

    def math_call(self, n, attr, args):
        if attr == 'hypot':
            self.fail(
                n,
                'math.hypot is not compiled: CPython uses its own algorithm that can differ from '
                'the C library in the last digit',
                'If that difference is fine for you, write math.sqrt(x * x + y * y).',
            )
        if attr in ('atan2', 'copysign', 'fmod'):
            if len(args) != 2:
                self.fail(n, f'math.{attr} takes two arguments')
            x, y = self.as_float(args[0]), self.as_float(args[1])
            if attr == 'atan2':
                return f'm_atan2({x},{y})', 'f64'
            if attr == 'copysign':
                return f'(({x}).copysign({y}))', 'f64'
            return f'm_fmod({x},{y})?', 'f64'
        if len(args) != 1:
            self.fail(n, f'math.{attr} takes one argument here')
        if attr in MATH_FLOAT:
            x = self.as_float(args[0])
            t = self.temp('m')
            return f'({{ let {t}: f64 = {x}; math1({t},{t}.{MATH_FLOAT[attr]}())? }})', 'f64'
        if attr == 'fabs':
            return f'(({self.as_float(args[0])}).abs())', 'f64'
        if attr in ('floor', 'ceil', 'trunc'):
            value, kind = self.scalar(args[0])
            if kind == 'i64':
                return f'({value})', 'i64'
            return f'to_int(({value}).{attr}())?', 'i64'
        if attr in ('isnan', 'isinf', 'isfinite'):
            x = self.as_float(args[0])
            method = {'isnan': 'is_nan', 'isinf': 'is_infinite', 'isfinite': 'is_finite'}[attr]
            return f'(({x}).{method}())', 'bool'
        self.fail(n, f'math.{attr} is not compiled')

    def builtin_call(self, n, name, args):
        self.builtin(name)
        if name == 'len' and len(args) == 1:
            arr, _ = self.array(args[0])
            return f'(({self.aref(arr)}).rows as i64)', 'i64'
        if name == 'float' and len(args) == 1:
            return self.as_float(args[0]), 'f64'
        if name == 'ord' and len(args) == 1:
            value, kind = self.expr(args[0])
            if kind != 'char':
                self.fail(n, 'ord() needs one character, like ord(s[i])')
            return f'({value} as i64)', 'i64'
        if name == 'chr' and len(args) == 1:
            return f'to_char({self.expect(args[0], "i64")})?', 'char'
        if name == 'int' and len(args) == 1:
            value, kind = self.scalar(args[0])
            return (f'({value})', 'i64') if kind == 'i64' else (f'to_int({value})?', 'i64')
        if name == 'bool' and len(args) == 1:
            value, kind = self.expr(args[0])
            if kind == 'bool':
                return value, 'bool'
            return f'({self.condition(args[0])})', 'bool'
        if name == 'abs' and len(args) == 1:
            value, kind = self.scalar(args[0])
            if kind == 'i64':
                return f'({value}).checked_abs().ok_or(1)?', 'i64'
            return f'(({value}).abs())', 'f64'
        if name == 'round':
            if len(args) != 1:
                self.fail(
                    n,
                    'round(x, ndigits) is not compiled',
                    'round(x) is supported; round to digits outside the decorated function.',
                )
            value, kind = self.scalar(args[0])
            return (f'({value})', 'i64') if kind == 'i64' else (f'round_int({value})?', 'i64')
        if name in ('min', 'max') and len(args) >= 2:
            values = [self.scalar(a) for a in args]
            kinds = {k for _, k in values}
            if len(kinds) != 1:
                # Python returns whichever argument wins, keeping its own type.
                self.fail(
                    n,
                    f'{name}() arguments must all be int or all be float',
                    'Write 0.0 instead of 0 (or float(x)) so every argument is a float.',
                )
            kind = kinds.pop()
            helper = f'{name}_{"i" if kind == "i64" else "f"}'
            out = values[0][0]
            for value, _ in values[1:]:
                out = f'{helper}({out},{value})'
            return out, kind
        if name == 'pow' and len(args) == 2:
            return self.binop(n, args[0], ast.Pow(), args[1])
        if name == 'sum' and len(args) == 1:
            arr, kind = self.array(args[0])
            if ARRAY_INFO[kind][1] != 1:
                self.fail(n, 'sum() of a 2-D array returns an array; it is not compiled')
            if kind in ('flist', 'series', 'str', 'bytes'):
                # Python 3.12+ adds Python floats with extra-precise (Neumaier) summation.
                self.fail(
                    n,
                    f'sum() of this {kind} is not compiled',
                    'Add the values in a loop (total += x), or pass a NumPy array.',
                )
            return (
                f'sum_f({self.aref(arr)},fuel)?' if ELEMENT[kind] == 'f64' else f'sum_i({self.aref(arr)},fuel)?'
            ), ELEMENT[kind]
        if name == 'divmod' and len(args) == 2:
            a, ta = self.scalar(args[0])
            b, tb = self.scalar(args[1])
            t1, t2 = self.temp('d'), self.temp('d')
            if ta == tb == 'i64':
                return (
                    f'({{ let {t1}: i64 = {a}; let {t2}: i64 = {b}; (floor_div({t1},{t2})?, modulo({t1},{t2})?) }})'
                ), 'tuple(i64,i64)'
            a = f'({a} as f64)' if ta == 'i64' else a
            b = f'({b} as f64)' if tb == 'i64' else b
            return (
                f'({{ let {t1}: f64 = {a}; let {t2}: f64 = {b}; '
                f'(float_floor_div({t1},{t2})?, float_mod({t1},{t2})?) }})'
            ), 'tuple(f64,f64)'
        self.fail(n, f'unsupported use of {name}()')

    def register(self, rust_name, key):
        """Make this function callable (by recursive calls) before its body is compiled."""
        params = [self.types[p] for p in self.params]
        self.program['index'][key] = (self.function, rust_name, params, self.declared_return, set(self.written))

    def user_call(self, n, name, callee):
        program = self.program
        sig = inspect.signature(callee)
        cparams = list(sig.parameters)
        # Bind positional, keyword and default arguments like Python does.
        bound = {}
        if len(n.args) > len(cparams):
            self.fail(n, f'{name}() takes {len(cparams)} arguments')
        for p, arg in zip(cparams, n.args):
            bound[p] = arg
        for kw in n.keywords:
            if kw.arg is None or kw.arg not in cparams or kw.arg in bound:
                self.fail(n, f'bad keyword argument for {name}()')
            bound[kw.arg] = kw.value
        values, kinds = [], []
        for p in cparams:
            if p in bound:
                if (
                    isinstance(bound[p], ast.Name)
                    and self.types.get(bound[p].id) in ARRAY_KINDS
                    and bound[p].id in self.assigned
                ):
                    values.append(('array', bound[p].id))
                    kinds.append(self.types[bound[p].id])
                    continue
                value, kind = self.expr(bound[p])
                if kind not in VALUES:
                    self.fail(bound[p], f'arguments to {name}() must be numbers, characters, arrays or lists')
                values.append(('value', value))
                kinds.append(kind)
            else:
                default = sig.parameters[p].default
                if default is inspect.Parameter.empty:
                    self.fail(n, f'{name}() is missing argument {p!r}')
                value, kind = self.expr(ast.Constant(value=default))
                values.append(('value', value))
                kinds.append(kind)
        key = (id(callee), tuple(kinds))
        if callee in program['stack'] and key in program['index']:
            entry = program['index'][key]
            if entry[3] in (None, 'none'):
                self.fail(n, f'recursive call to {name!r}: annotate its return type (-> int, -> float or -> bool)')
            program['recursive'][key] = entry[3]
        elif key not in program['index']:
            program['stack'].append(callee)
            try:
                try:
                    sub = Compiler(callee, self.max_iterations, _program=program, kinds=dict(zip(cparams, kinds)))
                except Unsupported as exc:
                    raise Unsupported(f'called function {name!r}: {exc}') from None
                except (OSError, TypeError) as exc:
                    raise Unsupported(f'called function {name!r}: source unavailable ({exc})') from None
                rust_name = f'k{program["next"]}'
                program['next'] += 1
                sub.register(rust_name, key)
                text = sub.function_text(rust_name, key)
            finally:
                program['stack'].pop()
            if sub.return_type == 'none':
                self.fail(n, f'called function {name!r} does not return a value')
            program['index'][key] = (callee, rust_name, kinds, sub.return_type, set(sub.written))
            program['functions'].append(text)
            if callee.__qualname__ not in program['callees']:
                program['callees'].append(callee.__qualname__)
            program['code_guards'][id(callee)] = (callee, callee.__code__, callee.__defaults__)
        _, rust_name, params, return_type, written = program['index'][key]
        self.guard(name, self.lookup(name))
        rendered = []
        for (what, value), kind, p in zip(values, params, cparams):
            if what == 'array':
                rendered.append(self.amut(value, n) if p in written else self.aref(value))
            else:
                rendered.append(value)
        # Each call spends one unit of the iteration budget, so runaway recursion stops too.
        return f'({{ tick(fuel)?; {rust_name}({", ".join(rendered + ["fuel", "depth+1"])})? }})', return_type

    # ------------------------------------------------------------- conditions
    def condition(self, n):
        if isinstance(n, ast.Constant) and type(n.value) is bool:
            return 'true' if n.value else 'false'
        if isinstance(n, ast.BoolOp) and isinstance(n.op, (ast.And, ast.Or)):
            op = ' && ' if isinstance(n.op, ast.And) else ' || '
            return '(' + op.join(f'({self.condition(v)})' for v in n.values) + ')'
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            return f'!({self.condition(n.operand)})'
        ops = {ast.Lt: '<', ast.LtE: '<=', ast.Gt: '>', ast.GtE: '>=', ast.Eq: '==', ast.NotEq: '!='}
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], (ast.In, ast.NotIn)):
            return self.membership(n)
        if isinstance(n, ast.Compare) and all(type(op) in ops for op in n.ops):
            # Chained comparisons: operands are pure, so evaluating one that Python
            # would have short-circuited can at most cause a guard failure and replay.
            operands = [self.compare_operand(x) for x in [n.left, *n.comparators]]
            parts = []
            for (a, ta, ca), op, (b, tb, cb) in zip(operands, n.ops, operands[1:]):
                if ta == 'bool':
                    a, ta = f'({a} as i64)', 'i64'
                if tb == 'bool':
                    b, tb = f'({b} as i64)', 'i64'
                if ta != tb:
                    # int literal vs float: exact when |literal| <= 2**53.
                    if ta == 'i64' and tb == 'f64' and ca is not None:
                        a, ta = f'({float(ca)!r}f64)', 'f64'
                    elif tb == 'i64' and ta == 'f64' and cb is not None:
                        b, tb = f'({float(cb)!r}f64)', 'f64'
                if {ta, tb} == {'i64', 'f64'}:
                    c = f'cmp_if({a},{b})?' if ta == 'i64' else f'cmp_if({b},{a})?'
                    flip = {ast.Lt: ast.Gt, ast.LtE: ast.GtE, ast.Gt: ast.Lt, ast.GtE: ast.LtE}
                    kind = type(op) if ta == 'i64' else flip.get(type(op), type(op))
                    test = {
                        ast.Lt: 'c==-1',
                        ast.LtE: 'c==-1 || c==0',
                        ast.Gt: 'c==1',
                        ast.GtE: 'c==1 || c==0',
                        ast.Eq: 'c==0',
                        ast.NotEq: 'c!=0',
                    }[kind]
                    parts.append(f'({{ let c={c}; {test} }})')
                    continue
                if ta != tb or ta not in ('i64', 'f64', 'char'):
                    self.fail(n, 'comparison operands must be numbers, or one character on both sides')
                parts.append(f'(({a}) {ops[type(op)]} ({b}))')
            return '(' + ' && '.join(parts) + ')'
        value, kind = self.expr(n)
        if kind == 'bool':
            return f'({value})'
        if kind == 'i64':
            return f'(({value}) != 0i64)'
        if kind == 'f64':
            return f'(({value}) != 0.0f64)'  # NaN is truthy, -0.0 is falsy, as in Python
        if kind == 'char':
            return 'true'  # a one-character string is never empty
        self.fail(
            n, 'arrays and tuples cannot be used as conditions', 'Use len(a) > 0 to test whether an array is empty.'
        )

    def membership(self, n):
        """ch in 'aeiou', ch in s, ch not in ...: one character looked up in a string."""
        value, kind = self.expr(n.left)
        right = n.comparators[0]
        negate = isinstance(n.ops[0], ast.NotIn)
        if kind != 'char':
            self.fail(
                n,
                '`in` is compiled only for one character in a string, like ch in "aeiou"',
                'For numbers, compare with == in a loop.',
            )
        if isinstance(right, ast.Constant) and type(right.value) is str:
            codes = sorted({ord(c) for c in right.value})
            test = f'[{",".join(f"{c}u32" for c in codes)}].contains(&({value}))' if codes else 'false'
        elif isinstance(right, ast.Name) and self.types.get(right.id) == 'str' and right.id in self.assigned:
            test = f'has({self.aref(right.id)},{value})'
        else:
            self.fail(n, 'the right side of `in` must be a string')
        return f'(!({test}))' if negate else f'({test})'

    def compare_operand(self, node):
        """Return (rust, kind, exact_int_literal_or_None)."""
        literal = None
        if isinstance(node, ast.Constant) and type(node.value) is int:
            literal = node.value
        elif (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and type(node.operand.value) is int
        ):
            literal = -node.operand.value
        if literal is not None and abs(literal) > 2**53:
            literal = None
        value, kind = self.expr(node)
        return value, kind, literal

    # ------------------------------------------------------------- statements
    def assign(self, node, name, value, kind):
        if kind in ARRAY_KINDS and not (name not in self.params and value.startswith('Arr::')):
            self.fail(
                node,
                'arrays can only be created with np.zeros/np.empty/np.ones, not copied or aliased',
                'Write into the original array, or create a new one with np.zeros_like(a).',
            )
        if tuple_parts(kind):
            self.fail(node, 'tuples can only be unpacked, like a, b = divmod(x, y)')
        if name in self.types and self.types[name] != kind:
            self.fail(
                node,
                f'{name!r} changes type from {self.types[name]} to {kind}',
                'Start float variables at 0.0, not 0, so the type never changes.',
            )
        if self.parallel is not None and kind in ARRAY_KINDS:
            self.fail(node, 'arrays cannot be created inside prange', 'Create them before the loop.')
        if self.parallel is not None and name in self.parallel['outer']:
            self.fail(
                node,
                f'prange iterations must not assign {name!r}, which exists outside the loop',
                'Only out[i] = ... writes and += on int totals are allowed for outer names inside prange.',
            )
        self.types[name] = kind
        self.assigned.add(name)
        if kind in ARRAY_KINDS:
            self.local_arrays.add(name)
        return f'{self.name(name)} = {value};'

    def store(self, node, target, value_node, op=None):
        """a[i] = v, a[i, j] = v, a[i][j] = v, or the op= forms."""
        name, kind, idx = self.indices(target)
        self.check_writable(name, node)
        if name in self.params and name not in self.written:
            self.written.add(name)
        element = ELEMENT[kind]
        j = self.temp('w')
        names = [f'{j}i{k}' for k in range(len(idx))]
        index_lets = ' '.join(f'let {nm}: i64 = {ix};' for nm, ix in zip(names, idx))
        reader = 'at' if len(idx) == 1 else 'at2'
        writer = 'put' if len(idx) == 1 else 'put2'
        current_name = f'__element{j}'
        if op is None:
            value, vkind = self.expr(value_node)
            if kind in LIST_KINDS and vkind != element:
                # A list keeps the exact object: lst[i] = 1 stores an int even in a list of floats.
                self.fail(
                    node,
                    f'a list of {"floats" if element == "f64" else "ints"} can only store '
                    f'{"float" if element == "f64" else "int"} values here',
                    'Convert first, for example lst[i] = float(x), so the list keeps one type.',
                )
            if vkind == 'bool':
                value, vkind = f'({value} as i64)', 'i64'
            elif vkind not in ('i64', 'f64'):
                self.fail(node, 'only numbers can be stored in arrays and lists')
        else:
            if not isinstance(op, (ast.Add, ast.Sub, ast.Mult)) and not (element == 'f64' and isinstance(op, ast.Div)):
                self.fail(node, 'array elements support +=, -=, *= (and /= for float arrays)')
            self.types[current_name] = element
            self.names[current_name] = f'{j}e'
            self.assigned.add(current_name)
            value, vkind = self.binop(node, ast.Name(id=current_name, ctx=ast.Load()), op, value_node)
            self.assigned.discard(current_name)
            del self.types[current_name]
        if element == 'f64' and vkind == 'i64':
            value, vkind = f'({value} as f64)', 'f64'
        if vkind != element:
            self.fail(node, 'int arrays can only be set to int values', 'Convert with int(x) or round(x) first.')
        read = f'let {j}e: {element} = {reader}({self.aref(name)},{",".join(names)})?; ' if op is not None else ''
        destination = (
            f'{writer}_par({self.aref(name)},' if self.parallel is not None else f'{writer}({self.amut(name, node)},'
        )
        return f'{{ {index_lets} {read}let {j}v: {element} = {value}; {destination}{",".join(names)},{j}v)?; }}'

    def unpack(self, node, targets, value_node):
        value, kind = self.expr(value_node)
        parts = tuple_parts(kind)
        if not parts or len(parts) != len(targets):
            self.fail(node, f'cannot unpack {kind} into {len(targets)} names')
        t = self.temp('u')
        lines = [f'let {t} = {value};']
        for k, (target, part) in enumerate(zip(targets, parts)):
            if not isinstance(target, ast.Name):
                self.fail(node, 'unpack into plain names, like a, b = ...')
            lines.append(self.assign(node, target.id, f'{t}.{k}', part))
        return '{ ' + ' '.join(lines) + ' }'

    def block(self, nodes):
        """Compile statements. Sets self.exits to True when control can never fall
        off the end of this block (return/break/continue on every path)."""
        lines = []
        exits = False
        for n in nodes:
            if exits:
                break  # unreachable in Python too
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
                continue
            if isinstance(n, ast.Pass):
                continue
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Subscript):
                lines.append(self.store(n, n.targets[0], n.value))
            elif isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Subscript):
                lines.append(self.store(n, n.target, n.value, n.op))
            elif isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Tuple):
                lines.append(self.unpack(n, n.targets[0].elts, n.value))
            elif isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                value, kind = self.expr(n.value)
                lines.append(self.assign(n, n.targets[0].id, value, kind))
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None:
                value, kind = self.expr(n.value)
                lines.append(self.assign(n, n.target.id, value, kind))
            elif isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name):
                lines.append(self.augassign(n))
            elif (
                isinstance(n, ast.Return)
                and n.value is not None
                and not (isinstance(n.value, ast.Constant) and n.value.value is None)
            ):
                lines.append(self.return_value(n))
                exits = True
            elif isinstance(n, ast.Return):
                if self.return_type not in (None, 'none'):
                    self.fail(n, 'a function cannot both return a value and return None')
                self.return_type = 'none'
                lines.append('return Ok(());')
                exits = True
            elif isinstance(n, ast.If):
                cond = self.condition(n.test)
                before = self.assigned.copy()
                yes = self.block(n.body)
                yes_exits = self.exits
                yes_assigned = self.assigned.copy()
                self.assigned = before.copy()
                no = self.block(n.orelse)
                no_exits = self.exits
                no_assigned = self.assigned
                # A branch that never falls through does not constrain what is
                # definitely assigned after the if statement.
                if yes_exits and not no_exits:
                    self.assigned = no_assigned
                elif no_exits and not yes_exits:
                    self.assigned = yes_assigned
                else:
                    self.assigned = yes_assigned & no_assigned
                exits = yes_exits and no_exits
                lines.append(f'if {cond} {{\n{yes}\n}} else {{\n{no}\n}}')
            elif isinstance(n, ast.For) and not n.orelse:
                lines.append(self.for_loop(n))
            elif isinstance(n, ast.While) and not n.orelse:
                if self.parallel is not None:
                    self.fail(n, 'loops inside prange are not compiled')
                before = self.assigned.copy()
                header = f'while {self.condition(n.test)} {{\ntick(fuel)?;'
                self.loop_depth += 1
                body = self.block(n.body)
                self.loop_depth -= 1
                self.assigned = before
                lines.append(f'{header}\n{body}\n}}')
            elif isinstance(n, (ast.Break, ast.Continue)) and self.loop_depth:
                if self.parallel is not None and self.loop_depth == self.parallel['depth']:
                    if isinstance(n, ast.Break):
                        self.fail(n, 'break is not allowed directly inside prange')
                    lines.append('return Ok(());')  # continue = end this iteration
                else:
                    lines.append('break;' if isinstance(n, ast.Break) else 'continue;')
                exits = True
            elif isinstance(n, ast.Expr) and isinstance(n.value, ast.Call):
                self.fail(
                    n,
                    'calls whose result is thrown away are not compiled (they can only have side effects)',
                    'Remove print()/logging from the hot function, or use the returned value.',
                )
            else:
                self.fail(n, f'unsupported statement ({type(n).__name__})')
        self.exits = exits
        return '\n'.join(lines)

    def augassign(self, n):
        name = n.target.id
        if self.parallel is not None and name in self.parallel['outer']:
            # Integer += reductions are exact in any order; floats are not.
            if not isinstance(n.op, ast.Add) or self.types.get(name) != 'i64':
                self.fail(
                    n,
                    'inside prange, only += on an int total defined outside the loop is allowed',
                    'Float sums depend on the order of additions, so they would not match Python. '
                    'Write each result to out[i] and sum afterwards.',
                )
            self.parallel['reductions'].discard(name)  # the right-hand side may not read it, the total may
            value = self.expect(n.value, 'i64')
            self.parallel['reductions'].add(name)
            return f'{self.name(name)}_part = ({self.name(name)}_part).checked_add({value}).ok_or(1)?;'
        value, kind = self.binop(n, ast.Name(id=name, ctx=ast.Load()), n.op, n.value)
        return self.assign(n, name, value, kind)

    def return_value(self, n):
        if self.parallel is not None:
            self.fail(n, 'return is not allowed inside prange')
        if isinstance(n.value, ast.Name) and self.types.get(n.value.id) in ARRAY_KINDS:
            name = n.value.id
            if name in self.params:
                self.fail(
                    n,
                    'returning an array argument is not compiled (Python returns the same object)',
                    'Return a new array made with np.zeros(...), or just write into the argument.',
                )
            kind = self.types[name]
            value = f'std::mem::replace(&mut {self.name(name)}, Arr::empty())'
        else:
            value, kind = self.expr(n.value)
        if self.return_type and kind != self.return_type:
            self.fail(
                n,
                'every return must have the same type',
                'Return 0.0 instead of 0 (or the other way round) on every path.',
            )
        self.return_type = kind
        return f'return Ok({value});'

    def for_loop(self, n):
        call = n.iter
        before = self.assigned.copy()
        j = self.temp('L')
        if self.parallel is not None:
            self.fail(n, 'loops inside prange are not compiled', 'Move the inner loop into a helper function.')
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id in ('range', 'enumerate')
            and call.func.id in self.local_names
        ):
            self.fail(n.iter, f'local {call.func.id} shadowing is unsupported')
        if (
            isinstance(call, ast.Call)
            and not call.keywords
            and (
                (isinstance(call.func, ast.Name) and call.func.id == 'range' and self.is_builtin('range'))
                or self.is_prange(call.func)
            )
            and 1 <= len(call.args) <= 3
        ):
            if not isinstance(n.target, ast.Name):
                self.fail(n.target, 'range loops need one loop variable')
            parallel = self.is_prange(call.func)
            if not parallel:
                self.builtin('range')
            args = [self.expect(a, 'i64') for a in call.args]
            start, stop, step = (
                ('0i64', args[0], '1i64')
                if len(args) == 1
                else (args[0], args[1], args[2] if len(args) == 3 else '1i64')
            )
            if parallel:
                if len(args) == 3:
                    self.fail(n, 'prange supports prange(stop) and prange(start, stop)')
                return self.parallel_loop(n, start, stop, before)
            target = self.assign(n, n.target.id, f'current{j} as i64', 'i64')
            header = f"""{{ let mut cursor{j} = ({start}) as i128;
let stop{j} = ({stop}) as i128; let step{j} = ({step}) as i128;
if step{j}==0 {{return Err(2)}}
while if step{j}>0 {{cursor{j}<stop{j}}} else {{cursor{j}>stop{j}}} {{
tick(fuel)?;
let current{j}=cursor{j}; cursor{j}+=step{j};
{target}"""
        elif (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == 'enumerate'
            and self.is_builtin('enumerate')
            and not call.keywords
            and 1 <= len(call.args) <= 2
        ):
            self.builtin('enumerate')
            if not (
                isinstance(n.target, ast.Tuple)
                and len(n.target.elts) == 2
                and all(isinstance(e, ast.Name) for e in n.target.elts)
            ):
                self.fail(n.target, 'use for i, x in enumerate(a)')
            arr, kind = self.array(call.args[0])
            if ARRAY_INFO[kind][1] != 1:
                self.fail(call, 'enumerate() over a 2-D array yields rows; it is not compiled')
            start = self.expect(call.args[1], 'i64') if len(call.args) == 2 else '0i64'
            index_target = self.assign(
                n, n.target.elts[0].id, f'(start{j}).checked_add(idx{j} as i64).ok_or(1)?', 'i64'
            )
            item_target = self.assign(n, n.target.elts[1].id, *self.read(f'at({self.aref(arr)},idx{j} as i64)?', kind))
            header = f"""{{ let start{j}: i64 = {start}; let n{j} = ({self.aref(arr)}).len; let mut k{j}: usize = 0;
while k{j} < n{j} {{
tick(fuel)?;
let idx{j}=k{j}; k{j}+=1;
{index_target}
{item_target}"""
        elif isinstance(call, ast.Name) and call.id in self.assigned and self.types.get(call.id) in ARRAY_KINDS:
            arr, kind = call.id, self.types[call.id]
            if ARRAY_INFO[kind][1] != 1:
                self.fail(call, 'looping over a 2-D array yields rows; loop over range(a.shape[0]) instead')
            if not isinstance(n.target, ast.Name):
                self.fail(n.target, 'use one loop variable: for x in a')
            item_target = self.assign(n, n.target.id, *self.read(f'at({self.aref(arr)},idx{j} as i64)?', kind))
            header = f"""{{ let n{j} = ({self.aref(arr)}).len; let mut k{j}: usize = 0;
while k{j} < n{j} {{
tick(fuel)?;
let idx{j}=k{j}; k{j}+=1;
{item_target}"""
        else:
            self.fail(
                n.iter,
                'for loops must use range(...), an array, list, str, bytes or Series argument, '
                'enumerate(...) of one of those, or prange(...)',
                'Loop over a local name, for example: for x in values.',
            )
        self.loop_depth += 1
        body = self.block(n.body)
        self.loop_depth -= 1
        self.assigned = before
        return f'{header}\n{body}\n}}}}'

    def parallel_loop(self, n, start, stop, before):
        """prange: iterations split across threads. Only independent writes (out[i]) and
        int += totals are allowed, so the result is exactly the sequential one."""
        if self.loop_depth:
            self.fail(n, 'prange must be the outermost loop', 'Put the prange loop at the top level of the function.')
        var = n.target.id
        writes, augmented = set(), set()
        for node in (x for stmt in n.body for x in ast.walk(stmt)):
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else ([node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else [])
            )
            for t in targets:
                base = t.value if isinstance(t, ast.Subscript) else None
                if isinstance(base, ast.Subscript):
                    base = base.value
                if isinstance(base, ast.Name):
                    writes.add(base.id)
            if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                augmented.add(node.target.id)
        outer = {v for v in self.assigned if v != var and self.types.get(v) in VALUES}
        arrays = {v for v in self.assigned if self.types.get(v) in ARRAY_KINDS}
        reductions = augmented & outer
        self.parallel = {
            'var': var,
            'outer': outer,
            'arrays': arrays,
            'writes': writes & arrays,
            'reductions': set(reductions),
            'depth': 1,
        }
        saved_types = dict(self.types)
        self.program['parallel'] = True
        try:
            self.types[var] = 'i64'
            self.assigned.add(var)
            loop_var = self.name(var)
            self.loop_depth += 1
            body = self.block(n.body)
            self.loop_depth -= 1
        finally:
            self.parallel = None
        self.assigned = before
        inner = [k for k in self.types if k not in saved_types and k != var and self.types[k] in VALUES]
        decls = ' '.join(f'let mut {self.name(k)}: {RUST_TYPE[self.types[k]]} = {ZERO[self.types[k]]};' for k in inner)
        reds = sorted(reductions)
        j = self.temp('P')
        part_type = '(' + ','.join('i64' for _ in reds) + (',' if len(reds) == 1 else '') + ')'
        part_value = '(' + ','.join(f'{self.name(r)}_part' for r in reds) + (',' if len(reds) == 1 else '') + ')'
        part_init = ' '.join(f'let mut {self.name(r)}_part: i64 = 0;' for r in reds)
        merge = ' '.join(
            f'{self.name(r)} = ({self.name(r)}).checked_add(p{j}.{k}).ok_or(1)?;' for k, r in enumerate(reds)
        )
        snapshots = ' '.join(
            f'(*{self.name(a)}).snapshot();' if a in self.params else f'{self.name(a)}.snapshot();'
            for a in sorted(writes & arrays)
            if a in self.params or a in self.local_arrays
        )
        views = ' '.join(
            f'let {self.name(a)}_r: &Arr<{ELEMENT[self.types[a]]}> = {"&*" if a in self.params else "&"}{self.name(a)};'
            for a in sorted(arrays)
        )
        return f"""{{ let start{j}: i64 = {start}; let stop{j}: i64 = {stop};
{snapshots}
let total{j}: i64 = if stop{j} > start{j} {{ stop{j} - start{j} }} else {{ 0 }};
if (fuel.fuel as u128) < (total{j} as u128) {{ return Err(4) }}
fuel.fuel -= total{j} as u64;
let threads{j}: i64 = if total{j} < {PARALLEL_MIN} {{ 1 }} else {{
    std::thread::available_parallelism().map(|c| c.get() as i64).unwrap_or(1).min(64) }};
let chunk{j}: i64 = if total{j} == 0 {{ 1 }} else {{ (total{j} + threads{j} - 1) / threads{j} }};
let cancel{j} = SendPtr(fuel.cancel as *mut AtomicU8);
{views}
let results{j}: Vec<Result<{part_type},i32>> = std::thread::scope(|scope| {{
    let mut handles = Vec::new();
    let mut lo = start{j};
    while lo < stop{j} {{
        let hi = if stop{j} - lo > chunk{j} {{ lo + chunk{j} }} else {{ stop{j} }};
        handles.push(scope.spawn(move || -> Result<{part_type},i32> {{
            let mut ctx = Ctx{{fuel: u64::MAX, cancel: cancel{j}.get() as *const AtomicU8, ticks: 0,
                              rng: std::ptr::null_mut(), gauss: std::ptr::null_mut()}};
            let fuel = &mut ctx;
            {part_init}
            let mut i = lo;
            while i < hi {{
                tick(fuel)?;
                let step: Result<(),i32> = (|| -> Result<(),i32> {{
                    {decls}
                    let mut {loop_var}: i64 = i;
                    {body}
                    Ok(())
                }})();
                step?;
                i += 1;
            }}
            Ok({part_value})
        }}));
        lo = hi;
    }}
    handles.into_iter().map(|h| h.join().unwrap_or(Err(8))).collect()
}});
for r{j} in results{j} {{ let p{j} = r{j}?; {merge} }}
}}"""

    # ------------------------------------------------------------- generation
    def function_text(self, rust_name, key=None):
        body = self.block(self.node.body)
        if self.return_type is None:
            self.return_type = 'none'  # no return statement: Python returns None
        recursive = self.program['recursive'].get(key) if key is not None else None
        if recursive is not None and recursive != self.return_type:
            raise Unsupported(
                f'{self.function.__name__}: recursive function returns {self.return_type} but is annotated {recursive}'
            )
        # A value-returning path that falls off the end returns Err(5): the call is
        # replayed in Python, which returns None exactly as the original function does.
        end = 'Ok(())' if self.return_type == 'none' else 'Err(5)'
        declarations = []
        for k, t in self.types.items():
            if k in self.params:
                continue
            if t in ARRAY_KINDS:
                declarations.append(f'let mut {self.name(k)}: {rust_type(t)} = Arr::empty();')
            elif t in VALUES:
                declarations.append(f'let mut {self.name(k)}: {RUST_TYPE[t]} = {ZERO[t]};')
        arguments = []
        for p in self.params:
            kind = self.types[p]
            if kind in ARRAY_KINDS:
                ref = '&mut ' if p in self.written else '&'
                arguments.append(f'{self.name(p)}: {ref}{rust_type(kind)}')
            else:
                arguments.append(f'mut {self.name(p)}: {RUST_TYPE[kind]}')
        arguments += ['fuel: &mut Ctx', 'depth: u32']
        return (
            f'fn {rust_name}({", ".join(arguments)})->Result<{rust_type(self.return_type)},i32>{{\n'
            f'if depth>{MAX_DEPTH} {{return Err(6)}}\n' + '\n'.join(declarations) + f'\n{body}\n{end}\n}}'
        )

    def generate(self):
        key = (id(self.function), tuple(self.types[p] for p in self.params))
        self.program['stack'].append(self.function)
        self.register('compute', key)
        try:
            main = self.function_text('compute', key)
        finally:
            self.program['stack'].pop()
        ffi, setup, calls, rollback = [], [], [], []
        for p in self.params:
            name, kind = self.name(p), self.types[p]
            if kind in ARRAY_KINDS:
                element = ELEMENT[kind]
                # Lists are private copies, so they need no undo information.
                # The result array of @vectorize is fresh on every call, so a failed call needs no undo.
                track = 'true' if p in self.written and kind in NUMPY_KINDS and p != '_pyrust_out' else 'false'
                ffi += [f'{name}: *mut {element}', f'{name}_rows: usize', f'{name}_cols: usize']
                setup.append(f'let mut a_{name} = Arr::<{element}>::view({name},{name}_rows,{name}_cols,{track});')
                calls.append(f'&mut a_{name}' if p in self.written else f'&a_{name}')
                if p in self.written:
                    rollback.append(f'a_{name}.rollback();')
            else:
                ffi.append(f'{name}: {RUST_TYPE[kind]}')
                calls.append(name)
        rt = self.return_type
        if rt in ARRAY_KINDS:
            element = ELEMENT[rt]
            ffi += [
                f'out_ptr: *mut *mut {element}',
                'out_len: *mut usize',
                'out_rows: *mut usize',
                'out_cols: *mut usize',
            ]
            store = '*out_rows=v.rows; *out_cols=v.cols; let (p,n)=into_raw(v); *out_ptr=p; *out_len=n;'
        elif tuple_parts(rt):
            parts = tuple_parts(rt)
            ffi += [f'out{k}: *mut {RUST_TYPE[t]}' for k, t in enumerate(parts)]
            store = ' '.join(f'*out{k}=v.{k};' for k in range(len(parts)))
        elif rt == 'none':
            store = ''
        else:
            ffi.append(f'out0: *mut {RUST_TYPE[rt]}')
            store = '*out0=v;'
        ffi += ['fuel: u64', 'cancel: *const AtomicU8']
        if self.program['random']:
            ffi += ['rng: *mut u32', 'gauss: *mut f64']
            context = 'Ctx{fuel, cancel, ticks: 0, rng, gauss}'
        else:
            context = 'Ctx{fuel, cancel, ticks: 0, rng: std::ptr::null_mut(), gauss: std::ptr::null_mut()}'
        calls += ['&mut ctx', '0']
        callees = '\n'.join(self.program['functions'])
        return f"""// Generated by PyRust Engine 0.7.1.
#![allow(unused_mut,unused_assignments,unused_parens,unreachable_code,dead_code,unused_variables,while_true,unused_braces,unused_unsafe,non_snake_case,unused_imports)]
{HELPERS}
{callees}
{main}
static QUIET: std::sync::Once = std::sync::Once::new();
#[no_mangle]
pub unsafe extern "C" fn pyrust_run({', '.join(ffi)})->i32{{
QUIET.call_once(|| std::panic::set_hook(Box::new(|_| {{}})));
let mut ctx = {context};
{chr(10).join(setup)}
let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| compute({', '.join(calls)})));
let status = match result {{ Ok(Ok(v)) => {{ {store} 0 }}, Ok(Err(code)) => code, Err(_) => 8 }};
if status != 0 {{ {' '.join(rollback)} }}
status
}}
"""

    # ---------------------------------------------------------- introspection
    @property
    def guards(self):
        return tuple(self.program['guards'].values())

    @property
    def code_guards(self):
        return tuple(self.program['code_guards'].values())

    @property
    def callees(self):
        return tuple(self.program['callees'])


class _Missing:
    def __repr__(self):
        return '<missing>'


_MISSING = _Missing()
