"""Rust helpers for strings, lists of strings, dicts, sets and float32 (0.7.2).

Every helper reproduces CPython's behaviour for the cases it accepts and returns an
error status for anything else, so the call replays in Python:
  str.find/rfind/count/split/strip use CPython's index rules (stringlib),
  whitespace is Python's str.isspace() table, lower()/upper() are exact for ASCII,
  dicts keep insertion order and detect changes during iteration like CPython.
"""

TEXT_HELPERS = r"""
use std::collections::{HashMap,HashSet};
// A fast keyed hash for dict/set keys (short strings and ints). The key is random per
// process, like Python's string hashing, so hash values cannot be predicted from outside.
static SEED: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
fn seed()->u64{*SEED.get_or_init(|| {
    use std::hash::{BuildHasher,Hasher};
    let mut h=std::collections::hash_map::RandomState::new().build_hasher(); h.write_u64(0x5eed); h.finish() | 1
})}
#[derive(Clone)] pub struct FastState(u64);
impl Default for FastState { fn default()->Self{FastState(seed())} }
pub struct FastHasher(u64);
impl std::hash::BuildHasher for FastState { type Hasher=FastHasher; fn build_hasher(&self)->FastHasher{FastHasher(self.0)} }
#[inline] fn fold(a:u64,b:u64)->u64{let r=(a as u128)*(b as u128); (r as u64)^((r>>64) as u64)}
impl std::hash::Hasher for FastHasher {
    #[inline] fn write(&mut self,bytes:&[u8]){
        let mut h=self.0; let mut chunks=bytes.chunks_exact(8);
        for c in &mut chunks {h=fold(h^u64::from_le_bytes([c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7]]),0x9E3779B97F4A7C15);}
        let rest=chunks.remainder(); let mut t=[0u8;8]; t[..rest.len()].copy_from_slice(rest);
        self.0=fold(h^u64::from_le_bytes(t)^(bytes.len() as u64),0xD6E8FEB86659FD93);
    }
    #[inline] fn write_u64(&mut self,x:u64){self.0=fold(self.0^x,0x9E3779B97F4A7C15);}
    #[inline] fn write_i64(&mut self,x:i64){self.write_u64(x as u64)}
    #[inline] fn write_usize(&mut self,x:usize){self.write_u64(x as u64)}
    #[inline] fn finish(&self)->u64{fold(self.0,0xA0761D6478BD642F)}
}
pub type HSet<K>=HashSet<K,FastState>;
// Strings are Unicode code points, shared and copied on write (s += t is amortized O(1)).
#[derive(Clone,PartialEq,Eq,PartialOrd,Ord,Default)]
pub struct S(std::sync::Arc<Vec<u32>>);
impl std::hash::Hash for S {
    fn hash<H:std::hash::Hasher>(&self,h:&mut H){
        let v:&[u32]=&self.0;
        h.write(unsafe{std::slice::from_raw_parts(v.as_ptr() as *const u8,v.len()*4)});
    }
}
impl std::ops::Deref for S { type Target=[u32]; fn deref(&self)->&[u32]{&self.0} }
fn sv(v:&[u32])->S{S(std::sync::Arc::new(v.to_vec()))}
fn sown(v:Vec<u32>)->S{S(std::sync::Arc::new(v))}
fn sl(a:&Arr<u32>)->&[u32]{if a.len==0 {&[]} else {unsafe{std::slice::from_raw_parts(a.ptr,a.len)}}}
fn av<T:Copy>(a:&Arr<T>)->&[T]{if a.len==0 {&[]} else {unsafe{std::slice::from_raw_parts(a.ptr,a.len)}}}
fn avm<T:Copy>(a:&mut Arr<T>)->&mut [T]{if a.len==0 {&mut []} else {unsafe{std::slice::from_raw_parts_mut(a.ptr,a.len)}}}
fn s_push(s:&mut S,t:&[u32]){std::sync::Arc::make_mut(&mut s.0).extend_from_slice(t);}
fn s_cat(a:&[u32],b:&[u32])->S{let mut v=Vec::with_capacity(a.len()+b.len()); v.extend_from_slice(a); v.extend_from_slice(b); sown(v)}
fn s_at(s:&[u32],i:i64)->Result<u32,i32>{let j=index(s.len(),i)?; Ok(s[j])}
fn v_at<T:Clone>(v:&[T],i:i64)->Result<T,i32>{let j=index(v.len(),i)?; Ok(v[j].clone())}
fn v_put<T>(v:&mut [T],i:i64,x:T)->Result<(),i32>{let j=index(v.len(),i)?; v[j]=x; Ok(())}
// Slice bounds with step 1 (PySlice_AdjustIndices).
fn s_adj(n:i64,i:Option<i64>,d:i64)->i64{match i {None=>d, Some(mut v)=>{if v<0 {v+=n; if v<0 {v=0}} else if v>n {v=n} v}}}
fn bounds(n:usize,a:Option<i64>,b:Option<i64>)->(usize,usize){
    let n=n as i64; let x=s_adj(n,a,0); let y=s_adj(n,b,n);
    if y<=x {(0,0)} else {(x as usize,y as usize)}
}
fn s_slice(s:&[u32],a:Option<i64>,b:Option<i64>)->S{let (x,y)=bounds(s.len(),a,b); sv(&s[x..y])}
fn v_slice<T:Clone>(s:&[T],a:Option<i64>,b:Option<i64>)->Vec<T>{let (x,y)=bounds(s.len(),a,b); s[x..y].to_vec()}
fn a_slice<T:Copy>(a:&Arr<T>,x:Option<i64>,y:Option<i64>,fill:T)->Result<Arr<T>,i32>{
    let part=v_slice(av(a),x,y); let mut out=Arr::<T>::owned(part.len() as i64,1,fill)?;
    avm(&mut out).copy_from_slice(&part); Ok(out)
}
// str.find / rfind / count index rules (ADJUST_INDICES in CPython's stringlib).
fn s_range(n:usize,start:Option<i64>,end:Option<i64>)->(i64,i64){
    let n=n as i64; let mut e=end.unwrap_or(n);
    if e>n {e=n} else if e<0 {e+=n; if e<0 {e=0}}
    let mut st=start.unwrap_or(0); if st<0 {st+=n; if st<0 {st=0}}
    (st,e)
}
fn s_find(s:&[u32],sub:&[u32],start:Option<i64>,end:Option<i64>)->i64{
    let (st,e)=s_range(s.len(),start,end); let m=sub.len() as i64;
    if e-st<m {return -1} if m==0 {return st}
    let mut i=st; while i+m<=e {if &s[i as usize..(i+m) as usize]==sub {return i} i+=1} -1
}
fn s_rfind(s:&[u32],sub:&[u32],start:Option<i64>,end:Option<i64>)->i64{
    let (st,e)=s_range(s.len(),start,end); let m=sub.len() as i64;
    if e-st<m {return -1} if m==0 {return e}
    let mut i=e-m; while i>=st {if &s[i as usize..(i+m) as usize]==sub {return i} i-=1} -1
}
fn s_count(s:&[u32],sub:&[u32],start:Option<i64>,end:Option<i64>)->i64{
    let (st,e)=s_range(s.len(),start,end); let m=sub.len() as i64;
    if e-st<m {return 0} if m==0 {return e-st+1}
    let mut c=0; let mut i=st;
    while i+m<=e {if &s[i as usize..(i+m) as usize]==sub {c+=1; i+=m} else {i+=1}} c
}
fn s_has(s:&[u32],sub:&[u32])->bool{s_find(s,sub,None,None)>=0}
fn s_starts(s:&[u32],p:&[u32])->bool{s.len()>=p.len() && &s[..p.len()]==p}
fn s_ends(s:&[u32],p:&[u32])->bool{s.len()>=p.len() && &s[s.len()-p.len()..]==p}
// Python's str.isspace() / str.split() whitespace.
fn is_ws(c:u32)->bool{matches!(c,9..=13|28..=32|0x85|0xA0|0x1680|0x2000..=0x200A|0x2028|0x2029|0x202F|0x205F|0x3000)}
fn s_split_ws(s:&[u32])->Vec<S>{
    let mut out=Vec::new(); let n=s.len(); let mut i=0;
    while i<n {
        while i<n && is_ws(s[i]) {i+=1}
        if i>=n {break}
        let st=i; while i<n && !is_ws(s[i]) {i+=1}
        out.push(sv(&s[st..i]));
    }
    out
}
fn s_split(s:&[u32],sep:&[u32])->Result<Vec<S>,i32>{
    if sep.is_empty() {return Err(6)}
    let mut out=Vec::new(); let m=sep.len(); let mut st=0; let mut i=0;
    while i+m<=s.len() {if &s[i..i+m]==sep {out.push(sv(&s[st..i])); i+=m; st=i} else {i+=1}}
    out.push(sv(&s[st..])); Ok(out)
}
fn s_strip(s:&[u32],chars:Option<&[u32]>,left:bool,right:bool)->S{
    let hit=|c:u32| match chars {None=>is_ws(c), Some(cs)=>cs.contains(&c)};
    let mut a=0; let mut b=s.len();
    if left {while a<b && hit(s[a]) {a+=1}}
    if right {while b>a && hit(s[b-1]) {b-=1}}
    sv(&s[a..b])
}
fn s_lower(s:&[u32])->Result<S,i32>{let mut v=Vec::with_capacity(s.len()); for &c in s {if c>=128 {return Err(6)} v.push((c as u8).to_ascii_lowercase() as u32)} Ok(sown(v))}
fn s_upper(s:&[u32])->Result<S,i32>{let mut v=Vec::with_capacity(s.len()); for &c in s {if c>=128 {return Err(6)} v.push((c as u8).to_ascii_uppercase() as u32)} Ok(sown(v))}
fn s_replace(s:&[u32],old:&[u32],new:&[u32])->Result<S,i32>{
    if old.is_empty() {return Err(6)}
    let mut v=Vec::with_capacity(s.len()); let m=old.len(); let mut i=0;
    while i<s.len() {if i+m<=s.len() && &s[i..i+m]==old {v.extend_from_slice(new); i+=m} else {v.push(s[i]); i+=1}}
    Ok(sown(v))
}
fn s_join(sep:&[u32],items:&[S])->S{
    let mut v=Vec::new();
    for (k,it) in items.iter().enumerate() {if k>0 {v.extend_from_slice(sep)} v.extend_from_slice(it)}
    sown(v)
}
fn s_mul(s:&[u32],n:i64)->Result<S,i32>{
    if n<=0 {return Ok(sv(&[]))}
    let total=(s.len() as u64).checked_mul(n as u64).ok_or(6)?; if total>(1u64<<34) {return Err(6)}
    let mut v=Vec::with_capacity(total as usize); for _ in 0..n {v.extend_from_slice(s)} Ok(sown(v))
}
fn s_chars(s:&[u32])->Vec<S>{s.iter().map(|&c| sv(&[c])).collect()}
fn v_pop(v:&mut Vec<S>,i:i64)->Result<S,i32>{let j=index(v.len(),i)?; Ok(v.remove(j))}
fn s_int(i:i64)->S{sown(i.to_string().bytes().map(|b| b as u32).collect())}
fn s_bool(b:bool)->S{let t:&[u32]=if b {&[84,114,117,101]} else {&[70,97,108,115,101]}; sv(t)}
fn s_all(s:&[u32],f:fn(u32)->Result<bool,i32>)->Result<bool,i32>{
    if s.is_empty() {return Ok(false)}
    for &c in s {if !f(c)? {return Ok(false)}} Ok(true)
}
fn s_parse_int(s:&[u32])->Result<i64,i32>{
    let t:&[u32]=&s_strip(s,None,true,true); if t.is_empty() {return Err(6)}
    let (neg,d)=match t[0] {43=>(false,&t[1..]),45=>(true,&t[1..]),_=>(false,t)};
    if d.is_empty() {return Err(6)}
    let mut v:i128=0; let mut under=true;
    for &c in d {
        if c==95 {if under {return Err(6)} under=true; continue}
        if !(48..=57).contains(&c) {return Err(6)}
        under=false; v=v*10+(c-48) as i128; if v>(1i128<<64) {return Err(1)}
    }
    if under {return Err(6)}
    i64::try_from(if neg {-v} else {v}).map_err(|_|1)
}
fn s_parse_float(s:&[u32])->Result<f64,i32>{
    let t:&[u32]=&s_strip(s,None,true,true); if t.is_empty() {return Err(6)}
    let mut b=String::with_capacity(t.len());
    for &c in t {if c>=128 {return Err(6)} b.push(c as u8 as char)}
    let x=b.as_bytes(); let n=x.len(); let mut i=0;
    if i<n && (x[i]==b'+' || x[i]==b'-') {i+=1}
    let ds=i; while i<n && x[i].is_ascii_digit() {i+=1} let whole=i-ds;
    let mut frac=0; if i<n && x[i]==b'.' {i+=1; let fs=i; while i<n && x[i].is_ascii_digit() {i+=1} frac=i-fs}
    if whole+frac==0 {return Err(6)}
    if i<n && (x[i]==b'e' || x[i]==b'E') {
        i+=1; if i<n && (x[i]==b'+' || x[i]==b'-') {i+=1}
        let es=i; while i<n && x[i].is_ascii_digit() {i+=1} if i==es {return Err(6)}
    }
    if i!=n {return Err(6)}
    b.parse::<f64>().map_err(|_|6)
}
// dict: insertion order, updates in place, CPython-like change detection while iterating.
pub struct OMap<K,V>{slots:Vec<Option<(K,V)>>,idx:HashMap<K,usize,FastState>,ver:u64}
impl<K:Clone+Eq+std::hash::Hash,V:Clone> OMap<K,V>{
    fn new()->Self{OMap{slots:Vec::new(),idx:HashMap::default(),ver:0}}
    // One hash lookup for d[k] = d.get(k, d0) + x and d[k] += x.
    fn slot_or(&mut self,k:K,d:V)->usize{
        match self.idx.entry(k) {
            std::collections::hash_map::Entry::Occupied(e)=>*e.get(),
            std::collections::hash_map::Entry::Vacant(e)=>{
                let i=self.slots.len(); let key=e.key().clone(); e.insert(i);
                self.slots.push(Some((key,d))); self.ver+=1; i
            }
        }
    }
    fn slot(&self,k:&K)->Result<usize,i32>{self.idx.get(k).copied().ok_or(3)}
    fn val(&self,i:usize)->Result<V,i32>{self.slots[i].as_ref().map(|e| e.1.clone()).ok_or(8)}
    fn put(&mut self,i:usize,v:V){if let Some(e)=self.slots[i].as_mut() {e.1=v}}
    fn len(&self)->i64{self.idx.len() as i64}
    fn has(&self,k:&K)->bool{self.idx.contains_key(k)}
    fn get(&self,k:&K)->Option<V>{self.idx.get(k).and_then(|&i| self.slots[i].as_ref().map(|e| e.1.clone()))}
    fn need(&self,k:&K)->Result<V,i32>{self.get(k).ok_or(3)}
    fn set(&mut self,k:K,v:V){
        if let Some(&i)=self.idx.get(&k) {if let Some(e)=self.slots[i].as_mut() {e.1=v}}
        else {self.idx.insert(k.clone(),self.slots.len()); self.slots.push(Some((k,v))); self.ver+=1;}
    }
    fn del(&mut self,k:&K)->Result<V,i32>{
        let i=self.idx.remove(k).ok_or(3)?; self.ver+=1;
        let v=self.slots[i].take().map(|e| e.1).ok_or(8)?;
        if self.slots.len()>32 && self.idx.len()*2<self.slots.len() {self.compact()}
        Ok(v)
    }
    fn clear(&mut self){self.slots.clear(); self.idx.clear(); self.ver+=1;}
    fn compact(&mut self){
        let old=std::mem::take(&mut self.slots); self.idx.clear();
        for (k,v) in old.into_iter().flatten() {self.idx.insert(k.clone(),self.slots.len()); self.slots.push(Some((k,v)));}
    }
    fn next(&self,k:&mut usize)->Option<(K,V)>{
        while *k<self.slots.len() {let e=self.slots[*k].clone(); *k+=1; if e.is_some() {return e}} None
    }
    fn keys(&self)->Vec<K>{self.slots.iter().flatten().map(|e| e.0.clone()).collect()}
    fn values(&self)->Vec<V>{self.slots.iter().flatten().map(|e| e.1.clone()).collect()}
}
impl<K:Clone+Eq+std::hash::Hash,V:Clone> Default for OMap<K,V>{fn default()->Self{OMap::new()}}
fn set_remove<K:Eq+std::hash::Hash>(s:&mut HSet<K>,k:&K)->Result<(),i32>{if s.remove(k) {Ok(())} else {Err(3)}}
fn set_and<K:Clone+Eq+std::hash::Hash>(a:&HSet<K>,b:&HSet<K>)->HSet<K>{a.intersection(b).cloned().collect()}
fn set_or<K:Clone+Eq+std::hash::Hash>(a:&HSet<K>,b:&HSet<K>)->HSet<K>{a.union(b).cloned().collect()}
fn set_sub<K:Clone+Eq+std::hash::Hash>(a:&HSet<K>,b:&HSet<K>)->HSet<K>{a.difference(b).cloned().collect()}
fn set_xor<K:Clone+Eq+std::hash::Hash>(a:&HSet<K>,b:&HSet<K>)->HSet<K>{a.symmetric_difference(b).cloned().collect()}
// sorted() / list.sort(): stable, reverse keeps equal items in order; NaN replays.
fn sort_f(v:&mut [f64],rev:bool)->Result<(),i32>{
    if v.iter().any(|x| x.is_nan()) {return Err(6)}
    if rev {v.sort_by(|a,b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal))}
    else {v.sort_by(|a,b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))}
    Ok(())
}
fn sort_o<T:Ord>(v:&mut [T],rev:bool){if rev {v.sort_by(|a,b| b.cmp(a))} else {v.sort()}}
// max()/min() of a sequence: Python compares item > best (item < best), first one wins.
fn seq_max<T:PartialOrd+Clone>(v:&[T],low:bool)->Result<T,i32>{
    if v.is_empty() {return Err(6)}
    let mut m=v[0].clone();
    for x in &v[1..] {if (!low && *x>m) || (low && *x<m) {m=x.clone()}}
    Ok(m)
}
fn arr_from<T:Copy>(v:Vec<T>,fill:T)->Result<Arr<T>,i32>{let mut out=Arr::<T>::owned(v.len() as i64,1,fill)?; avm(&mut out).copy_from_slice(&v); Ok(out)}
// float32 (NumPy 2 rules): division by zero replays, NumPy gives inf with a warning there.
fn div32(a:f32,b:f32)->Result<f32,i32>{if b==0.0 {Err(2)} else {Ok(a/b)}}
fn sum_f32(a:&Arr<f32>,c:&mut Ctx)->Result<f32,i32>{
    if a.len==0 {return Err(6)}
    let mut s=0.0f32; for j in 0..a.len {tick(c)?; s+=unsafe{*a.ptr.add(j)};} Ok(s)
}
fn min32(a:f32,b:f32)->f32{if b<a {b} else {a}}
fn max32(a:f32,b:f32)->f32{if b>a {b} else {a}}
// Data exchange with Python for strings, lists of strings, dicts and sets.
pub struct Wr{b:Vec<u8>}
impl Wr{
    fn new()->Wr{Wr{b:Vec::new()}}
    fn u(&mut self,x:u64){self.b.extend_from_slice(&x.to_le_bytes())}
}
pub struct Rd{p:*const u8,n:usize,pos:usize}
impl Rd{
    fn u(&mut self)->u64{
        if self.pos+8>self.n {return 0}
        let v=unsafe{std::ptr::read_unaligned(self.p.add(self.pos) as *const u64)}; self.pos+=8; u64::from_le(v)
    }
    fn w(&mut self)->u32{
        if self.pos+4>self.n {return 0}
        let v=unsafe{std::ptr::read_unaligned(self.p.add(self.pos) as *const u32)}; self.pos+=4; u32::from_le(v)
    }
}
pub trait Col:Sized+Clone{fn put(w:&mut Wr,v:&[Self]); fn get(r:&mut Rd,n:usize)->Vec<Self>;}
impl Col for i64{fn put(w:&mut Wr,v:&[i64]){for &x in v {w.u(x as u64)}} fn get(r:&mut Rd,n:usize)->Vec<i64>{(0..n).map(|_| r.u() as i64).collect()}}
impl Col for f64{fn put(w:&mut Wr,v:&[f64]){for &x in v {w.u(x.to_bits())}} fn get(r:&mut Rd,n:usize)->Vec<f64>{(0..n).map(|_| f64::from_bits(r.u())).collect()}}
impl Col for bool{fn put(w:&mut Wr,v:&[bool]){for &x in v {w.u(x as u64)}} fn get(r:&mut Rd,n:usize)->Vec<bool>{(0..n).map(|_| r.u()!=0).collect()}}
impl Col for S{
    fn put(w:&mut Wr,v:&[S]){
        let total:usize=v.iter().map(|s| s.len()).sum(); w.u(total as u64);
        for s in v {w.u(s.len() as u64)}
        for s in v {for &c in s.iter() {w.b.extend_from_slice(&c.to_le_bytes())}}
    }
    fn get(r:&mut Rd,n:usize)->Vec<S>{
        let _total=r.u(); let lens:Vec<u64>=(0..n).map(|_| r.u()).collect();
        lens.iter().map(|&l| sown((0..l).map(|_| r.w()).collect())).collect()
    }
}
fn pack_str(s:&S)->Wr{let mut w=Wr::new(); S::put(&mut w,std::slice::from_ref(s)); w}
fn pack_list<T:Col>(v:&[T])->Wr{let mut w=Wr::new(); w.u(v.len() as u64); T::put(&mut w,v); w}
fn pack_set<K:Col+Eq+std::hash::Hash>(s:&HSet<K>)->Wr{let v:Vec<K>=s.iter().cloned().collect(); pack_list(&v)}
fn pack_dict<K:Col+Eq+std::hash::Hash,V:Col>(m:&OMap<K,V>)->Wr{
    let mut w=Wr::new(); w.u(m.len() as u64); K::put(&mut w,&m.keys()); V::put(&mut w,&m.values()); w
}
fn unpack_list<T:Col>(p:*const u8,n:usize)->Vec<T>{let mut r=Rd{p,n,pos:0}; let k=r.u() as usize; T::get(&mut r,k)}
fn unpack_set<K:Col+Eq+std::hash::Hash>(p:*const u8,n:usize)->HSet<K>{unpack_list::<K>(p,n).into_iter().collect()}
fn unpack_dict<K:Col+Eq+std::hash::Hash,V:Col>(p:*const u8,n:usize)->OMap<K,V>{
    let mut r=Rd{p,n,pos:0}; let k=r.u() as usize; let keys=K::get(&mut r,k); let vals=V::get(&mut r,k);
    let mut m=OMap::new(); for (a,b) in keys.into_iter().zip(vals) {m.set(a,b)} m
}
unsafe fn out_bytes(w:Wr,op:*mut *mut u8,ol:*mut usize){let b=w.b.into_boxed_slice(); *ol=b.len(); *op=Box::into_raw(b) as *mut u8;}
#[no_mangle] pub unsafe extern "C" fn pyrust_free_u8(p:*mut u8,n:usize){
    if !p.is_null() {drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(p,n)));}
}
"""
