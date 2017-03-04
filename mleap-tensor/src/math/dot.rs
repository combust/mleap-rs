//use math::{Blas, Error};
//use std::result::Result;
//use std::iter::Zip;
//use tensor::*;
//use broadcast;
//use blas_sys::c::cblas_sdot;

//pub trait Dot<T> {
//fn dot_dense<'a>(a: &'a Tensor<T>, b: &'a Tensor<T>) -> Result<Box<TensorBuilder<'a, T>>, Error>;
//}

//pub fn dense_dot_iter_f32(a: &mut Iterator<Item=&[f32]>,
//b: &mut Iterator<Item=&[f32]>) -> Vec<f32> {
//a.zip(b).map(|(a, b)| {
//dot_f32(a, b)
//}).collect()
//}

//pub fn dot_f32(a: & [f32], b: & [f32]) -> f32 {
//unsafe {
//cblas_sdot(a.len() as i32, a.as_ptr(), 1, b.as_ptr(), 1)
//}
//}

//impl Dot<f32> for Blas<f32> {
//fn dot(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, Error> {
//let ia = a.broadcast_iter(b, 1);
//let ib = b.broadcast_iter(a, 1);

//match (ia, ib) {
//(Some(BroadcastIter::Dense(dia)), Some(BroadcastIter::Dense(dib))) => {
//Ok(dense_iter)
//},
//_ => Err(Error::Broadcast)
//}
//}
//}
