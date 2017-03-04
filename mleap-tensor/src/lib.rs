#![cfg_attr(feature="clippy", feature(plugin))]
#![cfg_attr(feature="clippy", plugin(clippy))]

extern crate num;
extern crate blas_sys;

pub mod broadcast;
pub mod tensor;
pub mod math;

pub use tensor::*;
pub use math::*;
