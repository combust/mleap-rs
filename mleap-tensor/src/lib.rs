#![cfg_attr(feature="clippy", feature(plugin))]
#![cfg_attr(feature="clippy", plugin(clippy))]

extern crate num;
extern crate blas_sys;

pub mod core;
pub mod op;
pub mod tensor;

pub use core::*;
pub use op::*;
pub use tensor::*;
