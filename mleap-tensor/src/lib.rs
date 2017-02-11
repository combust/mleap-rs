#![cfg_attr(feature="clippy", feature(plugin))]
#![cfg_attr(feature="clippy", plugin(clippy))]


extern crate num;

pub mod tensor;
pub mod math;

pub use tensor::*;