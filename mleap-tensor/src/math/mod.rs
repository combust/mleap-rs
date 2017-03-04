use std::marker::PhantomData;

mod builder;

pub use self::builder::*;

#[derive(Debug, Clone, Copy)]
pub enum Error {
  Broadcast
}

pub struct Blas<T> {
  _phantom: PhantomData<T>
}

const BlasF32: Blas<f32> = Blas { _phantom: PhantomData { } };
