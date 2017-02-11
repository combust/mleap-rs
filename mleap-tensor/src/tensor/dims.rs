use tensor::{DenseTensor, SparseTensor};

pub trait Dims {
  fn dims(&self) -> &[usize];
}

impl<T> Dims for DenseTensor<T> {
  fn dims(&self) -> &[usize] { &self.dims }
}

impl<T> Dims for SparseTensor<T> {
  fn dims(&self) -> &[usize] { &self.dims }
}
