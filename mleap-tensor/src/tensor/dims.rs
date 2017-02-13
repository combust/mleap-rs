use tensor::{DenseTensor, SparseTensor, Dimensions};

pub trait Dims {
  fn dims(&self) -> &Dimensions;
}

impl<T> Dims for DenseTensor<T> {
  fn dims(&self) -> &Dimensions { &self.dims }
}

impl<T> Dims for SparseTensor<T> {
  fn dims(&self) -> &Dimensions { &self.dims }
}
