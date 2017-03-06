use std::result::Result;

use core::spec;
use core::broadcast;

pub struct DenseTensor<T> {
  shape: Vec<usize>,
  buf: Vec<T>
}

impl<T> DenseTensor<T> {
  pub fn new(shape: Vec<usize>, buf: Vec<T>) -> DenseTensor<T> {
    DenseTensor { shape: shape, buf: buf }
  }

  pub fn spec<'a>(&'a self) -> spec::Dense<T> {
    spec::Dense::new(self.shape.clone(), &self.buf)
  }

  pub fn broadcast<'a>(&'a self, bshape: Vec<usize>) -> spec::DenseBroadcast<T> {
    self.try_broadcast(bshape).unwrap()
  }

  pub fn try_broadcast<'a>(&'a self, bshape: Vec<usize>) -> Result<spec::DenseBroadcast<T>, broadcast::Error> {
    if broadcast::compatible(&bshape, &self.shape) {
      Ok(spec::DenseBroadcast::new(bshape, self.shape.clone(), &self.buf))
    } else { Err(broadcast::Error::IncompatibleBroadcast) }
  }
}
