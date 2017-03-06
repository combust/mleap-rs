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

#[cfg(test)]
mod tests {
  use super::*;
  use op::dot::*;
  use core::build::*;
  use core::spec::Spec;

  #[test]
  fn dot_test() {
    let dt1 = DenseTensor::new(vec![1, 2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let dt2 = DenseTensor::new(vec![1, 2, 3], vec![2.0, 2.0, 2.0, 3.0, 3.0, 3.0]);

    let mut spec1 = dt1.spec();
    let mut spec2 = dt2.spec();

    spec1.pop_to_tshape();
    spec2.pop_to_tshape();

    let zip = spec1.zip(spec2).map(Vec::new(), |(a, b)| a.dot(b));
    let (nshape, nbuf) = zip.build_dense_scalar();

    assert_eq!(&nshape, &[1, 2]);
    assert_eq!(&nbuf, &[12.0, 45.0]);
  }

  #[test]
  fn broadcast_dot_test() {
    let shape: Vec<usize> = vec![1, 2, 3];
    let bshape: Vec<usize> = vec![2, 2, 3];
    let buf1: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let buf2: Vec<f32> = vec![2.0, 2.0, 2.0, 3.0, 3.0, 3.0];

    let dt1 = DenseTensor::new(shape.clone(), buf1);
    let dt2 = DenseTensor::new(shape.clone(), buf2);

    let mut spec1 = dt1.broadcast(bshape.clone());
    let mut spec2 = dt2.broadcast(bshape.clone());

    spec1.pop_to_tshape();
    spec2.pop_to_tshape();

    let mspec = spec1.zip(spec2).map(Vec::new(), |(a, b)| a.dot(b));

    let (nshape, nbuf) = mspec.build_dense_scalar();

    assert_eq!(&nshape, &[2, 2]);
    assert_eq!(&nbuf, &[12.0, 45.0, 12.0, 45.0]);
  }
}
