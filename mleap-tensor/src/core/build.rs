use spec::*;

pub trait BuildDense<'a, T> {
  fn build_dense(&'a self) -> (Vec<usize>, Vec<T>);
}

pub trait BuildDenseScalar<'a, T> {
  fn build_dense_scalar(&'a self) -> (Vec<usize>, Vec<T>);
}

impl<'a, T, I: Iterator<Item=T>, S: Spec<'a, Item=T, I=I>> BuildDenseScalar<'a, T> for S {
  fn build_dense_scalar(&'a self) -> (Vec<usize>, Vec<T>) {
    let shape = self.shape().to_vec();
    let buf: Vec<T> = self.iter().collect();

    (shape, buf)
  }
}

impl<'a, T, I: Iterator<Item=Vec<T>>, S: Spec<'a, Item=Vec<T>, I=I>> BuildDense<'a, T> for S {
  fn build_dense(&'a self) -> (Vec<usize>, Vec<T>) {
    let shape = self.shape().to_vec();
    let buf: Vec<T> = self.iter().flat_map(|v| v).collect();

    (shape, buf)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use blas_sys::c::cblas_sdot;

  fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
      cblas_sdot(a.len() as i32, a.as_ptr(), 1, b.as_ptr(), 1)
    }
  }

  #[test]
  fn dot_test() {
    let shape: Vec<usize> = vec![1, 2, 3];
    let buf1: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let buf2: Vec<f32> = vec![2.0, 2.0, 2.0, 3.0, 3.0, 3.0];

    let mut spec1 = Dense::new(shape.clone(), &buf1);
    let mut spec2 = Dense::new(shape.clone(), &buf2);

    spec1.pop_to_tshape();
    spec2.pop_to_tshape();

    let zip = spec1.zip(spec2).map(Vec::new(), |(a, b)| dot_f32(a, b));
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

    let mut spec1 = DenseBroadcast::new(bshape.clone(), shape.clone(), &buf1);
    let mut spec2 = DenseBroadcast::new(bshape.clone(), shape.clone(), &buf2);

    spec1.pop_to_tshape();
    spec2.pop_to_tshape();

    let mspec = spec1.zip(spec2).map(Vec::new(), |(a, b)| dot_f32(a, b));

    let (nshape, nbuf) = mspec.build_dense_scalar();

    assert_eq!(&nshape, &[2, 2]);
    assert_eq!(&nbuf, &[12.0, 45.0, 12.0, 45.0]);
  }
}
