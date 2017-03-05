use iter::TensorIterator;

pub trait Scalar { }

impl Scalar for f32 { }
impl Scalar for f64 { }

pub trait Builder<T> {
  fn build_dense(&mut self) -> (Vec<usize>, Vec<T>);
}

impl<T: Scalar, B: TensorIterator<Item=T> + ?Sized> Builder<T> for B {
  /// Build a dense tensor from a tensor iterator.
  ///
  /// ```
  /// use mleap_tensor::core::op::Dot;
  /// use mleap_tensor::core::build::*;
  ///
  /// let shape1: Vec<usize> = vec![1, 2, 3];
  /// let shape2: Vec<usize> = vec![1, 2, 3];
  ///
  /// let buf1: Vec<f32> = vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
  /// let buf2: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
  ///
  /// let mut iter = Dot::<f32>::try_from_dense(&shape1, &buf1, &shape2, &buf2).unwrap();
  /// let (bshape, buf) = iter.build_dense();
  ///
  /// assert_eq!(&bshape, &[1, 2, 1]);
  /// assert_eq!(&buf, &[6.0, 30.0]);
  /// ```
  fn build_dense(&mut self) -> (Vec<usize>, Vec<T>) {
    let shape = self.bshape().to_vec();
    let buf: Vec<T> = self.collect();

    (shape, buf)
  }
}
