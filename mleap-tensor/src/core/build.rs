use spec::*;

pub trait BuildDense<'a, T> {
  fn build_dense(&'a self) -> (Vec<usize>, Vec<T>);
}

pub trait BuildDenseScalar<'a, T> {
  fn build_dense_scalar(&'a self) -> (Vec<usize>, Vec<T>);
}

impl<'a, T, I: Iterator<Item=T>, S: ShapedSpec<'a, Item=T, I=I>> BuildDenseScalar<'a, T> for S {
  fn build_dense_scalar(&'a self) -> (Vec<usize>, Vec<T>) {
    let shape: Vec<usize> = self.shape().iter().chain(self.tshape().iter()).map(|x| *x).collect();
    let buf: Vec<T> = self.iter().collect();

    (shape, buf)
  }
}

impl<'a, T, I: Iterator<Item=Vec<T>>, S: ShapedSpec<'a, Item=Vec<T>, I=I>> BuildDense<'a, T> for S {
  fn build_dense(&'a self) -> (Vec<usize>, Vec<T>) {
    let shape: Vec<usize> = self.shape().iter().chain(self.tshape().iter()).map(|x| *x).collect();
    let buf: Vec<T> = self.iter().flat_map(|v| v).collect();

    (shape, buf)
  }
}
