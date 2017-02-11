use tensor::DenseTensor;

pub trait Ravel {
  fn ravel_index(&self, index: &[usize]) -> Option<usize>;
}

impl<T> Ravel for DenseTensor<T> {
  /// Ravels the index into the 1-D vector
  ///
  /// ```
  /// use mleap_tensor::{DenseTensor, Ravel};
  ///
  /// let tensor = DenseTensor::new(vec![2, 3, 2], vec![45, 67, 89, 23, 43, 66,  98, 34, 23, 23, 44, 54, 76]);
  /// assert_eq!(tensor.ravel_index(&vec![0, 1, 1]).unwrap(), 3);
  /// assert_eq!(tensor.ravel_index(&vec![0, 23, 4]), None);
  /// ```
  fn ravel_index(&self, index: &[usize]) -> Option<usize> {
    if index.len() > self.dims.len() { return None }

    index.iter().enumerate().fold(Some(0), |acc, x| {
      let (i, r) = x;
      if *r > self.dims[i] { return None }
      acc.map(|a| a + *r * self.dim_stride(i))
    })
  }
}
