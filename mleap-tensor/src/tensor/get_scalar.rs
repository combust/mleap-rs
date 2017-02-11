use tensor::DenseTensor;
use tensor::ravel::*;

pub trait GetScalar<T> {
  fn get_scalar(&self, index: &[usize]) -> Option<&T>;
}

impl<T> GetScalar<T> for DenseTensor<T> {
  /// Gets a scalar value at a given index
  ///
  ///
  /// ```
  /// use mleap_tensor::{DenseTensor, GetScalar};
  ///
  /// let tensor = DenseTensor::new(vec![2, 3, 2], vec![45, 67, 89, 23, 43, 66,  98, 34, 23, 23, 44, 54, 76]);
  /// assert_eq!(*tensor.get_scalar(&vec![0, 1, 1]).unwrap(), 23);
  /// assert_eq!(tensor.get_scalar(&vec![0, 23, 4]), None);
  /// ```
  fn get_scalar(&self, index: &[usize]) -> Option<&T> {
    self.ravel_index(index).map(|index| &self.values[index])
  }
}
