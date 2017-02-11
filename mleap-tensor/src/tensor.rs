use std::rc::Rc;

/// `DenseTensor`
///
/// 1. Uses row-major data storage
#[derive(Debug, Clone)]
pub struct DenseTensor<T> {
  dims: Vec<usize>,
  values: Rc<Vec<T>>
}

/// `SparseTensor`
///
/// Refer to: `http://ieee-hpec.org/2012/index_htm_files/Baskaranpaper.pdf for storage format choice`
#[derive(Debug, Clone)]
pub struct SparseTensor<T> {
  dims: Vec<usize>,
  values: Vec<T>,
  keys: Vec<Vec<usize>>
}

pub trait Dims {
  fn dims(&self) -> &[usize];
}

pub trait Ravel {
  fn ravel_index(&self, index: &[usize]) -> Option<usize>;
}

pub trait GetScalar<T> {
  fn get_scalar(&self, index: &[usize]) -> Option<&T>;
}

impl<T> DenseTensor<T> {
  /// Create a new `DenseTensor`
  ///
  /// ```
  /// use mleap_tensor::DenseTensor;
  ///
  /// let tensor = DenseTensor::new(vec![2, 3], vec![45, 67, 89,  98, 34, 23]);
  /// ```
  pub fn new(dims: Vec<usize>, values: Vec<T>) -> DenseTensor<T> {
    DenseTensor {
      dims: dims,
      values: Rc::new(values)
    }
  }

  /// Get the stride for a given dimension
  ///
  /// ```
  /// use mleap_tensor::DenseTensor;
  ///
  /// let tensor = DenseTensor::new(vec![2, 3, 2], vec![45, 67, 89, 23, 43, 66,  98, 34, 23, 23, 44, 54, 76]);
  /// assert_eq!(tensor.dim_stride(0), 6);
  /// assert_eq!(tensor.dim_stride(1), 2);
  /// assert_eq!(tensor.dim_stride(2), 1);
  /// ```
  pub fn dim_stride(&self, dim: usize) -> usize {
    if dim == self.dims.len() - 1 { return 1 }
    self.dims[(dim + 1)..self.dims.len()].iter().product()
  }
}

impl<T> Dims for DenseTensor<T> {
  fn dims(&self) -> &[usize] { &self.dims }
}

impl<T> Dims for SparseTensor<T> {
  fn dims(&self) -> &[usize] { &self.dims }
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

impl<T> GetScalar<T> for DenseTensor<T> {
  fn get_scalar(&self, index: &[usize]) -> Option<&T> {
    self.ravel_index(index).map(|index| &self.values[index])
  }
}
