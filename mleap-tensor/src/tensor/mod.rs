mod broadcast;
mod broadcast_iter;
mod dimensions;
mod dims;
mod get_scalar;
mod ravel;

pub use self::broadcast::*;
pub use self::broadcast_iter::*;
pub use self::dimensions::*;
pub use self::dims::*;
pub use self::get_scalar::*;
pub use self::ravel::*;

use std::rc::Rc;

pub enum Tensor<T> {
  Dense(DenseTensor<T>),
  Sparse(SparseTensor<T>)
}

/// `DenseTensor`
///
/// 1. Uses row-major data storage
#[derive(Debug, Clone)]
pub struct DenseTensor<T> {
  dims: Dimensions,
  values: Rc<Vec<T>>
}

/// `SparseTensor`
#[derive(Debug, Clone)]
pub struct SparseTensor<T> {
  dims: Dimensions,
  values: Vec<T>,
  keys: Vec<usize>
}

impl<T> DenseTensor<T> {
  /// Create a new `DenseTensor`
  ///
  /// ```
  /// use mleap_tensor::{DenseTensor, Dimensions};
  ///
  /// let tensor = DenseTensor::new(vec![2, 3], vec![45, 67, 89,  98, 34, 23]);
  /// ```
  pub fn new<D: Into<Dimensions>>(dims: D, values: Vec<T>) -> DenseTensor<T> {
    DenseTensor {
      dims: dims.into(),
      values: Rc::new(values)
    }
  }

  /// Get the stride for a given dimension
  ///
  /// ```
  /// use mleap_tensor::{DenseTensor, Dimensions};
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
