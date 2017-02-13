mod dimensions;

pub use self::dimensions::*;

pub trait Tensor<T> {
  fn dims(&self) -> &Dimensions;
  fn values(&self) -> &[T];
}

pub trait DenseTensor<T>: Tensor<T> {
  fn left_slice<'a>(&'a self, indices: &[usize]) -> Option<DenseSlice<'a, T>> {
    if indices.len() <= self.dims().len() {
      self.dims().iter().zip(self.dims().stride_iter()).zip(indices.iter()).fold(Some((0, self.values().len())), { |result, ((&dim, stride), &index)|
        result.and_then({ |(start, end)|
          if index < dim {
            Some((start + index * stride, end - (dim - index) * stride))
          } else { None }
        })
      }).map(|(start, end)| {
        DenseSlice {
          values: &self.values()[start..end],
          dims: self.dims().iter().map(|x| *x).skip(indices.len()).collect::<Vec<usize>>().into()
        }
      })
    } else { None }
  }

  /// Ravels the index into the 1-D vector
  ///
  /// ```
  /// use mleap_tensor::{DenseRaw, DenseTensor};
  ///
  /// let tensor = DenseRaw::new(vec![2, 3, 2], vec![45, 67, 89, 23, 43, 66,  98, 34, 23, 23, 44, 54, 76]);
  /// assert_eq!(tensor.ravel_index(&vec![0, 1, 1]).unwrap(), 3);
  /// assert_eq!(tensor.ravel_index(&vec![0, 23, 4]), None);
  /// ```
  fn ravel_index(&self, index: &[usize]) -> Option<usize> {
    if index.len() != self.dims().len() { return None }

    index.iter().zip(self.dims().iter().zip(self.dims().stride_iter())).fold(Some(0), |acc, (&index, (&dim, stride))| {
      if index < dim {
        acc.map(|a| a + index * stride)
      } else { None }
    })
  }

  /// Gets a scalar value at a given index
  ///
  ///
  /// ```
  /// use mleap_tensor::{DenseRaw, DenseTensor};
  ///
  /// let tensor = DenseRaw::new(vec![2, 3, 2], vec![45, 67, 89, 23, 43, 66,  98, 34, 23, 23, 44, 54, 76]);
  /// assert_eq!(*tensor.get_scalar(&vec![0, 1, 1]).unwrap(), 23);
  /// assert_eq!(tensor.get_scalar(&vec![0, 23, 4]), None);
  /// ```
  fn get_scalar(&self, index: &[usize]) -> Option<&T> {
    self.ravel_index(index).map(|index| &self.values()[index])
  }
}

pub trait SparseTensor<T>: Tensor<T> { }

/// `DenseTensor`
///
/// 1. Uses row-major data storage
#[derive(Debug, Clone)]
pub struct DenseRaw<T> {
  values: Vec<T>,
  dims: Dimensions
}

#[derive(Debug, Clone)]
pub struct DenseSlice<'a, T> where T: 'a {
  values: &'a [T],
  dims: Dimensions
}

/// `SparseTensor`
#[derive(Debug, Clone)]
pub struct SparseRaw<T> {
  dims: Dimensions,
  values: Vec<T>,
  keys: Vec<usize>
}

impl<T> DenseRaw<T> {
  /// Create a new `DenseTensor`
  ///
  /// ```
  /// use mleap_tensor::{DenseRaw, Dimensions};
  ///
  /// let tensor = DenseRaw::new(vec![2, 3], vec![45, 67, 89,  98, 34, 23]);
  /// ```
  pub fn new<D: Into<Dimensions>>(dims: D, values: Vec<T>) -> DenseRaw<T> {
    DenseRaw {
      dims: dims.into(),
      values: values
    }
  }
}

impl<T> Tensor<T> for DenseRaw<T> {
  fn values(&self) -> &[T] { &self.values }
  fn dims(&self) -> &Dimensions { &self.dims }
}

impl<'a, T> Tensor<T> for DenseSlice<'a, T> {
  fn values(&self) -> &[T] { self.values }
  fn dims(&self) -> &Dimensions { &self.dims }
}

impl<T> DenseTensor<T> for DenseRaw<T> { }
impl<'a, T> DenseTensor<T> for DenseSlice<'a, T> { }
