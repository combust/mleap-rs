use std::ops::Index;

use broadcast;

mod iter;
mod shape;

pub use self::iter::*;
pub use self::shape::*;

pub trait Tensor<T> {
  fn shape(&self) -> &TensorShape;
  fn broadcast_iter<'a>(&'a self, other: &Tensor<T>, drop: usize) -> Option<BroadcastIter<'a, T>>;

  fn get_scalar(&self, index: & [usize]) -> Option<&T>;
}

pub trait DenseTensor<T>: Tensor<T> { }
pub trait SparseTensor<T>: Tensor<T> { }

pub struct Dense<T> {
  shape: TensorShape,
  buffer: Vec<T>
}

pub struct Sparse<T> {
  shape: TensorShape,
  _keys: Vec<Vec<usize>>,
  _buffer: Vec<T>
}

impl<T> Dense<T> {
  pub fn new(shape: TensorShape, buffer: Vec<T>) -> Dense<T> {
    Dense {
      shape: shape,
      buffer: buffer
    }
  }

  /// Create a broadcasting iterator for this dense data.
  ///
  /// ```
  /// use mleap_tensor::broadcast;
  /// use mleap_tensor::tensor::*;
  ///
  /// let d1: Dense<f32> = Dense::new(TensorShape::new(vec![1, 2, 2]), vec![23.4, 32.2, 56.7, 89.1]);
  /// let d2: Dense<f32> = Dense::new(TensorShape::new(vec![1, 1, 2]), vec![1.0, 2.0]);
  ///
  /// let bdims1 = d1.shape().dense_broadcast_dimensions(d2.shape()).unwrap();
  /// let mut i1 = d1.dense_broadcast_iter(&d2, 1).unwrap();
  ///
  /// assert_eq!(bdims1[0], broadcast::Dimension { stride: 0, size: 4, target: 1 });
  /// assert_eq!(bdims1[1], broadcast::Dimension { stride: 2, size: 2, target: 2 });
  /// assert_eq!(bdims1[2], broadcast::Dimension { stride: 1, size: 1, target: 2 });
  ///
  /// assert_eq!(i1.dimensions.len(), 2);
  /// assert_eq!(i1.dimensions[0], broadcast::Dimension { stride: 0, size: 4, target: 1 });
  /// assert_eq!(i1.dimensions[1], broadcast::Dimension { stride: 2, size: 2, target: 2 });
  /// assert!(i1.iterators.is_none());
  ///
  /// assert_eq!(i1.next().map(|x| x.to_vec()), Some(vec![23.4, 32.2]));
  /// assert_eq!(i1.next().map(|x| x.to_vec()), Some(vec![56.7, 89.1]));
  /// assert!(i1.next().is_none());
  /// assert!(i1.next().is_none());
  /// assert!(i1.next().is_none());
  ///
  /// let mut i2 = d2.dense_broadcast_iter(&d1, 1).unwrap();
  ///
  /// assert_eq!(i2.next().map(|x| x.to_vec()), Some(vec![1.0, 2.0]));
  /// assert_eq!(i2.next().map(|x| x.to_vec()), Some(vec![1.0, 2.0]));
  /// assert!(i2.next().is_none());
  /// assert!(i2.next().is_none());
  /// assert!(i2.next().is_none());
  /// ```
  pub fn dense_broadcast_iter<'a>(&'a self, other: &Tensor<T>, drop: usize) -> Option<DenseBroadcastIter<'a, T>> {
    self.shape.dense_broadcast_dimensions(other.shape()).map(|dims| {
      let short_dims: Vec<broadcast::Dimension> = dims.iter().take(dims.len() - drop).map(|x| *x).collect();
      DenseBroadcastIter::new(&self.buffer, short_dims)
    })
  }

  /// Ravels the index into the 1-D vector
  ///
  /// ```
  /// use mleap_tensor::{Dense, TensorShape};
  ///
  /// let tensor = Dense::new(TensorShape::new(vec![2, 3, 2]), vec![45, 67, 89, 23, 43, 66,  98, 34, 23, 23, 44, 54, 76]);
  ///
  /// assert_eq!(tensor.ravel_index(&vec![0, 1, 1]).unwrap(), 3);
  /// assert_eq!(tensor.ravel_index(&vec![0, 23, 4]), None);
  /// ```
  pub fn ravel_index(&self, index: &[usize]) -> Option<usize> {
    if index.len() != self.shape().len() { return None }

    index.iter().zip(self.shape().iter().zip(self.shape().dense_strides_iter())).fold(Some(0), |acc, (&index, (&dim, stride))| {
      if index < dim {
        acc.map(|a| a + index * stride)
      } else { None }
    })
  }
}

impl<T> Tensor<T> for Dense<T> {
  fn shape(&self) -> &TensorShape { &self.shape }

  fn broadcast_iter<'a>(&'a self, other: &Tensor<T>, drop: usize) -> Option<BroadcastIter<'a, T>> {
    self.dense_broadcast_iter(other, drop).map(|iter| BroadcastIter::Dense(Box::new(iter)))
  }

  /// Get scalar value at specified index.
  ///
  /// ```
  /// use mleap_tensor::*;
  ///
  /// let tensor = Dense::new(TensorShape::new(vec![2, 3, 2]), vec![45, 67, 89, 23, 43, 66,  98, 34, 23, 23, 44, 54, 76]);
  ///
  /// assert_eq!(tensor.get_scalar(&vec![1, 1, 1]).map(|x| *x), Some(23));
  /// assert!(tensor.get_scalar(&vec![2, 1, 1]).is_none());
  /// ```
  fn get_scalar(&self, index: & [usize]) -> Option<&T> {
    self.ravel_index(index).and_then(|index| self.buffer.get(index))
  }
}

//impl<T> Tensor<T> for Sparse<T> {
  //fn shape(&self) -> &TensorShape { &self.shape }

  //fn broadcast_iter<'a>(&'a self, _other: &Tensor<T>, _drop: usize) -> Option<BroadcastIter<'a, T>>  {
    //None
  //}
//}

//pub trait DenseTensor<T>: Tensor<T> {
  //fn left_slice<'a>(&'a self, indices: &[usize]) -> Option<DenseSlice<'a, T>> {
    //if indices.len() <= self.dims().len() {
      //self.dims().iter().zip(self.dims().stride_iter()).zip(indices.iter()).fold(Some((0, self.values().len())), { |result, ((&dim, stride), &index)|
        //result.and_then({ |(start, end)|
          //if index < dim {
            //Some((start + index * stride, end - (dim - index) * stride))
          //} else { None }
        //})
      //}).map(|(start, end)| {
        //DenseSlice {
          //values: &self.values()[start..end],
          //dims: self.dims().iter().map(|x| *x).skip(indices.len()).collect::<Vec<usize>>().into()
        //}
      //})
    //} else { None }
  //}

  ///// Ravels the index into the 1-D vector
  /////
  ///// ```
  ///// use mleap_tensor::{DenseRaw, DenseTensor};
  /////
  ///// let tensor = DenseRaw::new(vec![2, 3, 2], vec![45, 67, 89, 23, 43, 66,  98, 34, 23, 23, 44, 54, 76]);
  ///// assert_eq!(tensor.ravel_index(&vec![0, 1, 1]).unwrap(), 3);
  ///// assert_eq!(tensor.ravel_index(&vec![0, 23, 4]), None);
  ///// ```
  //fn ravel_index(&self, index: &[usize]) -> Option<usize> {
    //if index.len() != self.dims().len() { return None }

    //index.iter().zip(self.dims().iter().zip(self.dims().stride_iter())).fold(Some(0), |acc, (&index, (&dim, stride))| {
      //if index < dim {
        //acc.map(|a| a + index * stride)
      //} else { None }
    //})
  //}

  ///// Gets a scalar value at a given index
  /////
  /////
  ///// ```
  ///// use mleap_tensor::{DenseRaw, DenseTensor};
  /////
  ///// let tensor = DenseRaw::new(vec![2, 3, 2], vec![45, 67, 89, 23, 43, 66,  98, 34, 23, 23, 44, 54, 76]);
  ///// assert_eq!(*tensor.get_scalar(&vec![0, 1, 1]).unwrap(), 23);
  ///// assert_eq!(tensor.get_scalar(&vec![0, 23, 4]), None);
  ///// ```
  //fn get_scalar(&self, index: &[usize]) -> Option<&T> {
    //self.ravel_index(index).map(|index| &self.values()[index])
  //}
//}

//pub trait SparseTensor<T>: Tensor<T> { }

///// `DenseTensor`
/////
///// 1. Uses row-major data storage
//#[derive(Debug, Clone)]
//pub struct DenseRaw<T> {
  //values: Vec<T>,
  //dims: Dimensions
//}

//#[derive(Debug, Clone)]
//pub struct DenseSlice<'a, T> where T: 'a {
  //values: &'a [T],
  //dims: Dimensions
//}

///// `SparseTensor`
//#[derive(Debug, Clone)]
//pub struct SparseRaw<T> {
  //dims: Dimensions,
  //values: Vec<T>,
  //keys: Vec<usize>
//}

//impl<T> DenseRaw<T> {
  ///// Create a new `DenseTensor`
  /////
  ///// ```
  ///// use mleap_tensor::{DenseRaw, Dimensions};
  /////
  ///// let tensor = DenseRaw::new(vec![2, 3], vec![45, 67, 89,  98, 34, 23]);
  ///// ```
  //pub fn new<D: Into<Dimensions>>(dims: D, values: Vec<T>) -> DenseRaw<T> {
    //DenseRaw {
      //dims: dims.into(),
      //values: values
    //}
  //}
//}

//impl<T> Tensor<T> for DenseRaw<T> {
  //fn values(&self) -> &[T] { &self.values }
  //fn dims(&self) -> &Dimensions { &self.dims }
//}

//impl<'a, T> Tensor<T> for DenseSlice<'a, T> {
  //fn values(&self) -> &[T] { self.values }
  //fn dims(&self) -> &Dimensions { &self.dims }
//}

//impl<T> DenseTensor<T> for DenseRaw<T> { }
//impl<'a, T> DenseTensor<T> for DenseSlice<'a, T> { }
