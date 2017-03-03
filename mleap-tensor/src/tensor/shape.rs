use std::ops::{Range, Deref};
use std::iter::{Iterator, repeat};
use std::cmp;

use broadcast;

#[derive(Debug, Clone)]
pub struct TensorShape {
  dimensions: Vec<usize>
}

pub struct DenseStrideIter<'a> {
  dimensions: &'a [usize],
  range: Range<usize>
}

fn is_broadcast_compat(a: &TensorShape, b: &TensorShape) -> bool {
  a.iter().rev().zip(b.iter().rev()).all(|(&a, &b)| {
    a == b || a == 1 || b == 1
  })
}

fn broadcast_shape_from_iter(a: &mut Iterator<Item=usize>,
                             b: &mut Iterator<Item=usize>) -> Vec<usize> {
  a.zip(b).map(|(a, b)| {
    if a == b { a }
    else if a == 1 { b }
    else { a }
  }).collect()
}

fn broadcast_shape(a: &[usize], b: &[usize]) -> TensorShape {
  let diff = (a.len() as isize) - (b.len() as isize);

  let dims = if diff == 0 {
    let mut a1 = a.iter().map(|x| *x);
    let mut b1 = b.iter().map(|x| *x);

    broadcast_shape_from_iter(&mut a1, &mut b1)
  } else if diff < 0 {
    let mut a1 = repeat(1 as usize).take((diff.abs() as usize)).
      chain(a.iter().map(|x| *x));
    let mut b1 = b.iter().map(|x| *x);

    broadcast_shape_from_iter(&mut a1, &mut b1)
  } else {
    let mut a1 = a.iter().map(|x| *x);
    let mut b1 = repeat(1 as usize).take((diff.abs() as usize)).
      chain(b.iter().map(|x| *x));

    broadcast_shape_from_iter(&mut a1, &mut b1)
  };

  TensorShape {
    dimensions: dims
  }
}

fn dense_broadcast_dimensions_from_iter(a: &mut Iterator<Item=usize>,
                                        broadcast: &mut Iterator<Item=usize>,
                                        strides: &mut Iterator<Item=usize>) -> Vec<broadcast::Dimension> {
  a.zip(strides).zip(broadcast).map(|((a, s), b)| {
    let stride = if a == 1 { 0 }
    else { s };

    let target = cmp::max(a, b);

    broadcast::Dimension {
      stride: stride,
      size: s,
      target: target
    }
  }).collect()
}

fn dense_broadcast_dimensions(a: &TensorShape, b: &TensorShape) -> Vec<broadcast::Dimension> {
  let bshape = broadcast_shape(a, b);
  let mut biter = bshape.iter().map(|x| *x);
  let diff = (a.len() as isize) - (b.len() as isize);
  if diff >= 0 {
    let mut a1 = a.iter().map(|x| *x);

    dense_broadcast_dimensions_from_iter(&mut a1, &mut biter, &mut a.dense_strides_iter())
  } else {
    let da = diff.abs() as usize;
    let mut a1 = repeat(1 as usize).take(da).
      chain(a.iter().map(|x| *x));
    let mut strides = repeat(a.big_stride()).take(da).
      chain(a.dense_strides_iter());

    dense_broadcast_dimensions_from_iter(&mut a1, &mut biter, &mut strides)
  }
}

impl TensorShape {
  pub fn new(dimensions: Vec<usize>) -> TensorShape {
    TensorShape {
      dimensions: dimensions
    }
  }

  /// Returns the broadcasted shape between two shapes.
  ///
  /// ```
  /// use mleap_tensor::TensorShape;
  ///
  /// let shape1 = TensorShape::new(vec![5, 2, 3, 1, 1]);
  /// let shape2 = TensorShape::new(vec![3, 1, 7]);
  /// let shape3 = TensorShape::new(vec![1]);
  /// let shape4 = TensorShape::new(vec![5, 6, 12]);
  ///
  /// let bshape1 = shape1.broadcast_shape(&shape2);
  /// let bshape2 = shape3.broadcast_shape(&shape2);
  /// let bshape3 = shape1.broadcast_shape(&shape4);
  ///
  /// assert!(bshape1.is_some());
  /// assert_eq!(bshape1.unwrap().to_vec(), vec![5, 2, 3, 1, 7]);
  ///
  /// assert!(bshape2.is_some());
  /// assert_eq!(bshape2.unwrap().to_vec(), vec![3,  1, 7]);
  ///
  /// assert!(bshape3.is_none());
  /// ```
  pub fn broadcast_shape(&self, other: &TensorShape) -> Option<TensorShape> {
    if self.is_broadcast_compat(other) {
      Some(broadcast_shape(&self, &other))
    } else { None }
  }

  /// Returns the broadcasted dimensions from this shape to `other`.
  ///
  /// ```
  /// use mleap_tensor::TensorShape;
  /// use mleap_tensor::broadcast;
  ///
  /// let shape1 = TensorShape::new(vec![5, 2, 3, 1, 2]);
  /// let shape2 = TensorShape::new(vec![3, 1, 2]);
  /// let shape3 = TensorShape::new(vec![1]);
  /// let shape4 = TensorShape::new(vec![5, 6, 12]);
  ///
  /// let bdims1 = shape1.dense_broadcast_dimensions(&shape2).unwrap();
  /// let bdims2 = shape2.dense_broadcast_dimensions(&shape1).unwrap();
  /// let bdims3 = shape3.dense_broadcast_dimensions(&shape4).unwrap();
  /// let bdims4 = shape4.dense_broadcast_dimensions(&shape3).unwrap();
  /// let bdims5 = shape1.dense_broadcast_dimensions(&shape4);
  ///
  /// assert!(bdims5.is_none());
  ///
  /// assert_eq!(bdims1[0], broadcast::Dimension { stride: 12, size: 12, target: 5 });
  /// assert_eq!(bdims1[1], broadcast::Dimension { stride: 6, size: 6, target: 2 });
  /// assert_eq!(bdims1[2], broadcast::Dimension { stride: 2, size: 2, target: 3 });
  /// assert_eq!(bdims1[3], broadcast::Dimension { stride: 0, size: 2, target: 1 });
  /// assert_eq!(bdims1[4], broadcast::Dimension { stride: 1, size: 1, target: 2 });
  ///
  /// assert_eq!(bdims2[0], broadcast::Dimension { stride: 0, size: 6, target: 5 });
  /// assert_eq!(bdims2[1], broadcast::Dimension { stride: 0, size: 6, target: 2 });
  /// assert_eq!(bdims2[2], broadcast::Dimension { stride: 2, size: 2, target: 3 });
  /// assert_eq!(bdims2[3], broadcast::Dimension { stride: 0, size: 2, target: 1 });
  /// assert_eq!(bdims2[4], broadcast::Dimension { stride: 1, size: 1, target: 2 });
  ///
  /// assert_eq!(bdims3[0], broadcast::Dimension { stride: 0, size: 1, target: 5 });
  /// assert_eq!(bdims3[1], broadcast::Dimension { stride: 0, size: 1, target: 6 });
  /// assert_eq!(bdims3[2], broadcast::Dimension { stride: 0, size: 1, target: 12 });
  ///
  /// assert_eq!(bdims4[0], broadcast::Dimension { stride: 72, size: 72, target: 5 });
  /// assert_eq!(bdims4[1], broadcast::Dimension { stride: 12, size: 12, target: 6 });
  /// assert_eq!(bdims4[2], broadcast::Dimension { stride: 1, size: 1, target: 12 });
  /// ```
  pub fn dense_broadcast_dimensions(&self, other: &TensorShape) -> Option<Vec<broadcast::Dimension>> {
    if self.is_broadcast_compat(other) {
      Some(dense_broadcast_dimensions(&self, &other))
    } else { None }
  }

  /// Returns if two shapes are compatible for broadcast operations.
  ///
  /// ```
  /// use mleap_tensor::TensorShape;
  ///
  /// let shape1 = TensorShape::new(vec![5, 2, 3, 1, 1]);
  /// let shape2 = TensorShape::new(vec![3, 1, 7]);
  /// let shape3 = TensorShape::new(vec![1]);
  /// let shape4 = TensorShape::new(vec![5, 6, 12]);
  /// let shape5 = TensorShape::new(vec![1, 1, 1]);
  ///
  /// assert!(shape1.is_broadcast_compat(&shape2));
  /// assert!(shape1.is_broadcast_compat(&shape3));
  /// assert!(!shape1.is_broadcast_compat(&shape4));
  /// assert!(shape2.is_broadcast_compat(&shape3));
  /// assert!(!shape2.is_broadcast_compat(&shape4));
  /// assert!(shape3.is_broadcast_compat(&shape4));
  /// assert!(shape4.is_broadcast_compat(&shape5));
  /// ```
  pub fn is_broadcast_compat(&self, other: &TensorShape) -> bool {
    is_broadcast_compat(self, other)
  }

  /// Returns the size of the buffer.
  ///
  /// ```
  /// use mleap_tensor::TensorShape;
  ///
  /// let shape = TensorShape::new(vec![4, 50, 3]);
  ///
  /// assert_eq!(shape.big_stride(), 600);
  /// ```
  pub fn big_stride(&self) -> usize {
    self.dimensions.iter().product()
  }

  /// Returns an iterator over the stride sizes for dense data.
  ///
  /// ```
  /// use mleap_tensor::TensorShape;
  ///
  /// let shape = TensorShape::new(vec![6, 5, 4, 1, 3, 2]);
  /// let strides: Vec<usize> = shape.dense_strides_iter().collect();
  ///
  /// assert_eq!(strides, vec![120, 24, 6, 6, 2, 1]);
  /// ```
  pub fn dense_strides_iter<'a>(&'a self) -> DenseStrideIter<'a> {
    DenseStrideIter {
      dimensions: &self.dimensions,
      range: 1..(self.dimensions.len() + 1)
    }
  }
}

impl Deref for TensorShape {
  type Target = [usize];

  fn deref(&self) -> &[usize] {
    &self.dimensions
  }
}

impl<'a> Iterator for DenseStrideIter<'a> {
  type Item = usize;

  fn next(&mut self) -> Option<usize> {
    self.range.next().map(|i| {
      self.dimensions.iter().rev().
        take(self.dimensions.len() - i).
        fold(1, |acc, size| acc * size)
    })
  }
}

