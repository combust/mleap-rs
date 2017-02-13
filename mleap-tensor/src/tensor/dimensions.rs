use std::ops::{Index, Range};
use std::slice::Iter;
use std::iter::Iterator;
use std::cmp::PartialEq;
use std::convert::From;

#[derive(Debug, Clone)]
pub struct Dimensions(Vec<usize>);

impl Dimensions {
  /// Create a new Dimensions object from something that can convert to it
  ///
  /// ```
  /// use mleap_tensor::Dimensions;
  ///
  /// let d = Dimensions::new(vec![2, 323, 213, 4]);
  /// ```
  pub fn new<D: Into<Dimensions>>(d: D) -> Dimensions { d.into() }

  /// Return the number of dimensions
  ///
  /// ```
  /// use mleap_tensor::Dimensions;
  ///
  /// let d = Dimensions::new(vec![23, 4, 2, 33]);
  ///
  /// assert_eq!(d.len(), 4);
  /// ```
  pub fn len(&self) -> usize { self.0.len() }

  pub fn iter(&self) -> Iter<usize> {
    self.0.iter()
  }

  pub fn stride_iter<'a>(&'a self) -> StrideIter<'a> {
    StrideIter {
      dimensions: self,
      index: 0
    }
  }

  /// If a two `Dimensions` can broadcast to each other
  ///
  /// ```
  /// use mleap_tensor::Dimensions;
  ///
  /// let d1 = Dimensions::new(vec![1, 23, 34, 22]);
  /// let d2 = Dimensions::new(vec![23, 34, 1]);
  /// let d3 = Dimensions::new(vec![24, 34, 22]);
  ///
  /// assert!(d1.can_broadcast(&d2));
  /// assert!(!d1.can_broadcast(&d3));
  /// assert!(!d3.can_broadcast(&d1));
  /// ```
  pub fn can_broadcast(&self, other: &Dimensions) -> bool {
    for (&a, &b) in self.iter().rev().zip(other.iter().rev()) {
      if a != 1 && b != 1 && a != b  { return false }
    }

    true
  }

  pub fn last(&self) -> Option<&usize> { self.0.last() }
}

impl Index<usize> for Dimensions {
  type Output = usize;

  fn index(&self, index: usize) -> &usize { self.0.index(index) }
}

impl Index<Range<usize>> for Dimensions {
  type Output = [usize];

  fn index(&self, range: Range<usize>) -> &[usize] { self.0.index(range) }
}

impl From<Vec<usize>> for Dimensions {
  fn from(v: Vec<usize>) -> Dimensions { Dimensions(v) }
}

impl PartialEq for Dimensions {
  fn eq(&self, other: &Dimensions) -> bool {
    self.0 == other.0
  }
}

pub struct StrideIter<'a> {
  dimensions: &'a Dimensions,
  index: usize
}

impl<'a> Iterator for StrideIter<'a> {
  type Item = usize;

  fn next(&mut self) -> Option<usize> {
    let len = self.dimensions.len();
    self.index += 1;

    if self.index < len {
      Some(self.dimensions[self.index..len].iter().product())
    } else if self.index == len {
      Some(1)
    } else { None }
  }
}
