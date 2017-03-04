use std::ops::Range;
use std::iter::{Iterator, Zip, repeat};
use std::cmp;
use std::cmp::Ordering;

pub use iter::DenseBroadcastIter;
pub use dim::BroadcastDimension;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
  IncompatibleBroadcast
}

pub struct DenseStrideIter<'a> {
  shape: &'a [usize],
  range: Range<usize>
}

#[derive(Debug, Clone)]
pub struct DenseBroadcast<'a, T: 'a> {
  // target shape for broadcast
  pub bshape: Vec<usize>,

  // actual shape of first buffer
  pub shape1: Vec<usize>,

  // broadcast shape of first buffer
  pub bdims1: Vec<BroadcastDimension>,

  // first buffer with dense values
  pub buf1: &'a [T],

  // actual shape of second buffer
  pub shape2: Vec<usize>,

  // broadcast shape of second buffer
  pub bdims2: Vec<BroadcastDimension>,

  // second buffer with dense values
  pub buf2: &'a [T]
}

/// Whether two shapes are compatible for broadcasting.
///
/// ```
/// use mleap_tensor::core::broadcast::compatible;
///
/// let shape1: Vec<usize> = vec![5, 2, 3, 1, 1];
/// let shape2: Vec<usize> = vec![3, 1, 7];
/// let shape3: Vec<usize> = vec![1];
/// let shape4: Vec<usize> = vec![5, 6, 12];
/// let shape5: Vec<usize> = vec![1, 1, 1];
///
/// assert!(compatible(&shape1, &shape2));
/// assert!(compatible(&shape1, &shape3));
/// assert!(!compatible(&shape1, &shape4));
/// assert!(compatible(&shape2, &shape3));
/// assert!(!compatible(&shape2, &shape4));
/// assert!(compatible(&shape3, &shape4));
/// assert!(compatible(&shape4, &shape5));
/// ```
pub fn compatible<'a>(shape1: &'a [usize],
                      shape2: &'a [usize]) -> bool {
  shape1.iter().rev().zip(shape2.iter().rev()).all(|(&a, &b)| {
    a == b || a == 1 || b == 1
  })
}

fn broadcast_shape<'a>(shape: &'a [usize],
                       bshape: &'a [usize]) -> Vec<BroadcastDimension> {
  let slen = shape.len();
  let tlen = bshape.len();
  let mut strides = DenseStrideIter::new(shape);

  match slen.cmp(&tlen) {
    Ordering::Equal => {
      BroadcastDimension::shape_from_iters(&mut shape.iter(), &mut bshape.iter(), &mut strides)
    },
    Ordering::Less => {
      let one: usize = 1;
      let big_stride = shape.iter().product();

      let mut strides2 = repeat(big_stride).take(tlen - slen).
        chain(&mut strides);
      let mut siter = repeat(&one).take(tlen - slen).
        chain(shape.iter());

      BroadcastDimension::shape_from_iters(&mut siter, &mut bshape.iter(), &mut strides2)
    },
    Ordering::Greater => panic!("shape cannot be larger than broadcast shape!")
  }
}

fn target_shape<'a>(shape1: &'a [usize],
                    shape2: &'a [usize]) -> Vec<usize> {
  let len1 = shape1.len();
  let len2 = shape2.len();

  match len1.cmp(&len2) {
    Ordering::Equal => {
      shape1.iter().zip(shape2.iter()).
        map(|(&a, &b)| cmp::max(a, b)).
        collect()
    },
    Ordering::Less => {
      let one: usize = 1;

      repeat(&one).take(len2 - len1).chain(shape1.iter()).zip(shape2.iter()).
        map(|(&a, &b)| cmp::max(a, b)).
        collect()
    },
    Ordering::Greater => {
      let one: usize = 1;

      shape1.iter().zip(repeat(&one).take(len1 - len2).chain(shape2.iter())).
        map(|(&a, &b)| cmp::max(a, b)).
        collect()
    }
  }
}

/// Attempts to create a dense broadcast between two tensors.
///
/// Both input tensors must be dense tensors.
///
/// ```
/// use mleap_tensor::core::broadcast::{BroadcastDimension, Error, dense_broadcast};
///
/// let shape1: Vec<usize> = vec![2, 1, 3];
/// let buf1: Vec<f32> = vec![23.4, 34.3, 55.4, 44.5, 23.1, 88.9];
/// let shape2: Vec<usize> = vec![3];
/// let buf2: Vec<f32> = vec![1.0, 2.0, 3.0];
/// let shape3: Vec<usize> = vec![3, 2, 1];
/// let buf3 = buf1.clone();
///
/// let b1 = dense_broadcast(&shape1, &buf1, &shape2, &buf2).unwrap();
/// assert_eq!(b1.bshape, &[2, 1, 3]);
/// assert_eq!(b1.shape1, shape1.as_slice());
/// assert_eq!(b1.bdims1, &[BroadcastDimension { size: 3, stride: 3, target: 2 },
///                          BroadcastDimension { size: 3, stride: 0, target: 1 },
///                          BroadcastDimension { size: 1, stride: 1, target: 3 }]);
/// assert_eq!(b1.buf1, buf1.as_slice());
/// assert_eq!(b1.shape2, shape2.as_slice());
/// assert_eq!(b1.bdims2, &[BroadcastDimension { size: 3, stride: 0, target: 2 },
///                          BroadcastDimension { size: 3, stride: 0, target: 1 },
///                          BroadcastDimension { size: 1, stride: 1, target: 3 }]);
/// assert_eq!(b1.buf2, buf2.as_slice());
///
/// let b2 = dense_broadcast(&shape1, &buf1, &shape3, &buf3).err().unwrap();
/// assert_eq!(b2, Error::IncompatibleBroadcast);
/// ```
pub fn dense_broadcast<'a, T: 'a>(shape1: &'a [usize],
                                  buf1: &'a [T],
                                  shape2: &'a [usize],
                                  buf2: &'a [T]) -> Result<DenseBroadcast<'a, T>, Error> {
  if compatible(shape1, shape2) {
    let bshape = target_shape(shape1, shape2);
    let bdims1 = broadcast_shape(shape1, &bshape);
    let bdims2 = broadcast_shape(shape2, &bshape);

    Ok(DenseBroadcast {
      bshape: bshape,
      shape1: shape1.to_vec(),
      bdims1: bdims1,
      buf1: buf1,
      shape2: shape2.to_vec(),
      bdims2: bdims2,
      buf2: buf2
    })
  } else { Err(Error::IncompatibleBroadcast) }
}

impl<'a, T: 'a> DenseBroadcast<'a, T> {
  pub fn chop(&mut self, n: usize) -> &mut Self {
    let len = self.bshape.len() - n;

    self.bshape.truncate(len);

    self.shape1.truncate(len);
    self.bdims1.truncate(len);

    self.shape2.truncate(len);
    self.bdims2.truncate(len);

    self
  }

  pub fn bshape_push(&mut self, n: usize) -> &mut Self {
    self.bshape.push(n);
    self
  }

  pub fn iters(&self) -> (Box<Iterator<Item=&'a [T]> + 'a>, Box<Iterator<Item=&'a [T]> + 'a>) {
    (Box::new(DenseBroadcastIter::new(&self.bdims1, self.buf1)),
    Box::new(DenseBroadcastIter::new(&self.bdims2, self.buf2)))
  }

  /// Does the thing
  ///
  /// ```
  /// use mleap_tensor::core::broadcast;
  ///
  /// let shape1: Vec<usize> = vec![3, 2, 1];
  /// let shape2: Vec<usize> = vec![3, 2, 1];
  ///
  /// let buf1: Vec<f32> = vec![12.3, 33.4, 44.1, 56.7, 34.2, 23.5];
  /// let buf2: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
  ///
  /// let db = broadcast::dense_broadcast(&shape1, &buf1, &shape2, &buf2).unwrap();
  /// let mut i = db.iter().map(|(a, b)| a[0] + b[0]);
  ///
  /// assert_eq!(i.size_hint(), (6, Some(6)));
  /// assert_eq!(i.next().unwrap(), buf1[0] + buf2[0]);
  /// assert_eq!(i.next().unwrap(), buf1[1] + buf2[1]);
  /// assert_eq!(i.next().unwrap(), buf1[2] + buf2[2]);
  /// assert_eq!(i.next().unwrap(), buf1[3] + buf2[3]);
  /// assert_eq!(i.next().unwrap(), buf1[4] + buf2[4]);
  /// assert_eq!(i.next().unwrap(), buf1[5] + buf2[5]);
  ///
  /// assert!(i.next().is_none());
  /// assert!(i.next().is_none());
  /// ```
  pub fn iter(&self) -> Zip<Box<Iterator<Item=&'a [T]> + 'a>, Box<Iterator<Item=&'a [T]> + 'a>> {
    let (i1, i2) = self.iters();
    i1.zip(i2)
  }
}

impl<'a> DenseStrideIter<'a> {
  pub fn new(shape: &'a [usize]) -> DenseStrideIter<'a> {
    DenseStrideIter {
      shape: shape,
      range: 1..(shape.len() + 1)
    }
  }
}

impl<'a> Iterator for DenseStrideIter<'a> {
  type Item = usize;

  fn next(&mut self) -> Option<usize> {
    self.range.next().map(|i| {
      self.shape.iter().rev().
        take(self.shape.len() - i).
        fold(1, |acc, size| acc * size)
    })
  }
}


