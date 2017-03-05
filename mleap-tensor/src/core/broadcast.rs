use std::ops::Range;
use std::iter::{Iterator, repeat};
use std::cmp;
use std::cmp::Ordering;
use std::rc::Rc;

pub use iter::{TensorIterator, TensorIter, DenseBroadcastIter};
pub use dim::BroadcastDimension;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
  IncompatibleBroadcast
}

pub struct DenseStrideIter<'a> {
  shape: &'a [usize],
  range: Range<usize>
}

pub struct DenseSpec<'a, T: 'a> {
  bshape: Rc<Vec<usize>>,

  bdims: Vec<BroadcastDimension>,
  buf: &'a [T],
}

pub struct DenseSpec2<'a, T: 'a> {
  bshape: Rc<Vec<usize>>,

  bdims1: Vec<BroadcastDimension>,
  buf1: &'a [T],

  bdims2: Vec<BroadcastDimension>,
  buf2: &'a [T]
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

fn broadcast_shape<'a>(bshape: &'a [usize],
                       shape: &'a [usize]) -> Vec<BroadcastDimension> {
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

pub fn dense_iter<'a, T: 'a>(bdims: &[BroadcastDimension],
                             buf: &'a [T]) -> Box<Iterator<Item=&'a [T]> + 'a> {
  Box::new(DenseBroadcastIter::new(bdims, buf))
}

impl<'a, T: 'a> DenseSpec<'a, T> {
  pub fn try_new(bshape: &[usize],
                 shape: &[usize],
                 buf: &'a [T]) -> Result<DenseSpec<'a, T>, Error> {
    if compatible(bshape, shape) {
      let bshape = Rc::new(target_shape(bshape, shape));
      let bdims = broadcast_shape(&bshape, shape);

      Ok(DenseSpec {
        bshape: bshape,
        bdims: bdims,
        buf: buf
      })
    } else { Err(Error::IncompatibleBroadcast) }
  }

  pub fn chop(&mut self, n: usize) -> &mut Self {
    let len = self.bshape.len() - n;

    Rc::get_mut(&mut self.bshape).unwrap().truncate(len);
    self.bdims.truncate(len);
    self
  }

  pub fn bshape(&self) -> &Rc<Vec<usize>> { &self.bshape }

  pub fn bshape_push(&mut self, dim: usize) -> &mut Self {
    Rc::get_mut(&mut self.bshape).unwrap().push(dim);
    self
  }

  /// Does the thing
  ///
  /// ```
  /// use mleap_tensor::core::broadcast;
  /// use std::rc::Rc;
  ///
  /// let shape1: Vec<usize> = vec![3, 2, 1];
  /// let shape2: Vec<usize> = vec![3, 2, 1];
  ///
  /// let buf: Vec<f32> = vec![12.3, 33.4, 44.1, 56.7, 34.2, 23.5];
  ///
  /// let spec = broadcast::DenseSpec::try_new(&shape1, &shape2, &buf).unwrap();
  /// let r: Vec<f32> = spec.iter().map(|a| a[0]).collect();
  ///
  /// assert_eq!(&r, &[12.3, 33.4, 44.1, 56.7, 34.2, 23.5]);
  /// ```
  pub fn iter(&self) -> Box<Iterator<Item=&'a [T]> + 'a> {
    Box::new(dense_iter(&self.bdims, self.buf))
  }
}

impl<'a, T: 'a> DenseSpec2<'a, T> {
  pub fn try_new(shape1: &[usize],
                 buf1: &'a [T],
                 shape2: &[usize],
                 buf2: &'a [T]) -> Result<DenseSpec2<'a, T>, Error> {
    if compatible(shape1, shape2) {
      let bshape = Rc::new(target_shape(shape1, shape2));
      let bdims1 = broadcast_shape(&bshape, shape1);
      let bdims2 = broadcast_shape(&bshape, shape2);

      Ok(DenseSpec2 {
        bshape: bshape,
        bdims1: bdims1,
        buf1: buf1,
        bdims2: bdims2,
        buf2: buf2
      })
    } else { Err(Error::IncompatibleBroadcast) }
  }

  pub fn chop(&mut self, n: usize) -> &mut Self {
    let len = self.bshape.len() - n;

    Rc::get_mut(&mut self.bshape).unwrap().truncate(len);
    self.bdims1.truncate(len);
    self.bdims2.truncate(len);

    self
  }

  pub fn bshape(&self) -> &Rc<Vec<usize>> { &self.bshape }

  pub fn bshape_push(&mut self, dim: usize) -> &mut Self {
    Rc::get_mut(&mut self.bshape).unwrap().push(dim);

    self
  }

  /// Does the thing
  ///
  /// ```
  /// use mleap_tensor::core::broadcast;
  /// use std::rc::Rc;
  ///
  /// let shape1: Vec<usize> = vec![3, 2, 1];
  /// let shape2: Vec<usize> = vec![3, 2, 1];
  ///
  /// let buf1: Vec<f32> = vec![12.3, 33.4, 44.1, 56.7, 34.2, 23.5];
  /// let buf2: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
  ///
  /// let spec = broadcast::DenseSpec2::try_new(&shape1, &buf1, &shape2, &buf2).unwrap();
  /// let mut i = spec.iter().map(|(a, b)| a[0] + b[0]);
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
  pub fn iter(&self) -> Box<Iterator<Item=(&'a [T], &'a [T])> + 'a> {
    let iter1 = dense_iter(&self.bdims1, self.buf1);
    let iter2 = dense_iter(&self.bdims2, self.buf2);

    Box::new(iter1.zip(iter2))
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


