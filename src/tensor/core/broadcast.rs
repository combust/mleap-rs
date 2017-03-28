use std::iter::repeat;
use std::cmp;
use std::cmp::Ordering;

use dim::BroadcastDimension;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
  IncompatibleBroadcast
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

pub fn broadcast_dims<'a>(bshape: &'a [usize],
                          shape: &'a [usize],
                          strides: &'a [usize]) -> Vec<BroadcastDimension> {
  BroadcastDimension::shape_from_iters(&mut shape.iter(), &mut bshape.iter(), &mut strides.iter())
}

pub fn target_shape<'a>(shape1: &'a [usize],
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

