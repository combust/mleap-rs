use tensor::Dimensions;
use std::iter::Iterator;

pub struct DimInfo {
  // this is the target dimensionality
  target: usize,

  // this is the actual dimensionality
  // actual will either be 1 or equal to target
  actual: usize,

  // current stride multiplier
  // if actual = 1, then stride_index will always = 0
  // otherwise, stride_index will range from [0..actual)
  stride_index: usize,

  // the actual stride for this dimension
  dim_stride: usize,

  // the current index in iteration for
  // this dimension
  // this value will range from [0..target)
  index: usize
}

impl DimInfo {
  fn inc(&mut self) -> bool {
    let next = self.index + 1;

    if next >= self.target {
      // overflow, reset dimension to 0
      self.index = 0;
      self.stride_index = 0;
      true
    } else {
      // no overflow, increase this dimension
      self.index = next;
      if self.actual > 1 {
        // increment stride index if
        // this dimension has actual dimensionality > 1
        self.stride_index = next;
      }
      false
    }
  }
}

impl DimInfo {
  fn offset(&self) -> usize { self.dim_stride * self.stride_index }
}

enum BroadcastIterState {
  Fresh,
  Iterating,
  Finished
}

pub struct BroadcastIterN<'a, T> where T: 'a {
  values: &'a Vec<T>,

  // contains current state of the iteration
  //
  // Fresh - iteration has not yet started
  // Iterating - in the middle of iterating
  // Finished - done iterating
  state: BroadcastIterState,

  // info and state of each dimension in this
  // iteration
  dims: Vec<DimInfo>,

  // size of each resulting slice
  size: usize
}

impl<'a, T> BroadcastIterN<'a, T> {
  pub fn new(actual: &Dimensions, other: &Dimensions, values: &'a Vec<T>) -> BroadcastIterN<'a, T> {
    // determine last dimension, this is the size
    let size = match actual.last() {
      Some(last) => *last, // nd-tensor
      None => 1 // scalar
    };

    BroadcastIterN {
      values: values,
      state: BroadcastIterState::Fresh,
      dims: vec![],
      size: 0
    }
  }

  fn fresh_next(&mut self) -> Option<&'a [T]> {
    if self.values.len() > 0 {
      self.state = BroadcastIterState::Iterating;
      Some(&self.values[0..self.size])
    } else {
      self.state = BroadcastIterState::Finished;
      None
    }
  }

  fn iterating_next(&mut self) -> Option<&'a [T]> {
    {
      let mut iter = self.dims.iter_mut().rev();

      while {
        match iter.next() {
          Some(ref mut dim) =>
            // increment the index of this dimension
            // returns true if it overflows, causing the
            // preceding dimension to increment
            //
            // returns false if dimension does not overflow,
            // causing iteration to stop
            dim.inc(),
          None => {
            // we have called inc on every dimension
            // and every dimension overflowed
            self.state = BroadcastIterState::Finished;
            return None
          }
        }
      } { }
    }

    // if we made it here, there must ve a vector slice to return
    let offset = self.offset();
    Some(&self.values[offset..(offset + self.size)])
  }

  fn offset(&self) -> usize {
    self.dims.iter().fold(0, |offset, dim| offset + dim.offset())
  }
}

impl<'a, T> Iterator for BroadcastIterN<'a, T> {
  type Item = &'a [T];

  fn next(&mut self) -> Option<&'a [T]> {
    match self.state {
      BroadcastIterState::Fresh => self.fresh_next(),
      BroadcastIterState::Iterating => self.iterating_next(),
      BroadcastIterState::Finished => None
    }
  }
}
