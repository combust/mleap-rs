use std::iter::Iterator;

pub struct DenseBroadcastDim {
  // target dimensionality
  target: usize,

  // actual dimensionality
  // either 1 or equal to target
  actual: usize,

  // dense stride for this dimension
  stride: usize
}

impl DenseBroadcastDim {
  /// Wehther or not this dimension is expanded
  fn is_expanded(&self) -> bool { self.target != self.actual }
}

pub struct DenseDimIter<'a, 'b, T> {
  values: &'a [
  dim: &'a DenseBroadcastDim,
  index: usize
}

impl<'a> DenseDimIter<'a> {
  fn new(dim: &'a DenseBroadcastDim) {
    DenseDimIter {
      dim: Dim,
      index: 0
    }
  }

  fn offset(&self) -> usize {
    if self.dim.is_expanded() {
      self.dim.stride
    } else {
      self.index * self.dim.stride
    }
  }
}

//impl<'a> Iterator for DenseDimIter<'a> {
  //type Item = &
//}

//pub struct DenseBroadcastIter<'a, T> {
  //values: &'a [T],
  //dims: Vec<DenseBroadcastDim>,
  //state: Vec<DenseBroadcastDimIter>
//}
