#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseBroadcastDimension {
  // Size of the slice
  pub size: usize,

  // Stride for each iteration
  // This will be 0 for broadcasted dimensions
  pub stride: usize,

  // Target number of iterations for this dimension
  pub target: usize
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseBroadcastShape {
  dimensions: Vec<BroadcastDimension>
}

#[derive(Debug)]
pub struct DenseBroadcast<'a, T: 'a> {
  bshape: DenseBroadcastShape,
  shape1: &'a [usize],
  buf1: &'a [T],
  shape2: &'a [usize]
  buf2: &'a [T]
}

