use std::cmp;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BroadcastDimension {
  // Size of the slice
  pub size: usize,

  // Stride for each iteration
  // This will be 0 for broadcasted dimensions
  // Or if all dimensions are sparse and it is not needed
  pub stride: usize,

  // Target number of iterations for this dimension
  pub target: usize
}

impl BroadcastDimension {
  pub fn shape_from_iters(shape: &mut Iterator<Item=&usize>,
                      bshape: &mut Iterator<Item=&usize>,
                      strides: &mut Iterator<Item=usize>) -> Vec<BroadcastDimension> {
    shape.zip(bshape).zip(strides).map(|((&a, &b), s)| {
      let stride = if a == 1 { 0 } else { s };
      let target = cmp::max(a, b);

      BroadcastDimension {
        stride: stride,
        size: s,
        target: target
      }
    }).collect()
  }
}
