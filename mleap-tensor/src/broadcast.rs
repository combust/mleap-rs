#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dimension {
  // Size of the slice
  pub size: usize,

  // Stride for each iteration
  // This will be 0 for broadcasted dimensions
  pub stride: usize,

  // Target number of iterations for this dimension
  pub target: usize
}
