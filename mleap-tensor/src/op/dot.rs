use std::cmp;
use blas_sys::c::cblas_sdot;

pub trait Dot<'a> {
  type B;
  type C;

  fn dot(self, b: Self::B, a_stride: usize, b_stride: usize) -> Self::C where Self: Sized;
}

impl<'a> Dot<'a> for &'a [f32] {
  type B = &'a [f32];
  type C = f32;

  fn dot(self, b: &'a [f32], a_stride: usize, b_stride: usize) -> f32 {
    unsafe {
      cblas_sdot(cmp::max(self.len(), b.len()) as i32, self.as_ptr(), a_stride as i32, b.as_ptr(), b_stride as i32)
    }
  }
}
