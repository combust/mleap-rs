use std::cmp;
use blas_sys::c::cblas_sdot;

pub trait Dot<'a> {
  type B;
  type C;

  fn dot(self, b: Self::B) -> Self::C where Self: Sized;
}

impl<'a> Dot<'a> for &'a [f32] {
  type B = &'a [f32];
  type C = f32;

  fn dot(self, b: &'a [f32]) -> f32 {
    unsafe {
      cblas_sdot(cmp::max(self.len(), b.len()) as i32, self.as_ptr(), 1, b.as_ptr(), 1)
    }
  }
}
