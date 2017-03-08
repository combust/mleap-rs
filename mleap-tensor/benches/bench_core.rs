#![feature(test)]

extern crate test;
extern crate mleap_tensor;
extern crate rand;
extern crate blas_sys;

use rand::Rng;
use blas_sys::c::cblas_sdot;

pub fn add_two(a: i32) -> i32 {
  a + 2
}

pub fn rand_vec_f32(n: usize) -> Vec<f32> {
  let mut rng = rand::thread_rng();

  (0..n).map(|_| rng.gen::<f32>()).collect()
}

pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
  unsafe {
    cblas_sdot(a.len() as i32, a.as_ptr(), 1, b.as_ptr(), 1)
  }
}

#[cfg(test)]
mod bench_core {
  use super::*;
  use test::Bencher;
  use mleap_tensor::core::spec::*;
  use mleap_tensor::core::broadcast;
  use mleap_tensor::op::dot::*;
  use mleap_tensor::core::build::*;

  #[bench]
  fn bench_dot_mleap_broadcast(b: &mut Bencher) {
    let shape1: Vec<usize> = vec![100, 100, 1000];
    let shape2: Vec<usize> = vec![100, 1, 1000];
    let bshape: Vec<usize> = broadcast::target_shape(&shape1, &shape2);

    let buf1: Vec<f32> = rand_vec_f32(100 * 100 * 1000);
    let buf2: Vec<f32> = rand_vec_f32(100 * 1 * 1000);

    let mut spec1 = DenseBroadcast::new(bshape.clone(), shape1, &buf1);
    let mut spec2 = DenseBroadcast::new(bshape.clone(), shape2, &buf2);

    spec1.pop_to_tshape();
    spec2.pop_to_tshape();

    let dot = spec1.zip(spec2).map(Vec::new(), |(a, b)| a.dot(b));

    b.iter(|| {
      dot.build_dense_scalar();
    });
  }

  #[bench]
  fn bench_dot_mleap(b: &mut Bencher) {
    let shape1: Vec<usize> = vec![100, 100, 1000];
    let shape2: Vec<usize> = vec![100, 100, 1000];

    let buf1: Vec<f32> = rand_vec_f32(100 * 100 * 1000);
    let buf2: Vec<f32> = rand_vec_f32(100 * 100 * 1000);

    let mut spec1 = Dense::new(shape1, &buf1);
    let mut spec2 = Dense::new(shape2, &buf2);

    spec1.pop_to_tshape();
    spec2.pop_to_tshape();

    let dot = spec1.zip(spec2).map(Vec::new(), |(a, b)| a.dot(b));

    b.iter(|| {
      dot.build_dense_scalar();
    });
  }

  #[bench]
  fn bench_dot_base(b: &mut Bencher) {
    let buf1: Vec<f32> = rand_vec_f32(100 * 100 * 1000);
    let buf2: Vec<f32> = rand_vec_f32(100 * 100 * 1000);

    b.iter(|| buf1.chunks(1000).zip(buf2.chunks(1000)).map(|(a, b)| dot_f32(a, b)).collect::<Vec<f32>>());
  }

  #[bench]
  fn bench_dot_oneshot(b: &mut Bencher) {
    let buf1: Vec<f32> = rand_vec_f32(100 * 100 * 1000);
    let buf2: Vec<f32> = rand_vec_f32(100 * 100 * 1000);

    b.iter(|| dot_f32(&buf1, &buf2));
  }
}
