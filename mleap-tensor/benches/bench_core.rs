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
  use mleap_tensor::core::broadcast;

  #[bench]
  fn bench_dot_big(b: &mut Bencher) {
    let shape1: Vec<usize> = vec![100, 100, 100];
    let shape2: Vec<usize> = vec![100, 1, 100];

    let buf1: Vec<f32> = rand_vec_f32(100 * 100 * 100);
    let buf2: Vec<f32> = rand_vec_f32(100 * 1 * 100);

    let mut br = broadcast::dense_broadcast(&shape1, &buf1, &shape2, &buf2).unwrap();
    br.chop(1);
    br.bshape_push(1);
    assert_eq!(br.iter().size_hint(), (100 * 100, Some(100 * 100)));

    b.iter(|| br.iter().map(|(a, b)| dot_f32(a, b)).collect::<Vec<f32>>());
  }

  #[bench]
  fn bench_dot_base(b: &mut Bencher) {
    let buf1: Vec<f32> = rand_vec_f32(100 * 100 * 100);
    let buf2: Vec<f32> = rand_vec_f32(100 * 100 * 100);

    b.iter(|| buf1.chunks(100).zip(buf2.chunks(100)).map(|(a, b)| dot_f32(a, b)).collect::<Vec<f32>>());
  }
}
