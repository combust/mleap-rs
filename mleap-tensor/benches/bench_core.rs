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
  fn bench_dot_mleap(b: &mut Bencher) {
    let shape1: Vec<usize> = vec![100, 100, 1000];
    let shape2: Vec<usize> = vec![100, 1, 1000];

    let buf1: Vec<f32> = rand_vec_f32(100 * 100 * 1000);
    let buf2: Vec<f32> = rand_vec_f32(100 * 1 * 1000);

    let mut br = broadcast::dense_broadcast(&shape1, &buf1, &shape2, &buf2).unwrap();
    br.chop(1);
    br.bshape_push(1);
    assert_eq!(br.iter().size_hint(), (100 * 100, Some(100 * 100)));

    b.iter(|| {
      let out = br.iter().map(|(a, b)| dot_f32(a, b)).collect::<Vec<f32>>();
      assert_eq!(out[0], dot_f32(&buf1[0..1000], &buf2[0..1000]));
      assert_eq!(out[1], dot_f32(&buf1[1000..2000], &buf2[0..1000]));
      assert_eq!(out.len(), 100 * 100)
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
