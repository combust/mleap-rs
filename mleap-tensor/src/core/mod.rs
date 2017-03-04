pub mod broadcast;
pub mod iter;
pub mod dim;

pub trait Op<T> {
  fn into_dense(&self) -> (Vec<usize>, Vec<T>);
  fn into_sparse(&self) -> (Vec<usize>, Vec<Vec<usize>>, Vec<T>);
}

pub struct DenseDotOp<'a, T: 'a> {
  b: &'a broadcast::DenseBroadcast<'a, T>
}

impl<'a> Op<f32> for DenseDotOp<'a, f32> {
  fn into_dense(&self) -> (Vec<usize>, Vec<f32>) {
    let mut b = self.b.clone();
    b.chop(1).bshape_push(1);

    let buf: Vec<f32> = b.iter().map(|_| {
      23.0
    }).collect();

    (b.bshape.clone(), buf)
  }

  fn into_sparse(&self) -> (Vec<usize>, Vec<Vec<usize>>, Vec<f32>) {
    panic!("unimplemented")
  }
}
