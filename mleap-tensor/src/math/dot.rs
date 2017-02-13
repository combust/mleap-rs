//use math::Error;
//use std::result::Result;
//use tensor::*;

//pub trait Dot<T> {
  //fn dot(&self, b: &Tensor<T>) -> Result<Tensor<T>, Error>;
//}

//impl Dot<f32> for Tensor<f32> {
  //fn dot(&self, bd: &Tensor<f32>) -> Result<Tensor<f32>, Error> {
    //match (self, bd) {
      //(&Tensor::Dense(ref a), &Tensor::Dense(ref b)) => Ok(Tensor::Dense(a.clone())),
      //_ => Err(Error::Broadcast)
    //}
  //}
//}

//fn dense_dot_f32(a: &DenseTensor<f32>, b: &DenseTensor<f32>) -> Result<DenseTensor<f32>, Error> {
  //if !a.dims().can_broadcast(b.dims()) { return Err(Error::Broadcast) }

  //Err(Error::Broadcast)
//}
