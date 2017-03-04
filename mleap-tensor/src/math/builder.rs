//use tensor::{Tensor, Dense, Sparse, TensorShape};
//use broadcast;

//pub struct ScalarTensorBuilderOp<'a, T: 'a> {
  //drop: usize,
  //dense_f: &'a Fn(&'a [T], &'a [T]) -> T,
//}

//pub struct TensorBuilderBase<'a, T: 'a> {
  //bshape: TensorShape,
  //bdims: Vec<broadcast::Dimension>,
  //a: &'a Tensor<T>,
  //b: &'a Tensor<T>
//}

//pub struct ScalarTensorBuilder<'a, T: 'a> {
  //base: TensorBuilderBase<'a, T>,
  //op: &'a ScalarTensorBuilderOp<'a, T>
//}

//impl<'a, T: 'a> ScalarTensorBuilder<'a, T> {
  //pub fn to_dense(&self) -> Dense<T> {
    //panic!("not implemented")
  //}

  //pub fn to_sparse(&self) -> Sparse<T> {
    //panic!("not implemented")
  //}
//}
