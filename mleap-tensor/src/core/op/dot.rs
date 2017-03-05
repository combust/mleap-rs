//use broadcast;

//use std::result::Result;
//use iter::{TensorIterator, TensorIter};
//use blas_sys::c::{cblas_sdot, cblas_ddot};

//use broadcast;

//pub trait Dot<T> { }

//impl Dot<f32> {
  ///// Try creating a dense dot product from two dense vectors.
  /////
  ///// ```
  ///// use mleap_tensor::core::op::Dot;
  /////
  ///// let shape1: Vec<usize> = vec![1, 2, 3];
  ///// let shape2: Vec<usize> = vec![1, 2, 3];
  /////
  ///// let buf1: Vec<f32> = vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
  ///// let buf2: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
  /////
  ///// let iter = Dot::<f32>::try_from_dense(&shape1, &buf1, &shape2, &buf2).unwrap();
  ///// let dotp: Vec<f32> = iter.collect();
  /////
  ///// assert_eq!(&dotp, &[6.0, 30.0]);
  ///// ```
  //pub fn try_from_dense<'a>(shape1: &[usize],
                            //buf1: &'a [f32],
                            //shape2: &[usize],
                            //buf2: &'a [f32]) -> Result<Box<TensorIterator<Item=f32> + 'a>, broadcast::Error> {
    //broadcast::DenseSpec2::try_new(shape1, buf1, shape2, buf2).map(|mut spec| {
      //spec.chop(1).bshape_push(1);
      //let iter = TensorIter::new(spec.bshape(), spec.iter().map(|(a, b)| Dot::<f32>::dot_impl(a, b)));

      //Box::new(iter) as Box<TensorIterator<Item=f32> + 'a>
    //})
  //}

  //fn dot_impl(buf1: &[f32], buf2: &[f32]) -> f32 {
    //unsafe {
      //cblas_sdot(buf1.len() as i32, buf1.as_ptr(), 1, buf2.as_ptr(), 1)
    //}
  //}
//}

//impl Dot<f64> {
  ///// Try creating a dense dot product from two dense vectors.
  /////
  ///// ```
  ///// use mleap_tensor::core::op::Dot;
  /////
  ///// let shape1: Vec<usize> = vec![1, 2, 3];
  ///// let shape2: Vec<usize> = vec![1, 2, 3];
  /////
  ///// let buf1: Vec<f64> = vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
  ///// let buf2: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
  /////
  ///// let iter = Dot::<f64>::try_from_dense(&shape1, &buf1, &shape2, &buf2).unwrap();
  ///// let dotp: Vec<f64> = iter.collect();
  /////
  ///// assert_eq!(&dotp, &[6.0, 30.0]);
  ///// ```
  //pub fn try_from_dense<'a>(shape1: &[usize],
                            //buf1: &'a [f64],
                            //shape2: &[usize],
                            //buf2: &'a [f64]) -> Result<Box<TensorIterator<Item=f64> + 'a>, broadcast::Error> {
    //broadcast::DenseSpec2::try_new(shape1, buf1, shape2, buf2).map(|mut spec| {
      //spec.chop(1).bshape_push(1);
      //let iter = TensorIter::new(spec.bshape(), spec.iter().map(|(a, b)| Dot::<f64>::dot_impl(a, b)));

      //Box::new(iter) as Box<TensorIterator<Item=f64> + 'a>
    //})
  //}

  //fn dot_impl(buf1: &[f64], buf2: &[f64]) -> f64 {
    //unsafe {
      //cblas_ddot(buf1.len() as i32, buf1.as_ptr(), 1, buf2.as_ptr(), 1)
    //}
  //}
//}

