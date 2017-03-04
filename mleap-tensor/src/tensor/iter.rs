//use broadcast;
//use std::ops::Range;
//use std::iter::{Iterator, repeat};

//pub enum BroadcastIter<'a, T: 'a> {
  //Dense(Box<Iterator<Item=&'a [T]> + 'a>),
  //Sparse(Box<Iterator<Item=(&'a [usize], &'a [T])> + 'a>)
//}

//pub struct DenseIter<'a, T: 'a> {
  //pub stride: usize,
  //pub size: usize,
  //pub range: Range<usize>,
  //pub buffer: &'a [T]
//}

//pub struct DenseBroadcastIter<'a, T: 'a> {
  //pub dimensions: Vec<broadcast::Dimension>,
  //pub buffer: &'a [T],
  //pub buffers: Vec<Option<&'a [T]>>,
  //pub iterators: Option<Vec<DenseIter<'a, T>>>
//}

//impl<'a, T: 'a> DenseBroadcastIter<'a, T> {
  //pub fn new(buffer: &'a [T], dimensions: Vec<broadcast::Dimension>) -> DenseBroadcastIter<'a, T> {
    //let buffers = repeat(None).take(dimensions.len()).collect();

    //DenseBroadcastIter {
      //dimensions: dimensions,
      //buffer: buffer,
      //buffers: buffers,
      //iterators: None
    //}
  //}

  //fn next_backtrack(iter: &mut Iterator<Item=(&mut DenseIter<'a, T>, &mut Option<&'a [T]>)>) -> Option<&'a [T]> {
    //match iter.next() {
      //Some((ref mut i, ref mut b)) => {
        //match i.next() {
          //Some(ref nb) => {
            //**b = Some(*nb);
            //**b
          //},

          //None => {
            //match DenseBroadcastIter::next_backtrack(iter) {
              //Some(ref prev_b) => {
                //i.reset_with_buffer(prev_b);
                //**b = i.next();
                //**b
              //},
              //None => None
            //}
          //}
        //}
      //},

      //None => None
    //}
  //}

  //fn next_first(&mut self) -> Option<&'a [T]> {
    //let mut is: Vec<DenseIter<'a, T>> = Vec::with_capacity(self.dimensions.len());
    //let dim_iter = self.dimensions.iter();
    //let buf_iter = self.buffers.iter_mut();

    //let r = dim_iter.zip(buf_iter).fold(Some(self.buffer), |b_opt, (dim, buf)| {
      //b_opt.and_then(|b| {
        //let mut i = DenseIter::with_dimension(b, dim);
        //let next = i.next();
        //is.push(i);

        //match next {
          //Some(ref nb) => *buf = Some(*nb),
          //None => () // do nothing
        //};

        //next
      //})
    //});
    //self.iterators = Some(is);

    //r
  //}
//}

//impl<'a, T: 'a> DenseIter<'a, T> {
  //fn with_dimension(buffer: &'a [T], dim: &broadcast::Dimension) -> DenseIter<'a, T> {
    //DenseIter {
      //stride: dim.stride,
      //size: dim.size,
      //range: 0..dim.target,
      //buffer: buffer
    //}
  //}

  //fn reset_with_buffer(&mut self, buffer: &'a [T]) {
    //self.buffer = buffer;
    //self.range = 0..self.range.end;
  //}
//}

//impl<'a, T: 'a> Iterator for DenseIter<'a, T> {
  //type Item = &'a [T];

  //fn next(&mut self) -> Option<&'a [T]> {
    //let start = self.range.start * self.stride;
    //self.range.next().map(|_| &self.buffer[start..(start + self.size)])
  //}
//}

//impl<'a, T: 'a> Iterator for DenseBroadcastIter<'a, T> {
  //type Item = &'a [T];

  //fn next(&mut self) -> Option<&'a [T]> {
    //match self.iterators {
      //Some(ref mut is) => {
        //let mut ii = is.iter_mut().zip(self.buffers.iter_mut()).rev();
        //DenseBroadcastIter::next_backtrack(&mut ii)
      //},
      //None => self.next_first()
    //}
  //}
//}
