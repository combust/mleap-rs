pub trait Op { }

pub trait DenseOp { }

pub struct DenseDot<'a, T: 'a> {
  bshape: &'a [usize],
  a: Box<Iterator<Item=&'a [T]> + 'a>,
  b: Box<Iterator<Item=&'a [T]> + 'a>
}
