//use iter::{TensorIterator, TensorIter, DenseBroadcastIter};
//use dim::BroadcastDimension;
//use broadcast;

use std::iter;
use std::slice::Chunks;

pub trait Spec<'a> {
  type Item;
  type I: Iterator<Item=Self::Item>;

  fn shape(&self) -> &[usize];
  fn tshape(&self) -> &[usize];

  fn iter(&'a self) -> Self::I;

  fn zip<S: Spec<'a>>(self, other: S) -> Zip<Self, S> where Self: Sized, S: Sized {
    Zip {
      shape: self.shape().to_vec(),
      tshape: self.tshape().to_vec(),
      spec1: self,
      spec2: other
    }
  }

  fn map<B, F>(self, f: F) -> Map<Self, F>
    where Self: Sized, F: Fn(Self::Item) -> B {
      Map {
        shape: self.shape().to_vec(),
        tshape: self.tshape().to_vec(),
        spec: self,
        f: f
      }
    }
}

pub struct Dense<'a, T: 'a> {
  shape: Vec<usize>,
  tshape: Vec<usize>,
  buf: &'a [T]
}

pub struct Zip<S1, S2> {
  shape: Vec<usize>,
  tshape: Vec<usize>,
  spec1: S1,
  spec2: S2
}

pub struct Map<S, F> {
  shape: Vec<usize>,
  tshape: Vec<usize>,
  spec: S,
  f: F
}

impl<'a, T: 'a> Dense<'a, T> {
  pub fn new(shape: Vec<usize>,
             buf: &'a [T]) -> Dense<'a, T> {
    Dense {
      shape: shape,
      tshape: Vec::new(),
      buf: buf
    }
  }

  pub fn chop(&mut self) -> &mut Self {
    let i = self.shape.len() - self.tshape.len() - 1;
    self.tshape.push(self.shape[i]);
    self
  }
}

impl<'a, T: 'a> Spec<'a> for Dense<'a, T> {
  type Item = &'a [T];
  type I = Chunks<'a, T>;

  fn shape(&self) -> &[usize] { &self.shape }
  fn tshape(&self) -> &[usize] { &self.tshape }

  fn iter(&'a self) -> Chunks<'a, T> {
    self.buf.chunks(self.tshape.iter().product())
  }
}

impl<'a, S1: Spec<'a>, S2: Spec<'a>> Spec<'a> for Zip<S1, S2> {
  type Item = (S1::Item, S2::Item);
  type I = iter::Zip<S1::I, S2::I>;

  fn shape(&self) -> &[usize] { &self.shape }
  fn tshape(&self) -> &[usize] { &self.tshape }

  fn iter(&'a self) -> iter::Zip<S1::I, S2::I> {
    self.spec1.iter().zip(self.spec2.iter())
  }
}

impl<'a, B, S: Spec<'a>, F> Spec<'a> for Map<S, F> where F: Fn(S::Item) -> B + 'a {
  type Item = B;
  type I = iter::Map<S::I, &'a F>;

  fn shape(&self) -> &[usize] { &self.shape }
  fn tshape(&self) -> &[usize] { &self.tshape }

  fn iter(&'a self) -> iter::Map<S::I, &'a F> {
    self.spec.iter().map(&self.f)
  }
}

pub trait Test { }

impl Test {
  fn dot<'a>(shape: &[usize],
             buf1: &'a [f32],
             buf2: &'a [f32]) -> (Vec<usize>, Vec<f32>) {
    let mut spec1 = Dense::new(shape.to_vec(), buf1);
    let mut spec2 = Dense::new(shape.to_vec(), buf2);

    spec1.chop();
    spec2.chop();

    let zip = spec1.zip(spec2).map(|(a, b)| 23.0);
    let i = zip.iter();

    (Vec::new(), Vec::new())
  }
}

