//use iter::{TensorIterator, TensorIter, DenseBroadcastIter};
//use dim::BroadcastDimension;
//use broadcast;

use std::iter;
use std::cmp;
use std::slice::Chunks;
use std::result::Result;

use broadcast;
use iter::{DenseStrideIter, DenseBroadcastIter};

#[derive(Debug, Clone, Copy)]
pub enum Error {
  EmptyShape,
  IncompatibleShape
}

pub trait Spec<'a> {
  type Item;
  type I: Iterator<Item=Self::Item>;

  fn shape(&self) -> &[usize];

  fn iter(&'a self) -> Self::I;

  fn zip<S: Spec<'a>>(self, other: S) -> Zip<Self, S>
    where Self: Sized, S: Sized {
      self.try_zip(other).unwrap()
    }

  fn try_zip<S: Spec<'a>>(self, other: S) -> Result<Zip<Self, S>, Error>
    where Self: Sized, S: Sized {
      if self.shape() == other.shape() {
        Ok(Zip {
          shape: self.shape().to_vec(),
          spec1: self,
          spec2: other
        })
      } else { Err(Error::IncompatibleShape) }
    }

  fn map<B, F>(self, tshape: Vec<usize>, f: F) -> Map<Self, F>
    where Self: Sized, F: Fn(Self::Item) -> B {
      Map {
        shape: self.shape().to_vec(),
        tshape: tshape.clone(),
        spec: self,
        f: f
      }
    }
}

pub trait ShapedSpec<'a> : Spec<'a> {
  fn tshape(&self) -> &[usize];
}

pub struct Dense<'a, T: 'a> {
  shape: Vec<usize>,
  tshape: Vec<usize>,
  buf: &'a [T]
}

pub struct DenseBroadcast<'a, T: 'a> {
  bshape: Vec<usize>,
  shape: Vec<usize>,
  tshape: Vec<usize>,
  buf: &'a [T]
}

pub struct Zip<S1, S2> {
  shape: Vec<usize>,
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

  pub fn pop_to_tshape(&mut self) -> &mut Self { self.try_pop_to_tshape().unwrap() }

  pub fn try_pop_to_tshape(&mut self) -> Result<&mut Self, Error> {
    match self.shape.pop() {
      Some(d) => {
        self.tshape.push(d);
        Ok(self)
      },
      None => Err(Error::EmptyShape)
    }
  }
}

impl<'a, T: 'a> Spec<'a> for Dense<'a, T> {
  type Item = &'a [T];
  type I = Chunks<'a, T>;

  fn shape(&self) -> &[usize] { &self.shape }

  fn iter(&'a self) -> Chunks<'a, T> {
    self.buf.chunks(cmp::max(1, self.tshape.iter().product()))
  }
}

impl<'a, T: 'a> ShapedSpec<'a> for Dense<'a, T> {
  fn tshape(&self) -> &[usize] { &self.tshape }
}

impl<'a, T: 'a> DenseBroadcast<'a, T> {
  pub fn new(bshape: Vec<usize>,
             shape: Vec<usize>,
             buf: &'a [T]) -> DenseBroadcast<'a, T> {
    DenseBroadcast {
      bshape: bshape,
      shape: shape,
      tshape: Vec::new(),
      buf: buf
    }
  }

  pub fn pop_to_tshape(&mut self) -> &mut Self { self.try_pop_to_tshape().unwrap() }

  pub fn try_pop_to_tshape(&mut self) -> Result<&mut Self, Error> {
    match self.bshape.pop() {
      Some(d) => {
        self.tshape.push(d);
        Ok(self)
      },
      None => Err(Error::EmptyShape)
    }
  }
}

impl<'a, T: 'a> Spec<'a> for DenseBroadcast<'a, T> {
  type Item = &'a [T];
  type I = DenseBroadcastIter<'a, T>;

  fn shape(&self) -> &[usize] { &self.bshape }

  fn iter(&'a self) -> DenseBroadcastIter<'a, T> {
    let strides: Vec<usize> = DenseStrideIter::new(&self.shape).collect();
    let bdims = broadcast::broadcast_dims(&self.bshape, &self.shape, &strides);

    DenseBroadcastIter::new(bdims, self.buf)
  }
}

impl<'a, T: 'a> ShapedSpec<'a> for DenseBroadcast<'a, T> {
  fn tshape(&self) -> &[usize] { &self.tshape }
}

impl<'a, S1: Spec<'a>, S2: Spec<'a>> Spec<'a> for Zip<S1, S2> {
  type Item = (S1::Item, S2::Item);
  type I = iter::Zip<S1::I, S2::I>;

  fn shape(&self) -> &[usize] { &self.shape }

  fn iter(&'a self) -> iter::Zip<S1::I, S2::I> {
    self.spec1.iter().zip(self.spec2.iter())
  }
}

impl<S, F> Map<S, F> {
  pub fn tshape(&self) -> &[usize] { &self.tshape }
}

impl<'a, B, S: Spec<'a>, F> Spec<'a> for Map<S, F> where F: Fn(S::Item) -> B + 'a {
  type Item = B;
  type I = iter::Map<S::I, &'a F>;

  fn shape(&self) -> &[usize] { &self.shape }

  fn iter(&'a self) -> iter::Map<S::I, &'a F> {
    self.spec.iter().map(&self.f)
  }
}

impl<'a, B, S: Spec<'a>, F> ShapedSpec<'a> for Map<S, F> where F: Fn(S::Item) -> B + 'a {
  fn tshape(&self) -> &[usize] { &self.tshape }
}

