use dim::BroadcastDimension;

pub struct DenseBroadcastIter<'a, T: 'a> {
  bufs: Vec<Option<&'a [T]>>,
  iterators: Vec<DenseIter<'a, T>>
}

pub struct DenseIter<'a, T: 'a> {
  dim: BroadcastDimension,
  index: usize,
  buf: &'a [T]
}

impl<'a, T: 'a> DenseIter<'a, T> {
  fn new(dim: BroadcastDimension,
         buf: &'a [T]) -> DenseIter<'a, T> {
    DenseIter {
      dim: dim,
      index: 0,
      buf: buf
    }
  }
}

impl<'a, T: 'a> Iterator for DenseIter<'a, T> {
  type Item = &'a [T];

  fn next(&mut self) -> Option<&'a [T]> {
    if self.index < self.dim.target {
      let start = self.index * self.dim.stride;
      self.index += 1;
      Some(&self.buf[start..(start + self.dim.size)])
    } else { None }
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    (self.dim.target, Some(self.dim.target))
  }
}

// TODO: OPTIMIZATION: coalesce matching sequences of density into a chunks iterator
// if target shape is [3, 5, 6, 1, 1, 4, 5, 7, 1] and actual is [3, 5, 6, 1, 1, 4, 1, 7, 1]
// then iterators can look like this [Chunks(3, 5, 6, 1, 1, 4), DenseIter(5), Chunks(7, 1)]
impl<'a, T: 'a> DenseBroadcastIter<'a, T> {
  pub fn new(bdims: &[BroadcastDimension],
             buf: &'a [T]) -> DenseBroadcastIter<'a, T> {
    let mut bufs: Vec<Option<&'a [T]>> = Vec::with_capacity(bdims.len());
    let mut iterators: Vec<DenseIter<'a, T>> = Vec::with_capacity(bdims.len());

    bdims.iter().fold(Some(buf), |opt_pb, &bdim| {
      opt_pb.and_then(|pb| {
        let mut iterator = DenseIter::new(bdim, pb);
        let buf = iterator.next();

        bufs.push(buf);
        iterators.push(iterator);

        buf
      })
    });

    DenseBroadcastIter {
      bufs: bufs,
      iterators: iterators
    }
  }

  fn wind(&mut self) {
    let mut n: usize = 0;
    let mut bufr: Option<&'a [T]>;

    {
      let mut ii = self.iterators.iter_mut().zip(self.bufs.iter_mut()).rev();

      loop {
        match ii.next() {
          Some((ref mut i, ref mut b)) => {
            let next = i.next();
            **b = next;
            bufr = next;

            if b.is_some() { break }
            else {
              n += 1
            }
          },
          None => return
        }
      }
    }

    let start = self.iterators.len() - n;
    self.iterators[start..].iter_mut().zip(self.bufs[start..].iter_mut()).fold(bufr, |opt_pb, (i, b)| {
      opt_pb.and_then(|pb| {
        *i = DenseIter::new(i.dim, pb);
        let next = i.next();
        *b = next;
        next
      })
    });
  }
}

impl<'a, T: 'a> Iterator for DenseBroadcastIter<'a, T> {
  type Item = &'a [T];

  fn next(&mut self) -> Option<&'a [T]> {
    let next = self.bufs[self.bufs.len() - 1];
    self.wind();

    next
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    let (min, max) = self.iterators.iter().map(|i| i.size_hint()).fold((1, 1), |(min, max), (hmin, hmax)| {
      (min * hmin, max * hmax.unwrap_or_else(|| 1))
    });

    (min, Some(max))
  }
}
