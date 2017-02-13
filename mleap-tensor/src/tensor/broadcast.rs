pub struct IterPadded<'a, 'b, Item> {
  slice: &'a [Item],
  pad: &'b Item,
  size: usize,
  index: usize,
  pad_index: usize
}

impl<'a, 'b, Item> IterPadded<'a, 'b, Item> {
  pub fn new(slice: &'a [Item], pad: &Item, size: usize) -> IterPadded<'a, 'b, Item> {
    IterPadded {
      slice: slice,
      pad: pad,
      size: size,
      index: 0,
      pad_index: size - slice.len()
    }
  }
}

impl<'a, 'b, I> Iterator for IterPadded<'a, 'b, I> {
  type Item = &I;

  fn next(&mut self) -> &I {

  }
}

//pub struct RepeatableIterator<Item, I: Iterator<Item=Item>, R: Iterator<Item=I>> {
  //iters: R,
  //cur: Option<I>
//}

//impl<Item, I: Iterator<Item=Item>, R: Iterator<Item=I>> RepeatableIterator<Item, I, R> {
  ///// Construct a new repeatable iterator
  /////
  ///// ```
  ///// use mleap_tensor::tensor::RepeatableIterator;
  /////
  ///// let v = vec![0, 23, 44];
  ///// let iters = (0..2).map(|_| v.iter());
  ///// let r = RepeatableIterator::new(iters);
  ///// let s = r.map(|x| x.to_string()).collect::<Vec<_>>().join(" ");
  ///// assert_eq!(s, "0 23 44 0 23 44");
  ///// ```
  //pub fn new(iters: R) -> RepeatableIterator<Item, I, R> {
    //RepeatableIterator {
      //iters: iters,
      //cur: None
    //}
  //}
//}

//impl<Item, I: Iterator<Item=Item>, R: Iterator<Item=I>> Iterator for RepeatableIterator<Item, I, R> {
  //type Item = Item;

  //fn next(&mut self) -> Option<Item> {
    //let next = match self.cur {
      //Some(ref mut i) => i.next(),
      //None => None
    //};

    //if next.is_none() {
      //self.cur = self.iters.next();
      //match self.cur {
        //Some(ref mut i) => i.next(),
        //None => None
      //}
    //} else {
      //next
    //}
  //}
//}
