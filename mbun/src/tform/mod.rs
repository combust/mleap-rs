pub mod linear_regression;

use std::any::{Any, TypeId};
use ser::OpNode;
use dsl;

pub trait Transformer: OpNode {
  fn name(&self) -> &str;
  fn model(&self) -> &Any;

  fn create_shape(&self) -> dsl::Shape;
  fn create_node(&self) -> dsl::Node {
    dsl::Node::new(self.name().to_string(), self.create_shape())
  }
}

impl OpNode for Box<Transformer> {
  fn type_id(&self) -> TypeId { Transformer::type_id(self.as_ref()) }
}
