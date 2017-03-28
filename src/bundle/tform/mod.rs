pub mod linear_regression;
pub mod string_indexer;
pub mod vector_assembler;
pub mod one_hot_encoder;
pub mod pipeline;

use std::any::{Any, TypeId};

use bundle::ser::OpNode;
use bundle::frame;
use bundle::dsl;

pub trait DefaultNode: OpNode + frame::Transformer {
  fn name(&self) -> &str;
  fn model(&self) -> &Any;

  fn create_shape(&self) -> dsl::Shape;
  fn create_node(&self) -> dsl::Node {
    dsl::Node::new(self.name().to_string(), self.create_shape())
  }
}

impl OpNode for Box<DefaultNode> {
  fn type_id(&self) -> TypeId { DefaultNode::type_id(self.as_ref()) }
}
