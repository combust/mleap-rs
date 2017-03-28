use std::any::*;
use std::collections::HashMap;

use bundle::tform::{self, DefaultNode};
use bundle::ser::*;
use bundle::frame;
use bundle::dsl;

pub const OP: &StringIndexerOp = &StringIndexerOp { };

#[derive(Clone)]
pub struct StringIndexerModel {
  labels: Vec<String>,
  label_to_index: HashMap<String, usize>
}

pub struct StringIndexer {
  name: String,
  input_col: String,
  output_col: String,
  model: StringIndexerModel
}

pub struct StringIndexerOp { }

impl StringIndexerModel {
  pub fn new(labels: Vec<String>) -> StringIndexerModel {
    let label_to_index = labels.iter().
      enumerate().
      map(|(i, s)| (s.clone(), i)).
      collect();

    StringIndexerModel {
      labels: labels,
      label_to_index: label_to_index
    }
  }

  pub fn try_encode(&self, label: &str) -> frame::Result<usize> {
    self.label_to_index.get(label).map(|x| Ok(*x)).unwrap_or_else(|| {
      Err(frame::Error::TransformError(String::from(format!("Invalid label: {}", label))))
    })
  }
}
impl OpNode for StringIndexer { }

impl frame::Transformer for StringIndexer {
  fn transform(&self, frame: &mut frame::LeapFrame) -> frame::Result<()> {
    frame.try_strings(&self.input_col).and_then(|labels| {
      let mut indices: Vec<i32> = Vec::with_capacity(labels.len());

      for label in labels.iter() {
        match self.model.try_encode(label) {
          Ok(r) => indices.push(r as i32),
          Err(err) => return Err(err)
        }
      }

      Ok(indices)
    }).and_then(|indices| {
      frame.try_with_ints(self.output_col.clone(), indices).map(|_| ())
    })
  }
}

impl DefaultNode for StringIndexer {
  fn name(&self) -> &str { &self.name }
  fn model(&self) -> &Any { &self.model as &Any }

  fn create_shape(&self) -> dsl::Shape {
    dsl::Shape::with_standard_io(String::from(self.input_col.as_ref()), String::from(self.output_col.as_ref()))
  }
}

impl Op for StringIndexerOp {
  type Node = Box<tform::DefaultNode>;

  fn type_id(&self) -> TypeId { TypeId::of::<StringIndexer>() }
  fn op(&self) -> &'static str { "string_indexer" }

  fn name<'a>(&self, node: &'a Self::Node) -> &'a str { node.name() }

  fn model<'a>(&self, node: &'a Self::Node) -> &'a Any { DefaultNode::model(node.as_ref()) }

  fn store_model(&self,
                 obj: &Any,
                 model: &mut dsl::Model,
                 _ctx: &Context<Self::Node>) -> Result<()> {
    obj.downcast_ref::<StringIndexerModel>().map(|si| {
      model.with_attr("labels", dsl::Attribute::Array(dsl::VectorValue::String(si.labels.clone())));
      Ok(())
    }).unwrap_or_else(|| Err(Error::InvalidOp("Expected a StringIndexerModel".to_string())))
  }

  fn load_model(&self,
                model: &dsl::Model,
                _ctx: &Context<Self::Node>) -> Result<Box<Any>> {
    model.get_string_vector("labels").and_then(|labels| {
      Some(StringIndexerModel::new(labels.to_vec()))
    }).map(|x| Ok(Box::new(x) as Box<Any>)).unwrap_or_else(|| Err(Error::InvalidModel("".to_string())))
  }

  fn node(&self, node: &Self::Node, _ctx: &Context<Self::Node>) -> dsl::Node {
    node.create_node()
  }

  fn load(&self,
          node: &dsl::Node,
          model: Box<Any>,
          _ctx: &Context<Self::Node>) -> Result<Self::Node> {
    model.downcast_ref::<StringIndexerModel>().and_then(|si| {
      node.shape().get_standard_io().map(|(i, o)| {
        StringIndexer {
          name: node.name().to_string(),
          input_col: i.name().to_string(),
          output_col: o.name().to_string(),
          model: si.clone()
        }
      })
    }).map(|x| Ok(Box::new(x) as Box<DefaultNode>)).
    unwrap_or_else(|| Err(Error::DowncastError(String::from(""))))
  }
}
