use std::any::*;
use std::ptr;

use bundle::tform::{self, DefaultNode};
use bundle::ser::*;
use bundle::frame;
use bundle::dsl;

use libc;

pub type Transform = extern fn(*mut frame::LeapFrame);

extern "C" fn dummy(frame: *mut frame::LeapFrame) {
    panic!("dummy transform has been called");
}

pub static mut OP: ExternalOp = ExternalOp {
    transform: dummy
};

pub struct ExternalModel {
    data: *const libc::c_void
}

pub struct External {
  ext_transform: Transform,
  name: String,
  features_col: String,
  prediction_col: String,
  model: ExternalModel
}

pub struct ExternalOp {
    pub transform: Transform
}

impl OpNode for External {
  fn op(&self) -> &'static str { "external" }
}

impl frame::Transformer for External {
  fn transform(&self, frame: &mut frame::LeapFrame) -> frame::Result<()> {
    (self.ext_transform)(frame as *mut frame::LeapFrame);
    Ok(())
  }
}

impl DefaultNode for External {
  fn name(&self) -> &str { &self.name }
  fn model(&self) -> &Any { &self.model as &Any }

  fn create_shape(&self) -> dsl::Shape {
    dsl::Shape::new(vec![dsl::Socket::new(self.features_col.clone(), String::from("feautres"))],
    vec![dsl::Socket::new(self.prediction_col.clone(), String::from("prediction"))])
  }
}

impl Op for ExternalOp {
  type Node = Box<tform::DefaultNode>;

  fn type_id(&self) -> TypeId { TypeId::of::<External>() }
  fn op(&self) -> &'static str { "external" }

  fn name<'a>(&self, node: &'a Self::Node) -> &'a str { node.name() }

  fn model<'a>(&self, node: &'a Self::Node) -> &'a Any { DefaultNode::model(node.as_ref()) }

  fn store_model(&self,
                 obj: &Any,
                 model: &mut dsl::Model,
                 _ctx: &Context<Self::Node>) -> Result<()> {
    Ok(())
  }

  fn load_model(&self,
                model: &dsl::Model,
                _ctx: &Context<Self::Node>) -> Result<Box<Any>> {
    Ok(Box::new(ExternalModel {
      data: ptr::null()
    }) as Box<Any>)
  }

  fn node(&self, node: &Self::Node, _ctx: &Context<Self::Node>) -> dsl::Node {
    node.create_node()
  }

  fn load(&self,
          node: &dsl::Node,
          model: Box<Any>,
          _ctx: &Context<Self::Node>) -> Result<Self::Node> {
    model.downcast::<ExternalModel>().
      map_err(|_| Error::DowncastError(String::from(""))).
      and_then(|lr| {
      node.shape().get_io("features", "prediction").map(move |(i, o)| {
        Ok(Box::new(External {
          ext_transform: self.transform,
          name: node.name().to_string(),
          features_col: i.name().to_string(),
          prediction_col: o.name().to_string(),
          model: *lr
        }) as Box<DefaultNode>)
      }).unwrap_or_else(|| Err(Error::InvalidOp(String::from("Error loading External"))))
    })
  }
}

