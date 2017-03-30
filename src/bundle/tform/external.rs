use std::any::*;
use std::ptr;

use bundle::tform::{self, DefaultNode};
use bundle::ser::*;
use bundle::frame;
use bundle::dsl;

use libc;

pub type LoadModel = extern fn(*const dsl::Model) -> *const libc::c_void;
pub type Transform = extern fn(*mut frame::LeapFrame, *const libc::c_void);

extern "C" fn dummy_load_model(model: *const dsl::Model) -> *const libc::c_void {
  panic!("dummy load model has been called");
}

extern "C" fn dummy_transform(frame: *mut frame::LeapFrame, model: *const libc::c_void) {
  panic!("dummy transform has been called");
}

pub static mut OP: ExternalOp = ExternalOp {
  load_model: dummy_load_model,
  transform: dummy_transform
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
  load_model: LoadModel,
  transform: Transform
}

impl ExternalOp {
  pub fn new(load_model: LoadModel, transform: Transform) -> ExternalOp {
    ExternalOp {
      load_model: load_model,
      transform: transform
    }
  }
}

impl OpNode for External {
  fn op(&self) -> &'static str { "external" }
}

impl frame::Transformer for External {
  fn transform(&self, frame: &mut frame::LeapFrame) -> frame::Result<()> {
    (self.ext_transform)(frame as *mut frame::LeapFrame, self.model.data);
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
    Err(Error::InvalidOp(String::from("FIXME: Cannot store external model")))
  }

  fn load_model(&self,
                model: &dsl::Model,
                _ctx: &Context<Self::Node>) -> Result<Box<Any>> {
    Ok(Box::new(ExternalModel {
      data: (self.load_model)(model as *const dsl::Model)
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

