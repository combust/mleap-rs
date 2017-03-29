use std::any::*;

use bundle::tform::{self, DefaultNode};
use bundle::ser::*;
use super::frame;
use super::dsl;

pub const OP: &PipelineOp = &PipelineOp { };

pub struct PipelineModel {
  children: Vec<Box<DefaultNode>>
}

pub struct Pipeline {
  name: String,
  model: PipelineModel
}

pub struct PipelineOp { }

impl OpNode for Pipeline {
  fn op(&self) -> &'static str { "pipeline" }
}

impl frame::Transformer for Pipeline {
  fn transform(&self, frame: &mut frame::LeapFrame) -> frame::Result<()> {
    for child in self.model.children.iter() {
      match child.transform(frame) {
        Ok(_) => { },
        Err(err) => return Err(err)
      }
    }

    Ok(())
  }
}

impl DefaultNode for Pipeline {
  fn name(&self) -> &str { &self.name }
  fn model(&self) -> &Any { &self.model as &Any }

  fn create_shape(&self) -> dsl::Shape {
    dsl::Shape::empty()
  }
}

impl Op for PipelineOp {
  type Node = Box<tform::DefaultNode>;

  fn type_id(&self) -> TypeId { TypeId::of::<Pipeline>() }
  fn op(&self) -> &'static str { "pipeline" }

  fn name<'a>(&self, node: &'a Self::Node) -> &'a str { node.name() }

  fn model<'a>(&self, node: &'a Self::Node) -> &'a Any { DefaultNode::model(node.as_ref()) }

  fn store_model(&self,
                 obj: &Any,
                 model: &mut dsl::Model,
                 ctx: &Context<Self::Node>) -> Result<()> {
    obj.downcast_ref::<PipelineModel>().map(|pipeline| {
      let names: Vec<String> = pipeline.children.iter().map(|c| c.name().to_string()).collect();

      for child in pipeline.children.iter() {
        let n_name = format!("{}.node", child.name());
        let r = ctx.try_next(&n_name).and_then(|ctx| ctx.write_node_and_model(child));

        match r {
          Ok(_) => { }, // do nothing
          Err(err) => return Err(err)
        }
      }

      model.with_attr("nodes", dsl::Attribute::Array(dsl::VectorValue::String(names)));
      Ok(())
    }).unwrap_or_else(|| Err(Error::InvalidOp("Expected a PipelineModel".to_string())))
  }

  fn load_model(&self,
                model: &dsl::Model,
                ctx: &Context<Self::Node>) -> Result<Box<Any>> {
    model.get_string_vector("nodes").map(|children| {
      let mut c_nodes: Vec<Box<DefaultNode>> = Vec::with_capacity(children.len());

      for name in children.iter() {
        let n_name = format!("{}.node", name);
        let r = ctx.try_next(&n_name).and_then(|ctx| ctx.read_node());

        match r {
          Ok(c_node) => c_nodes.push(c_node),
          Err(err) => return Err(err)
        }
      }

      let pm = Box::new(PipelineModel {
        children: c_nodes
      });
      Ok(pm as Box<Any>)
    }).unwrap_or_else(|| Err(Error::InvalidModel(String::from("Invalid pipeline, no children"))))
  }

  fn node(&self, node: &Self::Node, _ctx: &Context<Self::Node>) -> dsl::Node {
    node.create_node()
  }

  fn load(&self,
          node: &dsl::Node,
          model: Box<Any>,
          _ctx: &Context<Self::Node>) -> Result<Self::Node> {
    model.downcast::<PipelineModel>().map(|pm| {
      Box::new(Pipeline {
        name: String::from(node.name()),
        model: *pm
      }) as Box<DefaultNode>
    }).map_err(|_| Error::DowncastError(String::from("")))
  }
}
