use std::any::*;

use bundle::tform::{self, DefaultNode};
use bundle::ser::*;
use bundle::frame;
use bundle::dsl;

pub const OP: &'static StandardScalerOp = &StandardScalerOp { };

pub struct StandardScalerModel {
  mean: Option<dsl::DenseTensor<f64>>,
  std: Option<dsl::DenseTensor<f64>>
}

pub struct StandardScaler {
  name: String,
  input_col: String,
  output_col: String,
  model: StandardScalerModel
}

pub struct StandardScalerOp { }

impl StandardScalerModel {
  pub fn try_scale(&self, data: &frame::ColData) -> frame::Result<frame::ColData> {
    match data {
      &frame::ColData::DoubleTensor(ref data) => {
        match (&self.mean, &self.std) {
          (&None, &Some(ref std)) => {
            let col_data: Vec<dsl::DenseTensor<f64>> = data.iter().map(|features| {
              let vs: Vec<f64> = features.values().iter().zip(std.values().iter()).map(|(f, s)| {
                if *s != 0.0 {
                  f * (1.0 / *s)
                } else {
                  0.0
                }
              }).collect();

              dsl::DenseTensor::new(vec![vs.len()], vs)
            }).collect();

            Ok(frame::ColData::DoubleTensor(col_data))
          },
          _ => Err(frame::Error::TransformError(String::from("Must provide a mean or stddev for standard scaler")))
        }
      },
      _ => Err(frame::Error::InvalidType(String::from("Expected double tensors")))
    }
  }
}
impl OpNode for StandardScaler {
  fn op(&self) -> &'static str { "standard_scaler" }
}

impl frame::Transformer for StandardScaler {
  fn transform(&self, frame: &mut frame::LeapFrame) -> frame::Result<()> {
    frame.try_col(&self.input_col).and_then(|features_col| {
      self.model.try_scale(features_col.data())
    }).and_then(|scaled| {
      frame.try_with_col(frame::Col::new(self.output_col.clone(), scaled)).map(|_| ())
    })
  }
}

impl DefaultNode for StandardScaler {
  fn name(&self) -> &str { &self.name }
  fn model(&self) -> &Any { &self.model as &Any }

  fn create_shape(&self) -> dsl::Shape {
    dsl::Shape::with_standard_io(self.input_col.clone(), self.output_col.clone())
  }
}

impl Op for StandardScalerOp {
  type Node = Box<tform::DefaultNode>;

  fn type_id(&self) -> TypeId { TypeId::of::<StandardScaler>() }
  fn op(&self) -> &'static str { "standard_scaler" }

  fn name<'a>(&self, node: &'a Self::Node) -> &'a str { node.name() }

  fn model<'a>(&self, node: &'a Self::Node) -> &'a Any { DefaultNode::model(node.as_ref()) }

  fn store_model(&self,
                 obj: &Any,
                 model: &mut dsl::Model,
                 _ctx: &Context<Self::Node>) -> Result<()> {
    obj.downcast_ref::<StandardScalerModel>().map(|m| {
      for mean in &m.mean {
        model.with_attr("mean", dsl::Attribute::Tensor(dsl::TensorValue::Double(mean.clone())));
      }
      for std in &m.std {
        model.with_attr("std", dsl::Attribute::Tensor(dsl::TensorValue::Double(std.clone())));
      }

      Ok(())
    }).unwrap_or_else(|| Err(Error::InvalidOp(String::from("Expected a StandardScalerModel"))))
  }

  fn load_model(&self,
                model: &dsl::Model,
                _ctx: &Context<Self::Node>) -> Result<Box<Any>> {
    let mean = model.get_double_tensor("mean").map(|m| m.clone());
    let std = model.get_double_tensor("std").map(|m| m.clone());

    Ok(Box::new(StandardScalerModel {
      mean: mean,
      std: std
    }) as Box<Any>)
  }

  fn node(&self, node: &Self::Node, _ctx: &Context<Self::Node>) -> dsl::Node {
    node.create_node()
  }

  fn load(&self,
          node: &dsl::Node,
          model: Box<Any>,
          _ctx: &Context<Self::Node>) -> Result<Self::Node> {
    model.downcast::<StandardScalerModel>().
      map_err(|_| Error::DowncastError(String::from("Must provide a StandardScalerModel"))).
      and_then(|ss| {
      node.shape().get_standard_io().map(move |(i, o)| {
        Ok(Box::new(StandardScaler {
          name: node.name().to_string(),
          input_col: i.name().to_string(),
          output_col: o.name().to_string(),
          model: *ss
        }) as Box<DefaultNode>)
      }).unwrap_or_else(|| Err(Error::InvalidOp(String::from("Error loading StandardScaler"))))
    })
  }
}
