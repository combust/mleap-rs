use ser::*;
use std::any::*;
use tform::{self, DefaultNode};
use frame;
use dsl;

pub struct OneHotEncoderModel {
  size: usize
}

pub struct OneHotEncoder {
  name: String,
  input_col: String,
  output_col: String,
  model: OneHotEncoderModel
}

pub struct OneHotEncoderOp { }

impl OpNode for OneHotEncoder { }

impl OneHotEncoderModel {
  pub fn try_encode_col_data(&self, data: &frame::ColData) -> frame::Result<frame::ColData> {
    (match data {
      &frame::ColData::Byte(ref v) => self.try_encode(v, |x| x as usize),
      &frame::ColData::Short(ref v) => self.try_encode(v, |x| x as usize),
      &frame::ColData::Int(ref v) => self.try_encode(v, |x| x as usize),
      &frame::ColData::Long(ref v) => self.try_encode(v, |x| x as usize),
      _ => Err(frame::Error::TransformError(String::from("")))
    }).map(|v| frame::ColData::LongTensor(v))
  }

  fn try_encode<T: Copy, F>(&self, v: &[T], f: F) -> frame::Result<Vec<dsl::DenseTensor<i64>>>
    where F: Fn(T) -> usize {
      let mut tensors: Vec<dsl::DenseTensor<i64>> = Vec::with_capacity(v.len());

      for raw_v in v.iter() {
        let index: usize = f(*raw_v);
        let mut values: Vec<i64> = (0..self.size).map(|_| 0).collect();

        if index < values.len() {
          values[index] = 1;
          tensors.push(dsl::DenseTensor::new(vec![values.len()], values));
        } else {
          return Err(frame::Error::TransformError(String::from("")))
        }
      }

      Ok(tensors)
    }
}

impl frame::Transformer for OneHotEncoder {
  fn transform(&self, frame: &mut frame::LeapFrame) -> frame::Result<()> {
    frame.try_col(&self.input_col).and_then(|col| {
      self.model.try_encode_col_data(col.data())
    }).and_then(|oh_col| {
      frame.try_with_col(frame::Col::new(self.output_col.clone(), oh_col)).map(|_| ())
    })
  }
}

impl DefaultNode for OneHotEncoder {
  fn name(&self) -> &str { &self.name }
  fn model(&self) -> &Any { &self.model as &Any }

  fn create_shape(&self) -> dsl::Shape {
    dsl::Shape::with_standard_io(self.input_col.clone(), self.output_col.clone())
  }
}

impl Op for OneHotEncoderOp {
  type Node = Box<tform::DefaultNode>;

  fn type_id(&self) -> TypeId { TypeId::of::<OneHotEncoder>() }
  fn op(&self) -> &'static str { "one_hot_encoder" }

  fn name<'a>(&self, node: &'a Self::Node) -> &'a str { node.name() }

  fn model<'a>(&self, node: &'a Self::Node) -> &'a Any { DefaultNode::model(node.as_ref()) }

  fn store_model(&self,
                 obj: &Any,
                 model: &mut dsl::Model,
                 _ctx: &Context<Self::Node>) -> Result<()> {
    obj.downcast_ref::<OneHotEncoderModel>().map(|oh| {
      model.with_attr("size", dsl::Attribute::Basic(dsl::BasicValue::Long(oh.size as i64)));
      Ok(())
    }).unwrap_or_else(|| Err(Error::InvalidOp("Expected a OneHotEncoderModel".to_string())))
  }

  fn load_model(&self,
                model: &dsl::Model,
                _ctx: &Context<Self::Node>) -> Result<Box<Any>> {
    model.get_long("size").and_then(|i| {
      Some(OneHotEncoderModel {
        size: i as usize,
      })
    }).map(|x| Ok(Box::new(x) as Box<Any>)).unwrap_or_else(|| Err(Error::InvalidModel("".to_string())))
  }

  fn node(&self, node: &Self::Node, _ctx: &Context<Self::Node>) -> dsl::Node {
    node.create_node()
  }

  fn load(&self,
          node: &dsl::Node,
          model: Box<Any>,
          _ctx: &Context<Self::Node>) -> Result<Self::Node> {
    model.downcast::<OneHotEncoderModel>().
      map_err(|_| Error::DowncastError(String::from(""))).
      and_then(|oh| {
      node.shape().get_standard_io().map(move |(i, o)| {
        Ok(Box::new(OneHotEncoder {
          name: node.name().to_string(),
          input_col: i.name().to_string(),
          output_col: o.name().to_string(),
          model: *oh
        }) as Box<DefaultNode>)
      }).unwrap_or_else(|| Err(Error::InvalidOp(String::from(""))))
    })
  }
}

