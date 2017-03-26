use super::Transformer;
use ser::*;
use std::any::*;
use dsl;

#[derive(Clone)]
pub struct LinearRegressionModel {
  coefficients: dsl::DenseTensor<f64>,
  intercept: f64
}

pub struct LinearRegression {
  name: String,
  features_col: String,
  prediction_col: String,
  model: LinearRegressionModel
}

pub struct LinearRegressionOp { }

impl OpNode for LinearRegression { }
impl Transformer for LinearRegression {
  fn name(&self) -> &str { &self.name }
  fn model(&self) -> &Any { &self.model as &Any }

  fn create_shape(&self) -> dsl::Shape {
    dsl::Shape::new(vec![dsl::Socket::new(self.features_col.clone(), String::from("feautres"))],
    vec![dsl::Socket::new(self.prediction_col.clone(), String::from("prediction"))])
  }
}

impl Op for LinearRegressionOp {
    type Node = Box<Transformer>;

    fn type_id(&self) -> TypeId { TypeId::of::<LinearRegression>() }
    fn op(&self) -> &'static str { "my_node" }

    fn name<'a>(&self, node: &'a Self::Node) -> &'a str { node.name() }

    fn model<'a>(&self, node: &'a Self::Node) -> &'a Any { Transformer::model(node.as_ref()) }

    fn store_model(&self,
                   obj: &Any,
                   model: &mut dsl::Model,
                   _ctx: &Context<Self::Node>) -> Result<()> {
      obj.downcast_ref::<LinearRegressionModel>().map(|lr| {
        model.with_attr("intercept", dsl::Attribute::Basic(dsl::BasicValue::Double(lr.intercept))).
          with_attr("coefficients", dsl::Attribute::Tensor(dsl::TensorValue::Double(lr.coefficients.clone())));
        Ok(())
      }).unwrap_or_else(|| Err(Error::InvalidOp("Expected a LinearRegressionModel".to_string())))
    }

    fn load_model(&self,
                  model: &dsl::Model,
                  _ctx: &Context<Self::Node>) -> Result<Box<Any>> {
      model.get_double("intercept").and_then(|i| {
        model.get_double_tensor("coefficients").map(|c| {
          Some(LinearRegressionModel {
            intercept: i,
            coefficients: c.clone()
          })
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
      model.downcast_ref::<LinearRegressionModel>().and_then(|lr| {
        node.shape().get_io("features", "prediction").map(|(i, o)| {
          LinearRegression {
            name: node.name().to_string(),
            features_col: i.name().to_string(),
            prediction_col: o.name().to_string(),
            model: lr.clone()
          }
        })
      }).map(|x| Ok(Box::new(x) as Box<Transformer>)).
      unwrap_or_else(|| Err(Error::DowncastError(String::from(""))))
    }
  }

