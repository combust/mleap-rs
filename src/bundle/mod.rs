pub mod dsl;
pub mod json;
pub mod ser;
pub mod tform;
pub mod frame;

#[cfg(test)]
mod test {
  use super::*;
  use rand::{self, Rng};

  #[test]
  fn test_airbnb() {
    let mut rng = rand::thread_rng();
    let path = "/Users/hollinwilkins/Workspace/scratch/criteo/model";
    let builder = ser::FileBuilder::try_new(path).unwrap();
    let mut registry = ser::Registry::new();

    registry.insert_op(tform::linear_regression::OP);
    registry.insert_op(tform::string_indexer::OP);
    registry.insert_op(tform::one_hot_encoder::OP);
    registry.insert_op(tform::pipeline::OP);
    registry.insert_op(tform::vector_assembler::OP);

    let ctx = ser::Context::new(Box::new(builder), &registry);
    let bundle = ctx.read_dsl_bundle().unwrap();
    let ctx2 = ctx.try_next("root").unwrap();
    let node = ctx2.read_node().unwrap();

    let feature_vec: Vec<f64> = (0..27).map(|_| rng.gen()).collect();
    let feature_tensor = dsl::DenseTensor::new(vec![27], feature_vec);
    let feature_col_data = frame::ColData::DoubleTensor(vec![feature_tensor]);
    let feature_col = frame::Col::new(String::from("features"), feature_col_data);
    let mut frame = frame::LeapFrame::with_size(1);
    frame.try_with_col(feature_col).unwrap();

    node.transform(&mut frame).unwrap();
    let r = frame.get_doubles("price_prediction").and_then(|x| x.first()).unwrap();
  }
}
