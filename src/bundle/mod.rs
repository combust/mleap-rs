pub mod dsl;
pub mod json;
pub mod ser;
pub mod tform;
pub mod frame;

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn test_airbnb() {
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
    let ctx2 = ctx.try_next("root/linear_regression.node").unwrap();
    let node = ctx2.read_node();

    //println!("{:?}", &dsl_model);
  }
}