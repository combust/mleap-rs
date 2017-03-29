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
    registry.insert_op(tform::standard_scaler::OP);

    let ctx = ser::Context::new(Box::new(builder), &registry);
    let ctx2 = ctx.try_next("root").unwrap();
    let node = ctx2.read_node().unwrap();

    let mut frame = frame::LeapFrame::with_size(1);
    frame.try_with_doubles(String::from("bathrooms"), vec![2.0]).unwrap();
    frame.try_with_doubles(String::from("bedrooms"), vec![3.0]).unwrap();
    frame.try_with_doubles(String::from("security_deposit"), vec![50.0]).unwrap();
    frame.try_with_doubles(String::from("cleaning_fee"), vec![30.0]).unwrap();
    frame.try_with_doubles(String::from("extra_people"), vec![0.0]).unwrap();
    frame.try_with_doubles(String::from("number_of_reviews"), vec![23.0]).unwrap();
    frame.try_with_doubles(String::from("square_feet"), vec![1200.0]).unwrap();
    frame.try_with_doubles(String::from("review_scores_rating"), vec![93.0]).unwrap();

    frame.try_with_strings(String::from("cancellation_policy"), vec![String::from("strict")]).unwrap();
    frame.try_with_strings(String::from("host_is_superhost"), vec![String::from("1.0")]).unwrap();
    frame.try_with_strings(String::from("instant_bookable"), vec![String::from("1.0")]).unwrap();
    frame.try_with_strings(String::from("room_type"), vec![String::from("Entire home/apt")]).unwrap();
    frame.try_with_strings(String::from("state"), vec![String::from("NY")]).unwrap();

    node.transform(&mut frame).unwrap();

    let r = frame.get_doubles("price_prediction").and_then(|x| x.first()).unwrap();

    println!("Price prediction is: {}", r);
  }
}
