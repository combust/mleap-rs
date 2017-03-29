pub mod dsl;
pub mod json;
pub mod ser;
pub mod tform;
pub mod frame;

#[cfg(test)]
mod test {
  use super::*;
  use std::ffi;
  use c;

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

  #[test]
  fn test_airbnb_c() {
    let path = "/Users/hollinwilkins/Workspace/scratch/criteo/model";
    let c_path = ffi::CString::new(path).unwrap();

    let c_transformer = c::mleap_transformer_load(c_path.as_ptr());
    let c_frame = c::mleap_frame_with_size(1);

    let bathrooms = rust_to_c_string("bathrooms");
    let bedrooms = rust_to_c_string("bedrooms");
    let security_deposit = rust_to_c_string("security_deposit");
    let cleaning_fee = rust_to_c_string("cleaning_fee");
    let extra_people = rust_to_c_string("extra_people");
    let number_of_reviews = rust_to_c_string("number_of_reviews");
    let square_feet = rust_to_c_string("square_feet");
    let review_scores_rating = rust_to_c_string("review_scores_rating");

    c::mleap_frame_with_doubles(c_frame, bathrooms.as_ptr(), vec![2.0].as_ptr());
    c::mleap_frame_with_doubles(c_frame, bedrooms.as_ptr(), vec![3.0].as_ptr());
    c::mleap_frame_with_doubles(c_frame, security_deposit.as_ptr(), vec![50.0].as_ptr());
    c::mleap_frame_with_doubles(c_frame, cleaning_fee.as_ptr(), vec![30.0].as_ptr());
    c::mleap_frame_with_doubles(c_frame, extra_people.as_ptr(), vec![0.0].as_ptr());
    c::mleap_frame_with_doubles(c_frame, number_of_reviews.as_ptr(), vec![23.0].as_ptr());
    c::mleap_frame_with_doubles(c_frame, square_feet.as_ptr(), vec![1200.0].as_ptr());
    c::mleap_frame_with_doubles(c_frame, review_scores_rating.as_ptr(), vec![93.0].as_ptr());

    let strict = ffi::CString::new("strict").unwrap();
    let s10 = ffi::CString::new("1.0").unwrap();
    //let entire_home = ffi::CString::new("Entire home/apt").unwrap();
    //let ny = ffi::CString::new("NY").unwrap();

    let cp = vec![strict.as_ptr()];
    let his = vec![s10.as_ptr()];
    //let ib = vec![s10.as_ptr()];
    //let rt = vec![entire_home.as_ptr()];
    //let state = vec![ny.as_ptr()];

    println!("WHy is this {:?}", strict.as_ptr());
    println!("Different than this {:?}", *(cp.first().unwrap()));

    //c::mleap_frame_with_strings(c_frame, rust_to_c_string("cancellation_policy").as_ptr(), cp.as_ptr());
    //c::mleap_frame_with_strings(c_frame, rust_to_c_string("host_is_superhost").as_ptr(), his.as_ptr());
    //c::mleap_frame_with_strings(c_frame, rust_to_c_string("instant_bookable").as_ptr(), ib.as_ptr());
    //c::mleap_frame_with_strings(c_frame, rust_to_c_string("room_type").as_ptr(), rt.as_ptr());
    //c::mleap_frame_with_strings(c_frame, rust_to_c_string("state").as_ptr(), state.as_ptr());

    //c::mleap_transform(c_transformer, c_frame);

    //node.transform(&mut frame).unwrap();

    //let r = frame.get_doubles("price_prediction").and_then(|x| x.first()).unwrap();

    //let frame = unsafe { Box::from_raw(c_frame) };
    //let r = frame.get_doubles("price_prediction").and_then(|x| x.first()).unwrap();
    //println!("Price prediction is: {}", r);
  }

  //fn rust_to_c_string_arr(s: &[String]) -> Vec<ffi::CString> {
    //s.iter().map(|x| rust_to_c_string(x)).collect()
  //}

  fn rust_to_c_string<T: Into<Vec<u8>>>(s: T) -> ffi::CString {
    ffi::CString::new(s).unwrap()
  }
}
