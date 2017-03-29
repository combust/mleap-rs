use bundle::*;
use std::slice;
use std::ffi;

#[no_mangle]
pub extern fn mleap_frame_with_size(c_size: usize) -> *mut frame::LeapFrame {
  let f = Box::new(frame::LeapFrame::with_size(c_size));
  Box::into_raw(f)
}

#[no_mangle]
pub extern fn mleap_frame_free(c_frame: *mut frame::LeapFrame) {
  unsafe {
    drop(Box::from_raw(c_frame))
  }
}

#[no_mangle]
pub extern fn mleap_frame_with_doubles(c_frame: *mut frame::LeapFrame,
                                       c_name: *const i8,
                                       c_values: *const f64) {
  unsafe {
    let mut frame: Box<frame::LeapFrame> = Box::from_raw(c_frame);
    let name = c_string_to_rust(c_name);
    let values = slice::from_raw_parts(c_values, frame.size()).to_vec();
    frame.try_with_doubles(name, values).unwrap();
  }
}

#[no_mangle]
pub extern fn mleap_frame_with_strings(c_frame: *mut frame::LeapFrame,
                                       c_name: *const i8,
                                       c_values: *const *const i8) {
  unsafe {
    let mut frame: Box<frame::LeapFrame> = Box::from_raw(c_frame);
    let name = c_string_to_rust(c_name);
    let values: Vec<String> = slice::from_raw_parts(c_values, frame.size()).iter().map(|s| {
      println!("PTR: {:?}", *s);
      c_string_to_rust(*s)
    }).collect();
    frame.try_with_strings(name, values).unwrap();
  }
}

#[no_mangle]
pub extern fn mleap_transformer_load(c_path: *const i8) -> *mut Box<tform::DefaultNode> {
  let path = c_string_to_rust(c_path);
  let builder = ser::FileBuilder::try_new(path).unwrap();
  let mut registry: ser::Registry<Box<tform::DefaultNode>> = ser::Registry::new();

  registry.insert_op(tform::linear_regression::OP);
  registry.insert_op(tform::string_indexer::OP);
  registry.insert_op(tform::one_hot_encoder::OP);
  registry.insert_op(tform::pipeline::OP);
  registry.insert_op(tform::vector_assembler::OP);
  registry.insert_op(tform::standard_scaler::OP);

  let ctx = ser::Context::new(Box::new(builder), &registry);

  let (_, transformer) = ctx.read_bundle().unwrap();
  let r = Box::new(transformer);
  Box::into_raw(r)
}

#[no_mangle]
pub extern fn mleap_transformer_free(c_transformer: *mut Box<tform::DefaultNode>) {
  unsafe {
    drop(Box::from_raw(c_transformer))
  }
}

#[no_mangle]
pub extern fn mleap_transform(c_transformer: *mut Box<tform::DefaultNode>,
                              c_frame: *mut frame::LeapFrame) {
  unsafe {
    let transformer = Box::from_raw(c_transformer);
    let mut frame = Box::from_raw(c_frame);

    transformer.transform(frame.as_mut()).unwrap();
  }
}

fn c_string_to_rust(null_terminated_string: *const i8) -> String {
  unsafe {
    let c_str: &ffi::CStr = ffi::CStr::from_ptr(null_terminated_string);
    String::from(c_str.to_str().unwrap())
  }
}
