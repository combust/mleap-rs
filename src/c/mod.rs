use bundle::*;
use libc;
use std::slice;

pub extern fn mleap_frame_with_size(c_size: usize) -> *mut frame::LeapFrame {
  let f = Box::new(frame::LeapFrame::with_size(c_size));
  Box::into_raw(f)
}

pub extern fn mleap_frame_free(c_frame: *mut frame::LeapFrame) {
  unsafe {
    drop(Box::from_raw(c_frame))
  }
}

pub extern fn mleap_frame_with_doubles(c_frame: *mut frame::LeapFrame,
                                       c_name: *const u8,
                                       c_values: *const f64) {
  unsafe {
    let mut frame: Box<frame::LeapFrame> = Box::from_raw(c_frame);
    let name = String::from_utf8_unchecked(slice::from_raw_parts(c_name, libc::strlen(c_name as *const i8)).clone().to_vec());
    let values = slice::from_raw_parts(c_values, frame.size()).clone().to_vec();
    frame.try_with_doubles(name, values).unwrap();
  }
}

pub extern fn mleap_frame_with_strings(c_frame: *mut frame::LeapFrame,
                                       c_name: *const u8,
                                       c_values: *const *const u8) {
  unsafe {
    let mut frame: Box<frame::LeapFrame> = Box::from_raw(c_frame);
    let name = String::from_utf8_unchecked(slice::from_raw_parts(c_name, libc::strlen(c_name as *const i8)).clone().to_vec());
    let values: Vec<String> = slice::from_raw_parts(c_values, frame.size()).clone().iter().map(|s| {
      let len = libc::strlen(*s as *const i8);
      String::from_utf8_unchecked(slice::from_raw_parts(*s, len).clone().to_vec())
    }).collect();
    frame.try_with_strings(name, values).unwrap();
  }
}

pub extern fn mleap_transformer_load(c_path: *const u8) -> *mut Box<tform::DefaultNode> {
  unsafe {
    let path = String::from_utf8_unchecked(slice::from_raw_parts(c_path, libc::strlen(c_path as *const i8)).clone().to_vec());
    let builder = ser::FileBuilder::try_new(path).unwrap();
    let mut registry = ser::Registry::new();

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
}

pub extern fn mleap_transformer_free(c_transformer: *mut Box<tform::DefaultNode>) {
  unsafe {
    drop(Box::from_raw(c_transformer))
  }
}

pub extern fn mleap_transform(c_transformer: *mut Box<tform::DefaultNode>,
                              c_frame: *mut frame::LeapFrame) {
  unsafe {
    let transformer = Box::from_raw(c_transformer);
    let mut frame = Box::from_raw(c_frame);

    transformer.transform(&mut frame).unwrap();
  }
}
