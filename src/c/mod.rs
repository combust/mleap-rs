use bundle::*;
use std::slice;
use std::ffi;
use std::ptr;
use std::os::raw::c_char;

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
    let frame = c_frame.as_mut().unwrap();
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
    let frame = c_frame.as_mut().unwrap();
    let name = c_string_to_rust(c_name);
    let values: Vec<String> = slice::from_raw_parts(c_values, frame.size()).iter().map(|s| {
      c_string_to_rust(*s)
    }).collect();
    frame.try_with_strings(name, values).unwrap();
  }
}

#[no_mangle]
pub extern fn mleap_frame_get_doubles(c_frame: *mut frame::LeapFrame,
                                      c_name: *const i8,
                                      c_buffer: *mut f64) {
  unsafe {
    let frame = c_frame.as_mut().unwrap();
    let name = c_string_to_rust(c_name);
    let values = frame.get_doubles(&name).unwrap();
    ptr::copy_nonoverlapping(values.as_ptr(), c_buffer, values.len());
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
pub extern fn mleap_transformer_load_ex(c_path: *const i8, c_transform: tform::external::Transform) -> *mut Box<tform::DefaultNode> {
    let path = c_string_to_rust(c_path);
    let builder = ser::FileBuilder::try_new(path).unwrap();
    let mut registry: ser::Registry<Box<tform::DefaultNode>> = ser::Registry::new();

    registry.insert_op(tform::linear_regression::OP);
    registry.insert_op(tform::string_indexer::OP);
    registry.insert_op(tform::one_hot_encoder::OP);
    registry.insert_op(tform::pipeline::OP);
    registry.insert_op(tform::vector_assembler::OP);
    registry.insert_op(tform::standard_scaler::OP);
    // UNSAFE: modifying the singleton
    unsafe {
      tform::external::OP = tform::external::ExternalOp { transform: c_transform };
      registry.insert_op(&tform::external::OP);
    }

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
    let transformer = c_transformer.as_ref().unwrap();
    let frame = c_frame.as_mut().unwrap();

    transformer.transform(frame).unwrap();
  }
}

pub fn c_string_to_rust(null_terminated_string: *const c_char) -> String {
  unsafe {
    let c_str = ffi::CStr::from_ptr(null_terminated_string);
    String::from(c_str.to_str().unwrap())
  }
}
