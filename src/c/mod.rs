use bundle::*;
use libc;
use std::slice;

pub extern fn mleap_frame_with_size(size: usize) -> *mut frame::LeapFrame {
  let f = Box::new(frame::LeapFrame::with_size(size));
  Box::into_raw(f)
}

pub extern fn mleap_frame_free(frame: *mut frame::LeapFrame) {
  unsafe {
    drop(Box::from_raw(frame))
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
