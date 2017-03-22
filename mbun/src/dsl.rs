use std::collections::HashMap;
use uuid::Uuid;
use semver::Version;

pub enum VectorValue {
  Bool(Vec<bool>),
  Byte(Vec<i8>),
  Short(Vec<i16>),
  Int(Vec<i32>),
  Long(Vec<i64>),
  Float(Vec<f32>),
  Double(Vec<f64>),
  ByteString(Vec<Vec<u8>>)
}

pub struct TensorValue {
  pub dimensions: Vec<usize>,
  pub values: VectorValue
}

pub enum BasicValue {
  Bool(bool),
  Byte(i8),
  Short(i16),
  Int(i32),
  Long(i64),
  Float(f32),
  Double(f64),
  ByteString(Vec<u8>)
}

pub enum Attribute {
  Basic(BasicValue),
  Array(VectorValue),
  Tensor(TensorValue)
}

pub struct Model {
  op: String,
  attributes: HashMap<String, Attribute>
}

pub struct Socket {
  name: String,
  port: String
}

pub struct Shape {
  inputs: Vec<Socket>,
  outputs: Vec<Socket>
}

pub struct Node {
  name: String,
  shape: Shape
}

pub enum Format {
  Json,
  Proto,
  Mixed
}

pub struct Bundle {
  uid: Uuid,
  name: String,
  format: Format,
  version: Version
}
