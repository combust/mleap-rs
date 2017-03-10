use std::collections::HashMap;
use uuid::Uuid;
use semver::Version;

pub trait ProtoSerializer<T: Sized> {
  fn serialize(&self, t: &T) -> Vec<u8>;
}

pub trait ProtoDeserializer<T: Sized> {
  fn deserialize(&self, b: &[u8]) -> T;
}

pub trait ProtoSerialization<T: Sized> : ProtoSerializer<T> + ProtoDeserializer<T> { }

pub enum VectorValue {
  Bool(Vec<bool>),
  Byte(Vec<i8>),
  Short(Vec<i16>),
  Int(Vec<i32>),
  Long(Vec<i32>),
  Float(Vec<f32>),
  Double(Vec<f64>),
  ByteString(Vec<Vec<i8>>)
}

pub enum ArrayValue {
  Vector(VectorValue),
  Array(Box<ArrayValue>)
}

pub enum Value {
  Bool(bool),
  Byte(i8),
  Short(i16),
  Int(i32),
  Long(i32),
  Float(f32),
  Double(f64),
  ByteString(Vec<i8>),
  Array(ArrayValue),
  Tensor(Vec<usize>, VectorValue)
}

pub struct Attribute(Value);

pub struct Model {
  op: String,
  attributes: HashMap<String, Value>
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

pub struct ProtoSerializationImpl<'a, T: Sized + 'a> {
  serialize: &'a Fn(&T) -> Vec<u8>,
  deserialize: &'a Fn(&[u8]) -> T
}

impl<'a, T: Sized + 'a> ProtoSerializer<T> for ProtoSerializationImpl<'a, T> {
  fn serialize(&self, t: &T) -> Vec<u8> { (self.serialize)(t) }
}

impl<'a, T: Sized + 'a> ProtoDeserializer<T> for ProtoSerializationImpl<'a, T> {
  fn deserialize(&self, b: &[u8]) -> T { (self.deserialize)(b) }
}

impl<'a, T: Sized + 'a> ProtoSerialization<T> for ProtoSerializationImpl<'a, T> { }
