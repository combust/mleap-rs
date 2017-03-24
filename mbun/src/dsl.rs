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

#[derive(Clone)]
pub struct Socket {
  name: String,
  port: String
}

pub struct Shape {
  inputs: Vec<Socket>,
  outputs: Vec<Socket>
}

pub struct Model {
  op: String,
  attributes: HashMap<String, Attribute>
}

pub struct Node {
  name: String,
  shape: Shape
}

pub enum ConcreteFormat {
  Json,
  Proto
}

pub enum Format {
  Concrete(ConcreteFormat),
  Mixed
}

pub struct Bundle {
  uid: Uuid,
  name: String,
  format: Format,
  version: Version
}

impl Socket {
  pub fn new(name: String, port: String) -> Socket {
    Socket {
      name: name,
      port: port
    }
  }

  pub fn name(&self) -> &str { &self.name }
  pub fn port(&self) -> &str { &self.port }
}

impl Shape {
  pub fn new(inputs: Vec<Socket>, outputs: Vec<Socket>) -> Shape {
    Shape {
      inputs: inputs,
      outputs: outputs
    }
  }

  pub fn inputs(&self) -> &[Socket] { &self.inputs }
  pub fn outputs(&self) -> &[Socket] { &self.outputs }
}

impl Model {
  pub fn new(op: String,
             attributes: HashMap<String, Attribute>) -> Model {
    Model {
      op: op,
      attributes: attributes
    }
  }

  pub fn op(&self) -> &str { &self.op }
  pub fn attributes(&self) -> &HashMap<String, Attribute> { &self.attributes }

  pub fn with_attr(&mut self, name: &str, attr: Attribute) -> &mut Self {
    self.attributes.insert(name.to_string(), attr);
    self
  }
}

impl Node {
  pub fn new(name: String, shape: Shape) -> Node {
    Node {
      name: name,
      shape: shape
    }
  }

  pub fn name(&self) -> &str { &self.name }
  pub fn shape(&self) -> &Shape { &self.shape }
}

impl Bundle {
  pub fn new(uid: Uuid,
             name: String,
             format: Format,
             version: Version) -> Bundle {
    Bundle {
      uid: uid,
      name: name,
      format: format,
      version: version
    }
  }

  pub fn uid(&self) -> &Uuid { &self.uid }
  pub fn name(&self) -> &str { &self.name }
  pub fn format(&self) -> &Format { &self.format }
  pub fn version(&self) -> &Version { &self.version }
}
