use std::collections::HashMap;
use uuid::Uuid;
use semver::Version;

#[derive(Clone)]
pub struct DenseTensor<T> {
  dimensions: Vec<usize>,
  values: Vec<T>
}

pub enum VectorValue {
  Bool(Vec<bool>),
  String(Vec<String>),
  Byte(Vec<i8>),
  Short(Vec<i16>),
  Int(Vec<i32>),
  Long(Vec<i64>),
  Float(Vec<f32>),
  Double(Vec<f64>),
  ByteString(Vec<Vec<u8>>)
}

pub enum TensorValue {
  Bool(DenseTensor<bool>),
  String(DenseTensor<String>),
  Byte(DenseTensor<i8>),
  Short(DenseTensor<i16>),
  Int(DenseTensor<i32>),
  Long(DenseTensor<i64>),
  Float(DenseTensor<f32>),
  Double(DenseTensor<f64>),
  ByteString(DenseTensor<Vec<u8>>)
}

pub enum BasicValue {
  Bool(bool),
  String(String),
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

impl<T> DenseTensor<T> {
  pub fn new(dimensions: Vec<usize>, values: Vec<T>) -> DenseTensor<T> {
    DenseTensor {
      dimensions: dimensions,
      values: values
    }
  }

  pub fn dimensions(&self) -> &[usize] { &self.dimensions }
  pub fn values(&self) -> &[T] { &self.values }
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

  pub fn empty() -> Shape {
    Shape {
      inputs: vec![],
      outputs: vec![]
    }
  }

  pub fn with_standard_io(input: String, output: String) -> Shape {
    Shape {
      inputs: vec![Socket::new(input, String::from("input"))],
      outputs: vec![Socket::new(output, String::from("output"))]
    }
  }

  pub fn inputs(&self) -> &[Socket] { &self.inputs }
  pub fn outputs(&self) -> &[Socket] { &self.outputs }

  pub fn get_standard_io(&self) -> Option<(&Socket, &Socket)> {
    self.get_io("input", "output")
  }

  pub fn get_io(&self, input: &str, output: &str) -> Option<(&Socket, &Socket)> {
    self.inputs.iter().find(|s| s.port() == input).and_then(|i| {
      self.outputs.iter().find(|s| s.port() == output).map(|o| (i, o))
    })
  }

  pub fn get_output(&self, port: &str) -> Option<&Socket> {
    self.outputs.iter().find(|s| s.port() == port)
  }
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

  pub fn get_attr(&self, name: &str) -> Option<&Attribute> {
    self.attributes.get(name)
  }

  pub fn get_double(&self, name: &str) -> Option<f64> {
    self.attributes.get(name).and_then(|x| {
      match x {
        &Attribute::Basic(BasicValue::Double(i)) => Some(i),
        _ => None
      }
    })
  }

  pub fn get_double_tensor(&self, name: &str) -> Option<&DenseTensor<f64>> {
    self.attributes.get(name).and_then(|x| {
      match x {
        &Attribute::Tensor(TensorValue::Double(ref tensor)) => Some(tensor),
        _ => None
      }
    })
  }

  pub fn get_string_vector(&self, name: &str) -> Option<&[String]> {
    self.attributes.get(name).and_then(|x| {
      match x {
        &Attribute::Array(VectorValue::String(ref v)) => Some(v.as_slice()),
        _ => None
      }
    })
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
