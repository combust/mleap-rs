use serde_json::Value;
use serde_json::map::Map;
use std::collections::HashMap;
use std::result::Result;
use uuid::Uuid;
use semver::Version;
use base64;
use dsl;

#[derive(Debug, Clone)]
pub enum Error {
  WriteError(String),
  ReadError(String)
}

pub trait TryFrom<T> where Self: Sized {
  type Err;

  fn try_from(obj: T) -> Result<Self, Self::Err>;
}

impl<'a> From<&'a dsl::Attribute> for Value {
  fn from(value: &'a dsl::Attribute) -> Self {
    match value {
      &dsl::Attribute::Basic(ref basic) => {
        let (t, v) = match basic {
          &dsl::BasicValue::Bool(v) => ("bool", Value::from(v)),
          &dsl::BasicValue::String(ref v) => ("string", Value::from(v.as_str())),
          &dsl::BasicValue::Byte(v) => ("byte", Value::from(v)),
          &dsl::BasicValue::Short(v) => ("short", Value::from(v)),
          &dsl::BasicValue::Int(v) => ("int", Value::from(v)),
          &dsl::BasicValue::Long(v) => ("long", Value::from(v)),
          &dsl::BasicValue::Float(v) => ("float", Value::from(v)),
          &dsl::BasicValue::Double(v) => ("double", Value::from(v)),
          &dsl::BasicValue::ByteString(ref v) => ("byte_string", Value::from(base64::encode(v))),
        };

        let mut map = Map::with_capacity(2);
        map.insert(String::from("type"), Value::from(t));
        map.insert(String::from("value"), v);

        Value::Object(map)
      },
      &dsl::Attribute::Tensor(ref tv) => {
        let (b, d, v) = match tv {
          &dsl::TensorValue::Bool(ref v) => ("bool", v.dimensions(), Value::from(v.values())),
          &dsl::TensorValue::String(ref v) => ("string", v.dimensions(), Value::from(v.values())),
          &dsl::TensorValue::Byte(ref v) => ("byte", v.dimensions(), Value::from(v.values())),
          &dsl::TensorValue::Short(ref v) => ("short", v.dimensions(), Value::from(v.values())),
          &dsl::TensorValue::Int(ref v) => ("int", v.dimensions(), Value::from(v.values())),
          &dsl::TensorValue::Long(ref v) => ("long", v.dimensions(), Value::from(v.values())),
          &dsl::TensorValue::Float(ref v) => ("float", v.dimensions(), Value::from(v.values())),
          &dsl::TensorValue::Double(ref v) => ("double", v.dimensions(), Value::from(v.values())),
          &dsl::TensorValue::ByteString(ref v) => {
            let vs: Vec<String> = v.values().iter().map(|s| base64::encode(s)).collect();
            ("byte_string", v.dimensions(), Value::from(vs.as_slice()))
          }
        };

        let mut tmap = Map::with_capacity(2);
        tmap.insert(String::from("dimensions"), Value::from(d));
        tmap.insert(String::from("values"), v);

        let mut map = Map::with_capacity(3);
        map.insert(String::from("type"), Value::from("tensor"));
        map.insert(String::from("base"), Value::from(b));
        map.insert(String::from("value"), Value::Object(tmap));

        Value::Object(map)
      },
      &dsl::Attribute::Array(ref values) => {
        let (b, v) = match values {
          &dsl::VectorValue::Bool(ref v) => ("bool", Value::from(v.as_slice())),
          &dsl::VectorValue::String(ref v) => ("string", Value::from(v.as_slice())),
          &dsl::VectorValue::Byte(ref v) => ("byte", Value::from(v.as_slice())),
          &dsl::VectorValue::Short(ref v) => ("short", Value::from(v.as_slice())),
          &dsl::VectorValue::Int(ref v) => ("int", Value::from(v.as_slice())),
          &dsl::VectorValue::Long(ref v) => ("long", Value::from(v.as_slice())),
          &dsl::VectorValue::Float(ref v) => ("float", Value::from(v.as_slice())),
          &dsl::VectorValue::Double(ref v) => ("double", Value::from(v.as_slice())),
          &dsl::VectorValue::ByteString(ref v) => {
            let vs: Vec<String> = v.iter().map(|s| base64::encode(s)).collect();
            ("byte_string", Value::from(vs.as_slice()))
          }
        };

        let mut map = Map::with_capacity(3);
        map.insert(String::from("type"), Value::from("tensor"));
        map.insert(String::from("base"), Value::from(b));
        map.insert(String::from("value"), v);

        Value::Object(map)
      }
    }
  }
}

fn basic_attribute(basic: dsl::BasicValue) -> dsl::Attribute {
  dsl::Attribute::Basic(basic)
}

impl<'a> TryFrom<&'a Value> for usize {
  type Err = Error;

  fn try_from(value: &'a Value) -> Result<Self, Self::Err> {
    value.as_u64().map(|v| Ok(v as usize)).unwrap_or_else(|| Err(Error::ReadError("invalid usize".to_string())))
  }
}

impl<'a> TryFrom<&'a Value> for bool {
  type Err = Error;

  fn try_from(value: &'a Value) -> Result<Self, Self::Err> {
    value.as_bool().map(|v| Ok(v)).unwrap_or_else(|| Err(Error::ReadError("invalid usize".to_string())))
  }
}

impl<'a> TryFrom<&'a Value> for i8 {
  type Err = Error;

  fn try_from(value: &'a Value) -> Result<Self, Self::Err> {
    value.as_i64().map(|v| Ok(v as i8)).unwrap_or_else(|| Err(Error::ReadError("invalid usize".to_string())))
  }
}

impl<'a> TryFrom<&'a Value> for i16 {
  type Err = Error;

  fn try_from(value: &'a Value) -> Result<Self, Self::Err> {
    value.as_i64().map(|v| Ok(v as i16)).unwrap_or_else(|| Err(Error::ReadError("invalid usize".to_string())))
  }
}

impl<'a> TryFrom<&'a Value> for i32 {
  type Err = Error;

  fn try_from(value: &'a Value) -> Result<Self, Self::Err> {
    value.as_i64().map(|v| Ok(v as i32)).unwrap_or_else(|| Err(Error::ReadError("invalid usize".to_string())))
  }
}

impl<'a> TryFrom<&'a Value> for i64 {
  type Err = Error;

  fn try_from(value: &'a Value) -> Result<Self, Self::Err> {
    value.as_i64().map(|v| Ok(v as i64)).unwrap_or_else(|| Err(Error::ReadError("invalid usize".to_string())))
  }
}

impl<'a> TryFrom<&'a Value> for f32 {
  type Err = Error;

  fn try_from(value: &'a Value) -> Result<Self, Self::Err> {
    value.as_f64().map(|v| Ok(v as f32)).unwrap_or_else(|| Err(Error::ReadError("invalid usize".to_string())))
  }
}

impl<'a> TryFrom<&'a Value> for f64 {
  type Err = Error;

  fn try_from(value: &'a Value) -> Result<Self, Self::Err> {
    value.as_f64().map(|v| Ok(v)).unwrap_or_else(|| Err(Error::ReadError("invalid usize".to_string())))
  }
}

impl<'a> TryFrom<&'a Value> for String {
  type Err = Error;

  fn try_from(value: &'a Value) -> Result<Self, Self::Err> {
    value.as_str().map(|v| Ok(v.to_string())).unwrap_or_else(|| Err(Error::ReadError("invalid usize".to_string())))
  }
}

impl<'a, T: TryFrom<&'a Value, Err=Error>> TryFrom<&'a Value> for Vec<T> {
  type Err = Error;

  fn try_from(value: &'a Value) -> Result<Self, Self::Err> {
    value.as_array().map(|arr| {
      let mut acc: Vec<T> = Vec::with_capacity(arr.len());

      for v in arr.iter() {
        match T::try_from(v) {
          Ok(c) => acc.push(c),
          Err(err) => return Err(err)
        }
      }

      Ok(acc)
    }).unwrap_or_else(|| Err(Error::ReadError("invalid usize".to_string())))
  }
}

impl<'a, T: TryFrom<&'a Value, Err=Error>> TryFrom<&'a Value> for HashMap<String, T> {
  type Err = Error;

  fn try_from(value: &'a Value) -> Result<Self, Self::Err> {
    value.as_object().map(|map| {
      let mut acc: HashMap<String, T> = HashMap::with_capacity(map.len());

      for (k, v) in map.iter() {
        match T::try_from(v) {
          Ok(c) => acc.insert(k.clone(), c),
          Err(err) => return Err(err)
        };
      }

      Ok(acc)
    }).unwrap_or_else(|| Err(Error::ReadError("invalid usize".to_string())))
  }
}

impl<'a> TryFrom<&'a Value> for dsl::Attribute {
  type Err = Error;

  fn try_from(value: &'a Value) -> Result<Self, Self::Err> {
    value.as_object().and_then(|map| {
      map.get("type").and_then(|v| v.as_str()).and_then(|tpe| {
        match tpe.as_ref() {
          "tensor" => {
            match (map.get("base"), map.get("value")) {
              (Some(&Value::String(ref base)), Some(tensor)) => {
                tensor.as_object().and_then(|tmap| {
                  match (tmap.get("dimensions"), tmap.get("values")) {
                    (Some(jdims), Some(jvalues)) => {
                      let r = Vec::<usize>::try_from(jdims).and_then(|dims| {
                        (match base.as_ref() {
                          "bool" => Vec::<bool>::try_from(jvalues).map(|v| dsl::TensorValue::Bool(dsl::DenseTensor::new(dims, v))),
                          "string" => Vec::<String>::try_from(jvalues).map(|v| dsl::TensorValue::String(dsl::DenseTensor::new(dims, v))),
                          "byte" => Vec::<i8>::try_from(jvalues).map(|v| dsl::TensorValue::Byte(dsl::DenseTensor::new(dims, v))),
                          "short" => Vec::<i16>::try_from(jvalues).map(|v| dsl::TensorValue::Short(dsl::DenseTensor::new(dims, v))),
                          "int" => Vec::<i32>::try_from(jvalues).map(|v| dsl::TensorValue::Int(dsl::DenseTensor::new(dims, v))),
                          "long" => Vec::<i64>::try_from(jvalues).map(|v| dsl::TensorValue::Long(dsl::DenseTensor::new(dims, v))),
                          "float" => Vec::<f32>::try_from(jvalues).map(|v| dsl::TensorValue::Float(dsl::DenseTensor::new(dims, v))),
                          "double" => Vec::<f64>::try_from(jvalues).map(|v| dsl::TensorValue::Double(dsl::DenseTensor::new(dims, v))),
                          _ => Err(Error::ReadError("".to_string()))
                        })
                      }).map(|t| dsl::Attribute::Tensor(t));
                      Some(r)
                    },
                    _ => None
                  }
                })
              },
              _ => None
            }
          },
          "list" => {
            match (map.get("base"), map.get("value")) {
              (Some(&Value::String(ref base)), Some(jvalues)) => {
                let r = (match base.as_ref() {
                  "bool" => Vec::<bool>::try_from(jvalues).map(|v| dsl::VectorValue::Bool(v)),
                  "string" => Vec::<String>::try_from(jvalues).map(|v| dsl::VectorValue::String(v)),
                  "byte" => Vec::<i8>::try_from(jvalues).map(|v| dsl::VectorValue::Byte(v)),
                  "short" => Vec::<i16>::try_from(jvalues).map(|v| dsl::VectorValue::Short(v)),
                  "int" => Vec::<i32>::try_from(jvalues).map(|v| dsl::VectorValue::Int(v)),
                  "long" => Vec::<i64>::try_from(jvalues).map(|v| dsl::VectorValue::Long(v)),
                  "float" => Vec::<f32>::try_from(jvalues).map(|v| dsl::VectorValue::Float(v)),
                  "double" => Vec::<f64>::try_from(jvalues).map(|v| dsl::VectorValue::Double(v)),
                  _ => Err(Error::ReadError("".to_string()))
                }).map(|values| dsl::Attribute::Array(values));

                Some(r)
              },
              _ => None
            }
          },
          "bool" => value.as_bool().map(|v| Ok(basic_attribute(dsl::BasicValue::Bool(v)))),
          "string" => value.as_str().map(|v| Ok(basic_attribute(dsl::BasicValue::String(v.to_string())))),
          "byte" => value.as_i64().map(|v| Ok(basic_attribute(dsl::BasicValue::Byte(v as i8)))),
          "short" => value.as_i64().map(|v| Ok(basic_attribute(dsl::BasicValue::Short(v as i16)))),
          "int" => value.as_i64().map(|v| Ok(basic_attribute(dsl::BasicValue::Int(v as i32)))),
          "long" => value.as_i64().map(|v| Ok(basic_attribute(dsl::BasicValue::Long(v)))),
          "float" => value.as_f64().map(|v| Ok(basic_attribute(dsl::BasicValue::Float(v as f32)))),
          "double" => value.as_f64().map(|v| Ok(basic_attribute(dsl::BasicValue::Double(v)))),
          "byte_string" => {
            value.as_str().map(|v64| {
              base64::decode(v64).
                map(|v| basic_attribute(dsl::BasicValue::ByteString(v))).
                map_err(|_| Error::ReadError("Invalid base64 string".to_string()))
            })
          },
          _ => None
        }
      })
    }).unwrap_or_else(|| Err(Error::ReadError("".to_string())))
  }
}

impl<'a> From<&'a dsl::Socket> for Value {
  fn from(value: &'a dsl::Socket) -> Self {
    let mut map = Map::with_capacity(2);
    map.insert(String::from("name"), Value::from(value.name()));
    map.insert(String::from("port"), Value::from(value.port()));

    Value::Object(map)
  }
}

impl From<dsl::Socket> for Value {
  fn from(value: dsl::Socket) -> Self {
    Value::from(&value)
  }
}

impl<'a> TryFrom<&'a Value> for dsl::Socket {
  type Err = Error;

  fn try_from(value: &'a Value) -> Result<Self, Self::Err> {
    value.as_object().and_then(|map| {
      let m_name = map.get("name").and_then(|x| x.as_str());
      let m_port = map.get("port").and_then(|x| x.as_str());

      match (m_name, m_port) {
        (Some(name), Some(port)) => {
          Some(Ok(dsl::Socket::new(String::from(name), String::from(port))))
        },
        _ => None
      }
    }).unwrap_or_else(|| Err(Error::ReadError(String::from("Invalid socket"))))
  }
}

impl<'a> From<&'a dsl::Shape> for Value {
  fn from(value: &'a dsl::Shape) -> Self {
    let inputs = Value::from(value.inputs());
    let outputs = Value::from(value.outputs());
    let mut map = Map::with_capacity(2);
    map.insert(String::from("inputs"), inputs);
    map.insert(String::from("outputs"), outputs);

    Value::Object(map)
  }
}

impl<'a> TryFrom<&'a Value> for dsl::Shape {
  type Err = Error;

  fn try_from(value: &'a Value) -> Result<Self, Self::Err> {
    value.as_object().and_then(|map| {
      match (map.get("inputs"), map.get("outputs")) {
        (Some(jinputs), Some(joutputs)) => {
          let tinputs = Vec::<dsl::Socket>::try_from(jinputs);
          let toutputs = Vec::<dsl::Socket>::try_from(joutputs);

          match (tinputs, toutputs) {
            (Ok(inputs), Ok(outputs)) => Some(Ok(dsl::Shape::new(inputs, outputs))),
            _ => None
          }
        },
        _ => None
      }
    }).unwrap_or_else(|| Err(Error::ReadError(String::from(""))))
  }
}

impl<'a> From<&'a dsl::Model> for Value {
  fn from(value: &'a dsl::Model) -> Self {
    let attrs: Map<String, Value> = value.attributes().
      iter().
      map(|(k, v)| (k.clone(), Value::from(v))).
      collect();

    let mut map = Map::with_capacity(2);
    map.insert(String::from("op"), Value::from(value.op()));
    map.insert(String::from("attributes"), Value::from(attrs));

    Value::Object(map)
  }
}

impl<'a> TryFrom<&'a Value> for dsl::Model {
  type Err = Error;

  fn try_from(value: &'a Value) -> Result<Self, Self::Err> {
    value.as_object().and_then(|map| {
      let m_op = map.get("op").and_then(|x| x.as_str());
      let m_attrs = map.get("attributes");

      match (m_op, m_attrs) {
        (Some(op), Some(jattrs)) => {
          let r = HashMap::<String, dsl::Attribute>::try_from(jattrs).map(|attrs| {
            dsl::Model::new(op.to_string(), attrs)
          });
          Some(r)
        },
        _ => None
      }
    }).unwrap_or_else(|| Err(Error::ReadError(String::from(""))))
  }
}

impl<'a> From<&'a dsl::Node> for Value {
  fn from(value: &'a dsl::Node) -> Self {
    let mut map = Map::with_capacity(2);
    map.insert(String::from("name"), Value::from(value.name()));
    map.insert(String::from("shape"), Value::from(value.shape()));

    Value::Object(map)
  }
}

impl<'a> TryFrom<&'a Value> for dsl::Node {
  type Err = Error;

  fn try_from(value: &'a Value) -> Result<Self, Self::Err> {
    value.as_object().and_then(|map| {
      let m_name = map.get("name").and_then(|x| x.as_str());
      let m_shape = map.get("shape");

      match (m_name, m_shape) {
        (Some(name), Some(jshape)) => {
          let r = dsl::Shape::try_from(jshape).map(|shape| {
            dsl::Node::new(name.to_string(), shape)
          });
          Some(r)
        },
        _ => None
      }
    }).unwrap_or_else(|| Err(Error::ReadError(String::from(""))))
  }
}

impl<'a> From<&'a dsl::Format> for Value {
  fn from(value: &'a dsl::Format) -> Self {
    match *value {
      dsl::Format::Concrete(dsl::ConcreteFormat::Json) => Value::from("json"),
      dsl::Format::Concrete(dsl::ConcreteFormat::Proto) => Value::from("proto"),
      dsl::Format::Mixed => Value::from("mixed")
    }
  }
}

impl<'a> TryFrom<&'a Value> for dsl::Format {
  type Err = Error;

  fn try_from(value: &'a Value) -> Result<Self, Self::Err> {
    match value {
      &Value::String(ref name) => {
        match name.as_ref() {
          "json" => Ok(dsl::Format::Concrete(dsl::ConcreteFormat::Json)),
          "proto" => Ok(dsl::Format::Concrete(dsl::ConcreteFormat::Json)),
          "mixed" => Ok(dsl::Format::Mixed),
          _ => Err(Error::ReadError(String::from("")))
        }
      },
      _ => Err(Error::ReadError(String::from("")))
    }
  }
}

impl<'a> From<&'a dsl::Bundle> for Value {
  fn from(value: &'a dsl::Bundle) -> Self {
    let mut map = Map::with_capacity(4);
    map.insert(String::from("uid"), Value::from(value.uid().to_string()));
    map.insert(String::from("name"), Value::from(value.name()));
    map.insert(String::from("format"), Value::from(value.format()));
    map.insert(String::from("version"), Value::from(value.version().to_string()));

    Value::Object(map)
  }
}

impl<'a> TryFrom<&'a Value> for dsl::Bundle {
  type Err = Error;

  fn try_from(value: &'a Value) -> Result<Self, Self::Err> {
    value.as_object().and_then(|map| {
      let m_uid = map.get("uid").
        and_then(|x| x.as_str()).
        map(|u| {
          Uuid::parse_str(u).map_err(|_| {
            Error::ReadError(String::from("Invalid UUID"))
          })
        });
      let m_name = map.get("name").and_then(|x| x.as_str());
      let m_format = map.get("format").
        map(|x| dsl::Format::try_from(x));
      let m_version = map.get("version").
        and_then(|x| x.as_str()).
        map(|x| {
          Version::parse(x).map_err(|_| {
            Error::ReadError(String::from("Invalid semantic version"))
          })
        });

      match (m_uid, m_name, m_format, m_version) {
        (Some(r_uid), Some(name), Some(r_format), Some(r_version)) => {
          match (r_uid, r_format, r_version) {
            (Ok(uid), Ok(format), Ok(version)) => {
              Some(Ok(dsl::Bundle::new(uid, name.to_string(), format, version)))
            },
            _ => None
          }
        },
        _ => None
      }
    }).unwrap_or_else(|| Err(Error::ReadError(String::from(""))))
  }
}
