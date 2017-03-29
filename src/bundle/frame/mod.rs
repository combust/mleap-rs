use std::result;
use std::collections::HashMap;

use bundle::dsl::DenseTensor;

#[derive(Debug)]
pub enum Error {
  TransformError(String),
  InvalidType(String),
  ColumnAlreadyExists(String),
  NoSuchColumn(String)
}
pub type Result<T> = result::Result<T, Error>;

pub trait Transformer {
  fn transform(&self, frame: &mut LeapFrame) -> Result<()>;
}

pub enum ColData {
  Bool(Vec<bool>),
  String(Vec<String>),
  Byte(Vec<i8>),
  Short(Vec<i16>),
  Int(Vec<i32>),
  Long(Vec<i64>),
  Float(Vec<f32>),
  Double(Vec<f64>),
  ByteString(Vec<Vec<u8>>),

  BoolVector(Vec<Vec<bool>>),
  StringVector(Vec<Vec<String>>),
  ByteVector(Vec<Vec<i8>>),
  ShortVector(Vec<Vec<i16>>),
  IntVector(Vec<Vec<i32>>),
  LongVector(Vec<Vec<i64>>),
  FloatVector(Vec<Vec<f32>>),
  DoubleVector(Vec<Vec<f64>>),
  ByteStringVector(Vec<Vec<Vec<u8>>>),

  BoolTensor(Vec<DenseTensor<bool>>),
  StringTensor(Vec<DenseTensor<String>>),
  ByteTensor(Vec<DenseTensor<i8>>),
  ShortTensor(Vec<DenseTensor<i16>>),
  IntTensor(Vec<DenseTensor<i32>>),
  LongTensor(Vec<DenseTensor<i64>>),
  FloatTensor(Vec<DenseTensor<f32>>),
  DoubleTensor(Vec<DenseTensor<f64>>),
  ByteStringTensor(Vec<DenseTensor<Vec<u8>>>)
}

pub struct Col {
  name: String,
  data: ColData
}

pub struct LeapFrame {
  size: usize,
  cols: Vec<Col>,
  col_indices_by_name: HashMap<String, usize>
}

impl Col {
  pub fn new(name: String, data: ColData) -> Col {
    Col {
      name: name,
      data: data
    }
  }

  pub fn from_doubles(name: String, v: Vec<f64>) -> Col {
    Col {
      name: name,
      data: ColData::Double(v)
    }
  }

  pub fn from_double_tensors(name: String, v: Vec<DenseTensor<f64>>) -> Col {
    Col {
      name: name,
      data: ColData::DoubleTensor(v)
    }
  }

  pub fn from_strings(name: String, v: Vec<String>) -> Col {
    Col {
      name: name,
      data: ColData::String(v)
    }
  }

  pub fn from_ints(name: String, v: Vec<i32>) -> Col {
    Col {
      name: name,
      data: ColData::Int(v)
    }
  }

  pub fn from_long_tensors(name: String, v: Vec<DenseTensor<i64>>) -> Col {
    Col {
      name: name,
      data: ColData::LongTensor(v)
    }
  }

  pub fn name(&self) -> &str { &self.name }
  pub fn data(&self) -> &ColData { &self.data }

  pub fn get_doubles(&self) -> Option<&[f64]> {
    match self.data {
      ColData::Double(ref v) => Some(v),
      _ => None
    }
  }
  pub fn try_doubles(&self) -> Result<&[f64]> { Self::option_to_result(self.get_doubles()) }

  pub fn get_ints(&self) -> Option<&[i32]> {
    match self.data {
      ColData::Int(ref v) => Some(v),
      _ => None
    }
  }
  pub fn try_ints(&self) -> Result<&[i32]> { Self::option_to_result(self.get_ints()) }

  pub fn get_double_tensors(&self) -> Option<&[DenseTensor<f64>]> {
    match self.data {
      ColData::DoubleTensor(ref v) => Some(v),
      _ => None
    }
  }
  pub fn try_double_tensors(&self) -> Result<&[DenseTensor<f64>]> { Self::option_to_result(self.get_double_tensors()) }

  pub fn get_strings(&self) -> Option<&[String]> {
    match self.data {
      ColData::String(ref v) => Some(v),
      _ => None
    }
  }
  pub fn try_strings(&self) -> Result<&[String]> { Self::option_to_result(self.get_strings()) }

  fn option_to_result<T>(option: Option<T>) -> Result<T> {
    option.map(|x| Ok(x)).unwrap_or_else(|| Err(Error::InvalidType(String::from(""))))
  }
}

impl LeapFrame {
  pub fn with_size(size: usize) -> LeapFrame {
    LeapFrame {
      size: size,
      cols: Vec::new(),
      col_indices_by_name: HashMap::new()
    }
  }

  pub fn size(&self) -> usize { self.size }
  pub fn cols(&self) -> &[Col] { &self.cols }

  pub fn try_with_doubles(&mut self, name: String, v: Vec<f64>) -> Result<&mut Self> { self.try_with_col(Col::from_doubles(name, v)) }
  pub fn try_with_double_tensors(&mut self, name: String, v: Vec<DenseTensor<f64>>) -> Result<&mut Self> { self.try_with_col(Col::from_double_tensors(name, v)) }
  pub fn try_with_strings(&mut self, name: String, v: Vec<String>) -> Result<&mut Self> { self.try_with_col(Col::from_strings(name, v)) }
  pub fn try_with_ints(&mut self, name: String, v: Vec<i32>) -> Result<&mut Self> { self.try_with_col(Col::from_ints(name, v)) }

  pub fn try_with_col(&mut self, col: Col) -> Result<&mut Self> {
    if self.col_indices_by_name.contains_key(col.name()) {
      Err(Error::ColumnAlreadyExists(String::from(col.name())))
    } else {
      self.col_indices_by_name.insert(col.name().to_string(), self.cols.len());
      self.cols.push(col);

      Ok(self)
    }
  }

  pub fn get_col(&self, name: &str) -> Option<&Col> {
    self.col_indices_by_name.get(name).map(|i| &self.cols[*i])
  }

  pub fn try_col(&self, name: &str) -> Result<&Col> {
    self.get_col(name).map(|x| Ok(x)).unwrap_or_else(|| Err(Error::NoSuchColumn(String::from(name))))
  }

  pub fn try_cols(&self, names: &[String]) -> Result<Vec<&Col>> {
    let mut cols = Vec::with_capacity(names.len());

    for name in names.iter() {
      match self.try_col(name) {
        Ok(col) => cols.push(col),
        Err(err) => return Err(err)
      }
    }

    Ok(cols)
  }

  pub fn get_doubles(&self, name: &str) -> Option<&[f64]> { self.get_col(name).and_then(|c| c.get_doubles()) }
  pub fn try_doubles(&self, name: &str) -> Result<&[f64]> { self.try_col(name).and_then(|c| c.try_doubles()) }

  pub fn get_ints(&self, name: &str) -> Option<&[i32]> { self.get_col(name).and_then(|c| c.get_ints()) }
  pub fn try_ints(&self, name: &str) -> Result<&[i32]> { self.try_col(name).and_then(|c| c.try_ints()) }

  pub fn get_double_tensors(&self, name: &str) -> Option<&[DenseTensor<f64>]> { self.get_col(name).and_then(|c| c.get_double_tensors()) }
  pub fn try_double_tensors(&self, name: &str) -> Result<&[DenseTensor<f64>]> { self.try_col(name).and_then(|c| c.try_double_tensors()) }

  pub fn get_strings(&self, name: &str) -> Option<&[String]> { self.get_col(name).and_then(|c| c.get_strings()) }
  pub fn try_strings(&self, name: &str) -> Result<&[String]> { self.try_col(name).and_then(|c| c.try_strings()) }
}
