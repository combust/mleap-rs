use std::any::*;

use bundle::tform::{self, DefaultNode};
use bundle::ser::*;
use bundle::frame;
use bundle::dsl;

pub const OP: &VectorAssemblerOp = &VectorAssemblerOp { };

pub struct VectorAssemblerModel { }

pub struct VectorAssembler {
  name: String,
  input_cols: Vec<String>,
  output_col: String
}

pub struct VectorAssemblerOp { }

impl OpNode for VectorAssembler { }

impl VectorAssemblerModel {
  pub fn try_assemble(col_names: &[String],
                      frame: &frame::LeapFrame) -> frame::Result<frame::ColData> {
    match frame.try_cols(col_names) {
      Ok(cols) => {
        let mut t_size: usize = 0;
        for col in cols.iter() {
          let m_size: Option<usize> = match col.data() {
            &frame::ColData::Bool(_) => Some(1),
            &frame::ColData::Byte(_) => Some(1),
            &frame::ColData::Short(_) => Some(1),
            &frame::ColData::Int(_) => Some(1),
            &frame::ColData::Long(_) => Some(1),
            &frame::ColData::Float(_) => Some(1),
            &frame::ColData::Double(_) => Some(1),

            &frame::ColData::BoolTensor(ref v) => v.first().and_then(|f| f.dimensions().first().map(|x| *x)),
            &frame::ColData::ByteTensor(ref v) => v.first().and_then(|f| f.dimensions().first().map(|x| *x)),
            &frame::ColData::ShortTensor(ref v) => v.first().and_then(|f| f.dimensions().first().map(|x| *x)),
            &frame::ColData::IntTensor(ref v) => v.first().and_then(|f| f.dimensions().first().map(|x| *x)),
            &frame::ColData::LongTensor(ref v) => v.first().and_then(|f| f.dimensions().first().map(|x| *x)),
            &frame::ColData::FloatTensor(ref v) => v.first().and_then(|f| f.dimensions().first().map(|x| *x)),
            &frame::ColData::DoubleTensor(ref v) => v.first().and_then(|f| f.dimensions().first().map(|x| *x)),

            _ => None
          };

          match m_size {
            Some(size) => t_size += size,
            None => return Err(frame::Error::TransformError(String::from("")))
          }
        }

        let mut vs: Vec<Vec<f64>> = (0..frame.size()).map(|_| {
          Vec::with_capacity(t_size)
        }).collect();

        for col in cols.iter() {
          match col.data() {
            &frame::ColData::Bool(ref v) => Self::assemble_scalar(&mut vs, &v, |x| { if x { 1.0 } else { 0.0 } }),
            &frame::ColData::Byte(ref v) => Self::assemble_scalar(&mut vs, &v, f64::from),
            &frame::ColData::Short(ref v) => Self::assemble_scalar(&mut vs, &v, f64::from),
            &frame::ColData::Int(ref v) => Self::assemble_scalar(&mut vs, &v, f64::from),
            &frame::ColData::Long(ref v) => Self::assemble_scalar(&mut vs, &v, |x| { x as f64 }),
            &frame::ColData::Float(ref v) => Self::assemble_scalar(&mut vs, &v, f64::from),
            &frame::ColData::Double(ref v) => Self::assemble_scalar(&mut vs, &v, f64::from),

            &frame::ColData::BoolTensor(ref v) => Self::assemble_tensor(&mut vs, &v, |x| { if x { 1.0 } else { 0.0 } }),
            &frame::ColData::ByteTensor(ref v) => Self::assemble_tensor(&mut vs, &v, f64::from),
            &frame::ColData::ShortTensor(ref v) => Self::assemble_tensor(&mut vs, &v, f64::from),
            &frame::ColData::IntTensor(ref v) => Self::assemble_tensor(&mut vs, &v, f64::from),
            &frame::ColData::LongTensor(ref v) => Self::assemble_tensor(&mut vs, &v, |x| { x as f64 }),
            &frame::ColData::FloatTensor(ref v) => Self::assemble_tensor(&mut vs, &v, f64::from),
            &frame::ColData::DoubleTensor(ref v) => Self::assemble_tensor(&mut vs, &v, f64::from),

            _ => { } // do nothing
          }
        }

        // TODO: this to_vec is very inefficient
        let tensors = vs.drain(0..).map(|v| dsl::DenseTensor::new(vec![t_size], v)).collect();
        Ok(frame::ColData::DoubleTensor(tensors))
      },
      Err(err) => return Err(err)
    }
  }

  fn assemble_scalar<T: Copy, F>(vs: &mut [Vec<f64>], vi: &[T], f: F)
    where F: Fn(T) -> f64 {
      for (a, b) in vs.iter_mut().zip(vi.iter().map(|x| f(*x))) {
        a.push(b)
      }
    }

  fn assemble_tensor<T: Copy, F>(vs: &mut [Vec<f64>], vt: &[dsl::DenseTensor<T>], f: F)
    where F: Fn(T) -> f64 {
      for (a, b) in vs.iter_mut().zip(vt.iter()) {
        for t in b.values().iter() {
          a.push(f(*t))
        }
      }
    }
}

impl frame::Transformer for VectorAssembler {
  fn transform(&self, frame: &mut frame::LeapFrame) -> frame::Result<()> {
    VectorAssemblerModel::try_assemble(&self.input_cols, frame).and_then(|cd| {
      frame.try_with_col(frame::Col::new(self.output_col.clone(), cd))
    }).map(|_| ())
  }
}

const MODEL: &VectorAssemblerModel = &VectorAssemblerModel { };
impl DefaultNode for VectorAssembler {
  fn name(&self) -> &str { &self.name }
  fn model(&self) -> &Any { MODEL as &Any }

  fn create_shape(&self) -> dsl::Shape {
    let inputs = self.input_cols.iter().enumerate().map(|(i, name)| {
      let port = format!("input{}", i);
      dsl::Socket::new(name.to_string(), port)
    }).collect();
    let outputs = vec![dsl::Socket::new(self.output_col.clone(), String::from("output"))];

    dsl::Shape::new(inputs, outputs)
  }
}

impl Op for VectorAssemblerOp {
  type Node = Box<tform::DefaultNode>;

  fn type_id(&self) -> TypeId { TypeId::of::<VectorAssembler>() }
  fn op(&self) -> &'static str { "vector_assembler" }

  fn name<'a>(&self, node: &'a Self::Node) -> &'a str { node.name() }

  fn model<'a>(&self, node: &'a Self::Node) -> &'a Any { DefaultNode::model(node.as_ref()) }

  fn store_model(&self,
                 _obj: &Any,
                 _model: &mut dsl::Model,
                 _ctx: &Context<Self::Node>) -> Result<()> {
    Ok(())
  }

  fn load_model(&self,
                _model: &dsl::Model,
                _ctx: &Context<Self::Node>) -> Result<Box<Any>> {
    Ok(Box::new(VectorAssemblerModel { }))
  }

  fn node(&self, node: &Self::Node, _ctx: &Context<Self::Node>) -> dsl::Node {
    node.create_node()
  }

  fn load(&self,
          node: &dsl::Node,
          _model: Box<Any>,
          _ctx: &Context<Self::Node>) -> Result<Self::Node> {
    node.shape().get_output("output").map(|o| {
      let inputs = node.shape().inputs().iter().
        map(|s| s.name().to_string()).collect();
      VectorAssembler {
        name: node.name().to_string(),
        input_cols: inputs,
        output_col: o.name().to_string()
      }
    }).map(|x| Ok(Box::new(x) as Box<tform::DefaultNode>)).
    unwrap_or_else(|| Err(Error::InvalidOp(String::from("Need output socket"))))
  }
}

