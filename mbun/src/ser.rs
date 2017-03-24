use std::result;
use std::io;
use std::path;
use std::fs;
use std::any::{Any, TypeId};
use std::collections::HashMap;
use serde_json::{self, Value};
use json;
use dsl;

#[derive(Debug)]
pub struct DowncastError {
  msg: String
}

#[derive(Debug)]
pub enum Error {
  IoError(io::Error),
  SerdeJsonError(serde_json::Error),
  JsonError(json::Error),
  DowncastError(DowncastError),
  InvalidOp(String)
}
pub type Result<T> = result::Result<T, Error>;

pub trait AnyExt {
  fn try_downcast_ref<T: Any>(&self) -> result::Result<&T, DowncastError>;
}

impl AnyExt for Any + 'static {
  fn try_downcast_ref<T: Any>(&self) -> result::Result<&T, DowncastError> {
    self.downcast_ref().map(|dr| Ok(dr)).unwrap_or_else(|| Err(DowncastError { msg: "".to_string() }))
  }
}

pub trait OpNode where Self: 'static {
  fn type_id(&self) -> TypeId { TypeId::of::<Self>() }
}

pub trait Op {
  type Node: OpNode;

  fn type_id(&self) -> TypeId;
  fn op(&self) -> &'static str;

  fn name<'a>(&self, node: &'a Self::Node) -> &'a str;
  fn model<'a>(&self, node: &'a Self::Node) -> &'a Any;
  fn node(&self, node: &Self::Node, ctx: &Context<Self::Node>) -> dsl::Node;

  fn store_model(&self,
                 obj: &Any,
                 model: &mut dsl::Model,
                 ctx: &Context<Self::Node>) -> Result<()>;

  fn read(&self,
          node: &dsl::Node,
          model: &dsl::Model,
          ctx: &Context<Self::Node>) -> Result<Self::Node>;
}

pub struct Registry<'a, Node: 'a> {
  ops: HashMap<String, &'a Op<Node=Node>>,
  type_ops: HashMap<TypeId, &'a Op<Node=Node>>
}

pub trait Builder {
  fn next(&self, name: &str) -> Result<Box<Builder>>;

  fn write(&self, name: &str) -> Result<Box<io::Write>>;
  fn read(&self, name: &str) -> Result<Box<io::Read>>;

  fn write_buf(&self, name: &str) -> Result<io::BufWriter<Box<io::Write>>> {
    self.write(name).map(io::BufWriter::new)
  }
}

pub struct Context<'a, Node: 'a> {
  builder: Box<Builder>,
  registry: &'a Registry<'a, Node>
}

pub struct FileBuilder {
  path: path::PathBuf
}

fn from_io_result<T>(result: io::Result<T>) -> Result<T> {
  result.map_err(|e| {
    Error::IoError(e)
  })
}

fn from_serde_json_result<T>(result: serde_json::Result<T>) -> Result<T> {
  result.map_err(|e| {
    Error::SerdeJsonError(e)
  })
}

impl<'a, Node: OpNode + 'a> Registry<'a, Node> {
  pub fn new() -> Registry<'a, Node> {
    Registry {
      ops: HashMap::new(),
      type_ops: HashMap::new()
    }
  }

  pub fn insert_op<O: Op<Node=Node>>(&mut self, op: &'a O) -> &mut Self {
    self.ops.insert(op.op().to_string(), op);
    self.type_ops.insert(op.type_id(), op);
    self
  }

  pub fn get_op_for_name(&self, name: &str) -> Option<&& Op<Node=Node>> {
    self.ops.get(name)
  }

  pub fn get_op_for_node(&self, node: &Node) -> Option<&& Op<Node=Node>> {
    self.type_ops.get(&node.type_id())
  }

  pub fn try_op_for_node(&self, node: &Node) -> Result<&& Op<Node=Node>> {
    self.get_op_for_node(node).map(|n| Ok(n)).unwrap_or_else(|| Err(Error::InvalidOp("".to_string())))
  }
}

impl<'a, Node: Sized + 'a> Context<'a, Node> {
  pub fn new(builder: Box<Builder>,
             registry: &'a Registry<'a, Node>) -> Context<'a, Node> {
    Context {
      builder: builder,
      registry: registry
    }
  }

  pub fn builder(&self) -> &Builder { self.builder.as_ref() }
  pub fn registry(&self) -> &Registry<Node> { &self.registry }

  pub fn next(&self, name: &str) -> Result<Context<'a, Node>> {
    self.builder.next(name).map(|b| Context { builder: b, registry: self.registry })
  }
}

impl FileBuilder {
  pub fn new<P: AsRef<path::Path>>(path: P) -> Result<FileBuilder> {
    let r = fs::create_dir_all(&path).map(|_| {
      FileBuilder {
        path: path.as_ref().to_path_buf()
      }
    });

    from_io_result(r)
  }
}

impl Builder for FileBuilder {
  fn next(&self, name: &str) -> Result<Box<Builder>> {
    FileBuilder::new(self.path.join(name)).map(|x| Box::new(x) as Box<Builder>)
  }

  fn write(&self, name: &str) -> Result<Box<io::Write>> {
    from_io_result(fs::OpenOptions::new().
                   write(true).
                   create(true).
                   open(self.path.join(name))).map(|x| Box::new(x) as Box<io::Write>)
  }

  fn read(&self, name: &str) -> Result<Box<io::Read>> {
    from_io_result(fs::File::open(self.path.join(name))).map(|x| Box::new(x) as Box<io::Read>)
  }
}

impl<'a, Node: OpNode + 'a> Context<'a, Node> {
  pub fn write_bundle(&self, bundle: &dsl::Bundle, root: &Node) -> Result<()> {
    self.builder.write_buf("bundle.json").
      and_then(|ref mut out| {
        let json = Value::from(bundle);
        from_serde_json_result(serde_json::to_writer_pretty(out, &json)).
          and_then(|_| {
            self.create_node(root).
              and_then(|(ctx, op)| {
                ctx.write_node(root, op).and_then(|_| {
                  let model = op.model(root);
                  ctx.write_model(model, op)
                })
              })
          })
      })
  }

  fn create_node(&self, node: &Node) -> Result<(Context<'a, Node>, &'a Op<Node=Node>)> {
    self.registry.try_op_for_node(node).and_then(|op| {
      let name = op.name(node);
      let dir = format!("{}.node", name);

      self.next(dir.as_str()).map(|ctx| (ctx, *op))
    })
  }

  fn write_node(&self, node: &Node, op: &Op<Node=Node>) -> Result<()> {
    self.builder.write_buf("node.json").
      and_then(|ref mut out| {
        let dsl_node = op.node(node, self);
        let json = Value::from(&dsl_node);

        from_serde_json_result(serde_json::to_writer_pretty(out, &json))
      })
  }

  fn write_model(&self, obj: &Any, op: &Op<Node=Node>) -> Result<()> {
    self.builder.write_buf("model.json").
      and_then(|ref mut out| {
        let mut model = dsl::Model::new(op.op().to_string(), HashMap::new());
        op.store_model(obj, &mut model, self).and_then(|_| {
          let json = Value::from(&model);

          from_serde_json_result(serde_json::to_writer_pretty(out, &json))
        })
      })
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use uuid::Uuid;
  use semver::Version;

  pub trait Transformer: OpNode {
    fn name(&self) -> &str;
  }
  pub struct LinearRegression {
    intercept: f32
  }
  pub struct LinearRegressionOp { }
  impl OpNode for LinearRegression { }

  impl Transformer for LinearRegression {
    fn name(&self) -> &str { "hello" }
  }

  impl OpNode for Box<Transformer> {
    fn type_id(&self) -> TypeId { Transformer::type_id(self.as_ref()) }
  }
  impl Op for LinearRegressionOp {
    type Node = Box<Transformer>;

    fn type_id(&self) -> TypeId { TypeId::of::<LinearRegression>() }
    fn op(&self) -> &'static str { "my_node" }

    fn name<'a>(&self, node: &'a Self::Node) -> &'a str { node.name() }

    fn model<'a>(&self, node: &'a Self::Node) -> &'a Any { node as &Any }

    fn store_model(&self,
                   obj: &Any,
                   model: &mut dsl::Model,
                   ctx: &Context<Self::Node>) -> Result<()> {
      obj.downcast_ref::<LinearRegression>().map(|lr| {
        model.with_attr("intercept", dsl::Attribute::Basic(dsl::BasicValue::Float(lr.intercept)));
        Ok(())
      }).unwrap_or_else(|| Err(Error::InvalidOp("".to_string())))
    }

    fn node(&self, node: &Self::Node, ctx: &Context<Self::Node>) -> dsl::Node {
      let shape = dsl::Shape::new(vec![dsl::Socket::new(String::from("MyInput"), String::from("input"))],
      vec![dsl::Socket::new(String::from("MyOutput"), String::from("output"))]);

      dsl::Node::new("hello".to_string(), shape)
    }

    fn read(&self,
            node: &dsl::Node,
            model: &dsl::Model,
            ctx: &Context<Self::Node>) -> Result<Self::Node> {
      Err(Error::InvalidOp("".to_string()))
    }
  }
  const LINEAR_REGRESSION_OP: &LinearRegressionOp = &LinearRegressionOp { };

  #[test]
  fn bundle_serializer_test() {
    let mut registry: Registry<Box<Transformer>> = Registry::new();
    registry.insert_op(LINEAR_REGRESSION_OP);
    let builder = FileBuilder::new("/tmp/test-rs").unwrap();
    let context = Context::new(Box::new(builder), &registry);

    let bundle = dsl::Bundle::new(Uuid::new_v4(),
    "hello".to_string(),
    dsl::Format::Mixed,
    Version::parse("0.6.0-SNAPSHOT").unwrap());
    let root = Box::new(LinearRegression { intercept: 32.0 }) as Box<Transformer>;

    context.write_bundle(&bundle, &root).unwrap();
  }
}
