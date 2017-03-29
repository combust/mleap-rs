use std::result;
use std::io;
use std::path;
use std::fs;
use std::any::{Any, TypeId};
use std::collections::HashMap;
use serde_json::{self, Value};
use super::json::{self, TryFrom};
use super::dsl;

#[derive(Debug)]
pub enum Error {
  IoError(io::Error),
  SerdeJsonError(serde_json::Error),
  JsonError(json::Error),
  DowncastError(String),
  InvalidOp(String),
  InvalidModel(String)
}
pub type Result<T> = result::Result<T, Error>;

pub trait OpNode where Self: 'static {
  fn type_id(&self) -> TypeId { TypeId::of::<Self>() }
  fn op(&self) -> &'static str;
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

  fn load_model(&self,
                model: &dsl::Model,
                ctx: &Context<Self::Node>) -> Result<Box<Any>>;

  fn load(&self,
          node: &dsl::Node,
          model: Box<Any>,
          ctx: &Context<Self::Node>) -> Result<Self::Node>;
}

pub struct Registry<'a, Node: 'a> {
  ops: HashMap<String, &'a Op<Node=Node>>,
  type_ops: HashMap<TypeId, &'a Op<Node=Node>>
}

pub trait Builder {
  fn try_next(&self, name: &str) -> Result<Box<Builder>>;

  fn write(&self, name: &str) -> Result<Box<io::Write>>;
  fn read(&self, name: &str) -> Result<Box<io::Read>>;

  fn write_buf(&self, name: &str) -> Result<io::BufWriter<Box<io::Write>>> {
    self.write(name).map(io::BufWriter::new)
  }

  fn read_buf(&self, name: &str) -> Result<io::BufReader<Box<io::Read>>> {
    self.read(name).map(io::BufReader::new)
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

fn from_json_result<T>(result: result::Result<T, json::Error>) -> Result<T> {
  result.map_err(|e| {
    Error::JsonError(e)
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

  pub fn try_op_for_name(&self, name: &str) -> Result<&& Op<Node=Node>> {
    self.get_op_for_name(name).map(|n| Ok(n)).unwrap_or_else(|| Err(Error::InvalidOp(format!("Op {} does not exist", name))))
  }

  pub fn try_op_for_node(&self, node: &Node) -> Result<&& Op<Node=Node>> {
    self.get_op_for_node(node).map(|n| Ok(n)).unwrap_or_else(|| Err(Error::InvalidOp(format!("Op {} does not exist", node.op()))))
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

  pub fn try_next(&self, name: &str) -> Result<Context<'a, Node>> {
    self.builder.try_next(name).map(|b| Context { builder: b, registry: self.registry })
  }
}

impl FileBuilder {
  pub fn try_new<P: AsRef<path::Path>>(path: P) -> Result<FileBuilder> {
    let r = fs::create_dir_all(&path).map(|_| {
      FileBuilder {
        path: path.as_ref().to_path_buf()
      }
    });

    from_io_result(r)
  }
}

impl Builder for FileBuilder {
  fn try_next(&self, name: &str) -> Result<Box<Builder>> {
    FileBuilder::try_new(self.path.join(name)).map(|x| Box::new(x) as Box<Builder>)
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
            self.try_next("root").
              and_then(|ctx| {
                ctx.write_node_and_model(root)
              })
          })
      })
  }

  pub fn write_node_and_model(&self, node: &Node) -> Result<()> {
    self.registry.try_op_for_node(node).and_then(|op| {
      self.write_node(node, *op).and_then(|_| {
        let model = op.model(node);
        self.write_model(model, *op)
      })
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

impl<'a, Node: OpNode + 'a> Context<'a, Node> {
  pub fn read_bundle(&self) -> Result<(dsl::Bundle, Node)> {
    self.read_dsl_bundle().and_then(|bundle| {
      self.try_next("root").and_then(|ctx| {
        ctx.read_node().map(|node| (bundle, node))
      })
    })
  }

  pub fn read_node(&self) -> Result<Node> {
    self.read_dsl_model().and_then(|d_model| {
      self.registry.try_op_for_name(d_model.op()).and_then(|op| {
        op.load_model(&d_model, self).and_then(|model| {
          self.read_dsl_node().and_then(|d_node| {
            op.load(&d_node, model, self)
          })
        })
      })
    })
  }

  pub fn read_dsl_bundle(&self) -> Result<dsl::Bundle> {
    self.builder.read_buf("bundle.json").and_then(|ref mut r| {
      from_serde_json_result(serde_json::from_reader(r)).and_then(|json: Value| {
        from_json_result(dsl::Bundle::try_from(&json))
      })
    })
  }

  pub fn read_dsl_node(&self) -> Result<dsl::Node> {
    self.builder.read_buf("node.json").and_then(|r| {
      from_serde_json_result(serde_json::from_reader(r)).and_then(|json: Value| {
        from_json_result(dsl::Node::try_from(&json))
      })
    })
  }

  pub fn read_dsl_model(&self) -> Result<dsl::Model> {
    self.builder.read_buf("model.json").and_then(|r| {
      from_serde_json_result(serde_json::from_reader(r)).and_then(|json: Value| {
        from_json_result(dsl::Model::try_from(&json))
      })
    })
  }
}
