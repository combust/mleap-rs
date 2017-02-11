use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Socket {
  name: String,
  port: String
}

impl Socket {
  pub fn name(&self) -> &str { &self.name }
  pub fn port(&self) -> &str { &self.port }
}

pub struct NodeShape {
  inputs: HashSet<Socket>,
  outputs: HashSet<Socket>
}

impl NodeShape {
  pub fn new() -> NodeShape {
    NodeShape {
      inputs: HashSet::new(),
      outputs: HashSet::new()
    }
  }

  pub fn attach(&mut self, other: &NodeShape) -> &NodeShape {
    self.inputs = self.inputs.difference(&other.outputs).cloned().collect();
    self
  }

  pub fn inputs(&self) -> &HashSet<Socket> { &self.inputs }
  pub fn outputs(&self) -> &HashSet<Socket> { &self.outputs }

  pub fn input(&self, name: &str) -> &Socket {
    self.get_input(name).expect(format!("no socket for name: {}", name).as_str())
  }

  pub fn get_input(&self, name: &str) -> Option<&Socket> {
    self.inputs().iter().find(|x| x.name() == name)
  }

  pub fn input_for_port(&self, port: &str) -> &Socket {
    self.get_input_for_port(port).expect(format!("no socket for port: {}", port).as_str())
  }

  pub fn get_input_for_port(&self, port: &str) -> Option<&Socket> {
    self.inputs().iter().find(|x| x.port() == port)
  }

  pub fn output(&self, name: &str) -> &Socket {
    self.get_output(name).expect(format!("no socket for name: {}", name).as_str())
  }

  pub fn get_output(&self, name: &str) -> Option<&Socket> {
    self.outputs().iter().find(|x| x.name() == name)
  }

  pub fn output_for_port(&self, port: &str) -> &Socket {
    self.get_output_for_port(port).expect(format!("no socket for port: {}", port).as_str())
  }

  pub fn get_output_for_port(&self, port: &str) -> Option<&Socket> {
    self.outputs().iter().find(|x| x.port() == port)
  }
}

pub trait Node {
  fn name(&self) -> &str;
  fn shape(&self) -> &NodeShape;
}

pub struct Graph {
  name: String,
  shape: NodeShape,
  nodes: Vec<Box<Node>>
}

impl Graph {
  pub fn new(name: String) -> Graph {
    Graph {
      name: name,
      shape: NodeShape::new(),
      nodes: Vec::new()
    }
  }

  pub fn nodes(&self) -> &Vec<Box<Node>> { &self.nodes }

  pub fn add_node(&mut self, node: Box<Node>) -> &Graph {
    self
  }
}

impl Node for Graph {
  fn name(&self) -> &str { &self.name }
  fn shape(&self) -> &NodeShape { &self.shape }
}
