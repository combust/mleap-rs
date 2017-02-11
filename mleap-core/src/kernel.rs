pub struct Kernel {
  name: String
}

impl Kernel {
  pub fn name(&self) -> &str { &self.name }
}
