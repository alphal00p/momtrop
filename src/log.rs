use serde::Serialize;
pub trait Logger {
    fn write<T: Serialize>(&self, msg: &str, data: &T);
}

impl Logger for () {
    fn write<T: Serialize>(&self, _msg: &str, _data: &T) {}
}
