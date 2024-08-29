use serde::Serialize;
pub trait Logger {
    fn write<T: Serialize>(&self, msg: &str, data: &T);
}

pub struct DummyLogger {}

impl Logger for DummyLogger {
    fn write<T: Serialize>(&self, _msg: &str, _data: &T) {}
}
