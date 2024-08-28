use serde::Serialize;
pub trait Logger {
    fn write<T: Serialize>(&self, msg: &str, data: &T);
}
