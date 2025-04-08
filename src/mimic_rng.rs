use crate::float::MomTropFloat;

/// Fake random number generator, allows you to specifiy your own random variables
pub struct MimicRng<'a, T> {
    cache: &'a [T],
    counter: usize,
    tokens: Vec<&'a str>,
}

impl<'a, T> MimicRng<'a, T> {
    #[inline]
    pub fn new(cache: &'a [T]) -> Self {
        Self {
            cache,
            counter: 0,
            tokens: Vec::new(),
        }
    }

    #[inline]
    pub fn get_random_number(&mut self, debug_token: Option<&'a str>) -> &'a T {
        let random_number = &self.cache[self.counter];
        self.counter += 1;
        if let Some(token) = debug_token {
            self.tokens.push(token);
        }
        random_number
    }
}

impl<T: MomTropFloat> MimicRng<'_, T> {
    pub fn zero(&self) -> T {
        self.cache[0].zero()
    }

    pub fn one(&self) -> T {
        self.cache[0].one()
    }
}
