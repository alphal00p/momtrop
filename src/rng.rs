pub trait MyRng<'a, T: Copy> {
    fn get_random_number(&mut self, debug_token: Option<&'a str>) -> T;
}

/// Fake random number generator, allows you to specifiy your own random variables
pub struct MimicRng<'a, T> {
    cache: &'a [T],
    counter: usize,
    tokens: Vec<&'a str>,
}

impl<'a, T: Copy> MyRng<'a, T> for MimicRng<'a, T> {
    fn get_random_number(&mut self, debug_token: Option<&'a str>) -> T {
        let random_number = self.cache[self.counter];
        self.counter += 1;
        if let Some(token) = debug_token {
            self.tokens.push(token);
        }
        random_number
    }
}

impl<'a, T: Copy> MimicRng<'a, T> {
    pub fn new(cache: &'a [T]) -> Self {
        Self {
            cache,
            counter: 0,
            tokens: Vec::new(),
        }
    }
}

pub struct FastrandRngWrapper<'a> {
    rng: fastrand::Rng,
    tokens: Vec<&'a str>,
}

impl<'a> MyRng<'a, f64> for FastrandRngWrapper<'a> {
    fn get_random_number(&mut self, debug_token: Option<&'a str>) -> f64 {
        if let Some(token) = debug_token {
            self.tokens.push(token);
        }
        self.rng.f64()
    }
}

impl<'a> FastrandRngWrapper<'a> {
    pub fn new(seed: u64) -> Self {
        FastrandRngWrapper {
            rng: fastrand::Rng::with_seed(seed),
            tokens: Vec::new(),
        }
    }
}
