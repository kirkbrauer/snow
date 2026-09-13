//! Owned noise streams for repeatable device emulation.

use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// ChaCha8 streams are independent of host thread scheduling. Unseeded callers
/// preserve the interactive emulator's existing entropy-backed behavior.
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct Noise {
    stream: Option<ChaCha8Rng>,
    draws: u64,
}

impl Noise {
    pub fn seeded(seed: u64) -> Self {
        Self {
            stream: Some(ChaCha8Rng::seed_from_u64(seed)),
            draws: 0,
        }
    }

    pub fn byte(&mut self) -> u8 {
        self.draws = self.draws.wrapping_add(1);
        match self.stream.as_mut() {
            Some(stream) => stream.next_u32() as u8,
            None => rand::rng().random(),
        }
    }

    pub fn bit(&mut self) -> bool {
        self.byte() & 1 != 0
    }
}

#[cfg(test)]
mod tests {
    use super::Noise;

    #[test]
    fn independent_streams() {
        let mut a = Noise::seeded(42);
        let mut b = Noise::seeded(42);
        let mut other = Noise::seeded(43);
        for _ in 0..1000 {
            other.byte();
            assert_eq!(a.byte(), b.byte());
        }
    }
}
