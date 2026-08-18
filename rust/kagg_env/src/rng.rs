//! CPython `random.Random` compatibility for deterministic environment events.

#[derive(Clone, Debug)]
pub struct PyRandom {
    state: [u32; 624],
    index: usize,
}

impl PyRandom {
    pub fn seed_u64(seed: u64) -> Self {
        let mut key = vec![seed as u32];
        let high = (seed >> 32) as u32;
        if high != 0 {
            key.push(high);
        }
        let mut rng = Self {
            state: [0; 624],
            index: 624,
        };
        rng.init_by_array(&key);
        rng
    }

    #[inline]
    pub fn random(&mut self) -> f64 {
        let a = (self.gen_u32() >> 5) as u64;
        let b = (self.gen_u32() >> 6) as u64;
        ((a * 67_108_864 + b) as f64) / 9_007_199_254_740_992.0
    }

    pub fn randbelow(&mut self, n: u32) -> u32 {
        debug_assert!(n > 0);
        // CPython uses n.bit_length(), including one rejection bit for powers of two.
        let bits = 32 - n.leading_zeros();
        loop {
            let value = self.gen_u32() >> (32 - bits);
            if value < n {
                return value;
            }
        }
    }

    fn init_genrand(&mut self, seed: u32) {
        self.state[0] = seed;
        for index in 1..624 {
            self.state[index] = 1_812_433_253_u32
                .wrapping_mul(self.state[index - 1] ^ (self.state[index - 1] >> 30))
                .wrapping_add(index as u32);
        }
        self.index = 624;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19_650_218);
        let mut i = 1usize;
        let mut j = 0usize;
        for _ in 0..624.max(key.len()) {
            self.state[i] = (self.state[i]
                ^ ((self.state[i - 1] ^ (self.state[i - 1] >> 30)).wrapping_mul(1_664_525)))
            .wrapping_add(key[j])
            .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= 624 {
                self.state[0] = self.state[623];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..623 {
            self.state[i] = (self.state[i]
                ^ ((self.state[i - 1] ^ (self.state[i - 1] >> 30)).wrapping_mul(1_566_083_941)))
            .wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                self.state[0] = self.state[623];
                i = 1;
            }
        }
        self.state[0] = 0x8000_0000;
        self.index = 624;
    }

    #[inline]
    fn gen_u32(&mut self) -> u32 {
        const N: usize = 624;
        const M: usize = 397;
        const MATRIX_A: u32 = 0x9908_b0df;
        const UPPER_MASK: u32 = 0x8000_0000;
        const LOWER_MASK: u32 = 0x7fff_ffff;
        if self.index >= N {
            for i in 0..(N - M) {
                let y = (self.state[i] & UPPER_MASK) | (self.state[i + 1] & LOWER_MASK);
                self.state[i] =
                    self.state[i + M] ^ (y >> 1) ^ if y & 1 != 0 { MATRIX_A } else { 0 };
            }
            for i in (N - M)..(N - 1) {
                let y = (self.state[i] & UPPER_MASK) | (self.state[i + 1] & LOWER_MASK);
                self.state[i] =
                    self.state[i + M - N] ^ (y >> 1) ^ if y & 1 != 0 { MATRIX_A } else { 0 };
            }
            let y = (self.state[N - 1] & UPPER_MASK) | (self.state[0] & LOWER_MASK);
            self.state[N - 1] =
                self.state[M - 1] ^ (y >> 1) ^ if y & 1 != 0 { MATRIX_A } else { 0 };
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }
}

#[cfg(test)]
mod tests {
    use super::PyRandom;

    #[test]
    fn matches_cpython_reference_values() {
        let expected = [
            0.844_421_851_525_048_1,
            0.757_954_402_940_302_5,
            0.420_571_580_830_845,
            0.258_916_750_292_963_35,
        ];
        let mut rng = PyRandom::seed_u64(0);
        for value in expected {
            assert_eq!(rng.random(), value);
        }
        let mut rng = PyRandom::seed_u64(4_294_967_297);
        assert_eq!(rng.random(), 0.230_933_103_717_691_5);
    }

    #[test]
    fn choice_power_of_two_uses_cpython_rejection_rule() {
        let mut rng = PyRandom::seed_u64(123);
        let expected = [0, 4, 1, 6, 4, 1, 0, 6];
        for value in expected {
            assert_eq!(rng.randbelow(8), value);
        }
    }
}
