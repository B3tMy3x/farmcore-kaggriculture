//! CPython Random integer seeding, MT19937, random() and _randbelow().
use num_bigint::BigInt;

// CPython's init_genrand(19650218) prefix is independent of the episode seed.
// Compute it at compile time; init_by_array below still consumes every seed word.
const INITIAL_STATE: [u32; 624] = {
    let mut mt = [0u32; 624];
    mt[0] = 19650218;
    let mut i = 1;
    while i < 624 {
        mt[i] = 1812433253u32
            .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
            .wrapping_add(i as u32);
        i += 1;
    }
    mt
};

pub struct PyRandom {
    mt: [u32; 624],
    index: usize,
}
impl PyRandom {
    pub fn new(seed: &BigInt) -> Self {
        let (_, mut key) = seed.to_u32_digits();
        if key.is_empty() {
            key.push(0);
        }
        let mut r = Self {
            mt: INITIAL_STATE,
            index: 624,
        };
        let (mut i, mut j) = (1, 0);
        for _ in 0..624.max(key.len()) {
            r.mt[i] = (r.mt[i] ^ (r.mt[i - 1] ^ (r.mt[i - 1] >> 30)).wrapping_mul(1664525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= 624 {
                r.mt[0] = r.mt[623];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        // Exactly 623 updates traverse indices 1..624 once, starting at i.
        // Split at the wrap to keep the state-copy boundary outside the loop.
        let start = i;
        for i in start..624 {
            r.mt[i] = (r.mt[i] ^ (r.mt[i - 1] ^ (r.mt[i - 1] >> 30)).wrapping_mul(1566083941))
                .wrapping_sub(i as u32);
        }
        r.mt[0] = r.mt[623];
        for i in 1..start {
            r.mt[i] = (r.mt[i] ^ (r.mt[i - 1] ^ (r.mt[i - 1] >> 30)).wrapping_mul(1566083941))
                .wrapping_sub(i as u32);
        }
        r.mt[0] = 0x80000000;
        r
    }
    fn word(&mut self) -> u32 {
        if self.index >= 624 {
            // MT19937 twists in place: the second range deliberately reads
            // words already updated by the first. Splitting at the wrap points
            // removes the modulo operations while preserving CPython's order.
            for i in 0..227 {
                let y = (self.mt[i] & 0x80000000) | (self.mt[i + 1] & 0x7fffffff);
                self.mt[i] = self.mt[i + 397] ^ (y >> 1) ^ (0u32.wrapping_sub(y & 1) & 0x9908b0df);
            }
            for i in 227..623 {
                let y = (self.mt[i] & 0x80000000) | (self.mt[i + 1] & 0x7fffffff);
                self.mt[i] = self.mt[i - 227] ^ (y >> 1) ^ (0u32.wrapping_sub(y & 1) & 0x9908b0df);
            }
            let y = (self.mt[623] & 0x80000000) | (self.mt[0] & 0x7fffffff);
            self.mt[623] = self.mt[396] ^ (y >> 1) ^ (0u32.wrapping_sub(y & 1) & 0x9908b0df);
            self.index = 0;
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        y
    }
    pub fn random(&mut self) -> f64 {
        let a = self.word() >> 5;
        let b = self.word() >> 6;
        ((a as u64) * 67108864 + b as u64) as f64 / 9007199254740992.0
    }
    pub fn below(&mut self, n: u32) -> usize {
        let k = 32 - n.leading_zeros();
        loop {
            let x = self.word() >> (32 - k);
            if x < n {
                return x as usize;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matches_cpython_vectors() {
        let cases: serde_json::Value =
            serde_json::from_str(include_str!("../tests/rng_vectors.json")).unwrap();
        for case in cases.as_array().unwrap() {
            let seed = case["seed"].as_str().unwrap().parse().unwrap();
            let mut r = PyRandom::new(&seed);
            for v in case["random"].as_array().unwrap() {
                assert_eq!(r.random().to_le_bytes(), v.as_f64().unwrap().to_le_bytes());
            }
            for v in case["choices"].as_array().unwrap() {
                assert_eq!(r.below(8), v.as_u64().unwrap() as usize);
            }
        }
    }

    #[test]
    fn matches_cpython_for_seed_longer_than_state_and_word_boundaries() {
        // Generated independently with CPython random.Random(seed).getrandbits(32),
        // then randrange(n). 688 little-endian seed words exercise key.len() > 624.
        let seed = (BigInt::from(1) << 22000usize)
            + (BigInt::from(1) << 19936usize)
            + (BigInt::from(1) << 127usize)
            + BigInt::from(123456789);
        let words = [
            (0, 3447583537),
            (1, 1382185916),
            (225, 2359304119),
            (226, 373838110),
            (227, 3237653121),
            (228, 1360699564),
            (621, 2069569639),
            (622, 1876283811),
            (623, 2665149902),
            (624, 350084011),
            (625, 1019099936),
            (1246, 3638729059),
            (1247, 3095445822),
            (1248, 1084786694),
            (1249, 1549195842),
            (1871, 1333888718),
            (1872, 423179864),
            (2495, 3621450277),
            (2496, 629349482),
        ];
        let choices = [
            (1, 0),
            (2, 1),
            (3, 2),
            (7, 1),
            (8, 1),
            (31, 15),
            (32, 15),
            (33, 26),
            (2147483648, 2117741964),
            (4294967295, 698912355),
        ];
        for signed_seed in [seed.clone(), -seed] {
            let mut rng = PyRandom::new(&signed_seed);
            let mut index = 0;
            for (position, expected) in words {
                while index < position {
                    rng.word();
                    index += 1;
                }
                assert_eq!(rng.word(), expected, "word {position}");
                index += 1;
            }
            for (n, expected) in choices {
                assert_eq!(rng.below(n), expected, "below {n}");
            }
        }
    }
}
