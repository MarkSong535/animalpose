//! numpy's `np.random.default_rng(seed)`: `SeedSequence` seeding of the `PCG64` bit generator, and
//! the few `Generator` methods the ported code draws from (`random`, `uniform`, `integers`-style
//! bounded draws and `permutation`), stream for stream.
//!
//! [`crate::data::synthetic::Rng`] is a different generator (identically distributed, different
//! draws); this one exists where Python's *exact* draws decide something visible, e.g. which animal
//! `data/datasets.py:PoseCropDataset` crops (`np.random.default_rng((13, base)).permutation(ids)`).

/// `SeedSequence(entropy)` with the default pool size of 4 words.
#[derive(Clone, Debug)]
pub struct SeedSequence {
    pool: [u32; 4],
}

const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const XSHIFT: u32 = 16;

fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut v = value ^ *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    v = v.wrapping_mul(*hash_const);
    v ^ (v >> XSHIFT)
}

fn mix(x: u32, y: u32) -> u32 {
    let r = MIX_MULT_L.wrapping_mul(x).wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    r ^ (r >> XSHIFT)
}

/// `_int_to_uint32_array`: a non-negative int as little-endian 32-bit words (`[0]` for zero).
fn int_words(mut n: u128) -> Vec<u32> {
    if n == 0 {
        return vec![0];
    }
    let mut out = Vec::new();
    while n > 0 {
        out.push((n & 0xffff_ffff) as u32);
        n >>= 32;
    }
    out
}

impl SeedSequence {
    /// `SeedSequence(entropy)` for an int or a tuple of ints (each split into 32-bit words).
    pub fn new(entropy: &[u128]) -> Self {
        let words: Vec<u32> = entropy.iter().flat_map(|v| int_words(*v)).collect();
        let mut pool = [0u32; 4];
        let mut hash_const = INIT_A;
        for (i, slot) in pool.iter_mut().enumerate() {
            *slot = hashmix(words.get(i).copied().unwrap_or(0), &mut hash_const);
        }
        for i_src in 0..4 {
            for i_dst in 0..4 {
                if i_src != i_dst {
                    let h = hashmix(pool[i_src], &mut hash_const);
                    pool[i_dst] = mix(pool[i_dst], h);
                }
            }
        }
        for w in words.iter().skip(4) {
            for slot in pool.iter_mut() {
                let h = hashmix(*w, &mut hash_const);
                *slot = mix(*slot, h);
            }
        }
        Self { pool }
    }

    /// `generate_state(n_words, np.uint64)`.
    pub fn generate_state_u64(&self, n_words: usize) -> Vec<u64> {
        let mut hash_const = INIT_B;
        let words: Vec<u32> = (0..n_words * 2)
            .map(|i| {
                let mut v = self.pool[i % 4] ^ hash_const;
                hash_const = hash_const.wrapping_mul(MULT_B);
                v = v.wrapping_mul(hash_const);
                v ^ (v >> XSHIFT)
            })
            .collect();
        words
            .chunks_exact(2)
            .map(|p| u64::from(p[0]) | (u64::from(p[1]) << 32))
            .collect()
    }
}

const PCG_MULT: u128 = (2_549_297_995_355_413_924u128 << 64) | 4_865_540_595_714_422_341u128;

/// `np.random.Generator(PCG64(seed))`, i.e. `np.random.default_rng(seed)`.
#[derive(Clone, Debug)]
pub struct NpRng {
    state: u128,
    inc: u128,
    buffered: Option<u32>,
}

impl NpRng {
    /// `np.random.default_rng(seed)` for an int seed.
    pub fn new(seed: u128) -> Self {
        Self::from_entropy(&[seed])
    }

    /// `np.random.default_rng((a, b, ...))` for a tuple of ints.
    pub fn from_entropy(entropy: &[u128]) -> Self {
        let s = SeedSequence::new(entropy).generate_state_u64(4);
        let initstate = (u128::from(s[0]) << 64) | u128::from(s[1]);
        let initseq = (u128::from(s[2]) << 64) | u128::from(s[3]);
        let mut rng = Self {
            state: 0,
            inc: (initseq << 1) | 1,
            buffered: None,
        };
        rng.step();
        rng.state = rng.state.wrapping_add(initstate);
        rng.step();
        rng
    }

    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
    }

    /// `bit_generator.random_raw()`: the next 64-bit output (XSL-RR).
    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let hi = (self.state >> 64) as u64;
        let lo = self.state as u64;
        (hi ^ lo).rotate_right((self.state >> 122) as u32)
    }

    /// The buffered 32-bit draw numpy's `next_uint32` makes: low half first, high half next.
    pub fn next_u32(&mut self) -> u32 {
        if let Some(v) = self.buffered.take() {
            return v;
        }
        let v = self.next_u64();
        self.buffered = Some((v >> 32) as u32);
        v as u32
    }

    /// `rng.random()`.
    pub fn random(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// `rng.uniform(low, high)`.
    pub fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.random()
    }

    /// numpy's `random_interval(max)`: a uniform integer in `0..=max` by masked rejection.
    pub fn random_interval(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        let mut mask = max;
        for s in [1, 2, 4, 8, 16, 32] {
            mask |= mask >> s;
        }
        if max <= 0xffff_ffff {
            loop {
                let v = u64::from(self.next_u32()) & mask;
                if v <= max {
                    return v;
                }
            }
        }
        loop {
            let v = self.next_u64() & mask;
            if v <= max {
                return v;
            }
        }
    }

    /// `rng.shuffle(x)` for a 1-D array (Fisher-Yates from the end).
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        for i in (1..x.len()).rev() {
            let j = self.random_interval(i as u64) as usize;
            x.swap(i, j);
        }
    }

    /// `rng.permutation(x)` for a 1-D array.
    pub fn permutation<T: Clone>(&mut self, x: &[T]) -> Vec<T> {
        let mut out = x.to_vec();
        self.shuffle(&mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    //! Reference streams printed by numpy 2.4 (`SeedSequence((13, 5)).generate_state(4, np.uint64)`,
    //! `default_rng(...).bit_generator.random_raw(...)`, `.random()`, `.permutation(...)`).
    use super::*;

    #[test]
    fn seed_sequence_and_pcg64_match_numpy() {
        let s = SeedSequence::new(&[13, 5]).generate_state_u64(4);
        assert_eq!(
            s,
            [0x1481e3e38d40e6a6, 0x2941404b50acbe34, 0x89f1eee3e0bc592a, 0x6c95064ffb1e8156]
        );
        let mut g = NpRng::from_entropy(&[13, 5]);
        assert_eq!(
            [g.next_u64(), g.next_u64(), g.next_u64()],
            [18422293464129305842, 11661211149954911736, 17269690747565966683]
        );
        let mut g = NpRng::from_entropy(&[13, 5]);
        assert_eq!((g.random(), g.random()), (0.9986745297987251, 0.632155523129665));
        let mut g = NpRng::new(17);
        assert_eq!([g.next_u64(), g.next_u64()], [15588878425886018675, 2969429415550431307]);
    }

    #[test]
    fn permutation_matches_numpy() {
        let cases: [(u128, [i64; 2], [i64; 5]); 6] = [
            (0, [1, 0], [3, 9, 11, 7, 20]),
            (1, [0, 1], [7, 11, 9, 3, 20]),
            (2, [1, 0], [7, 20, 3, 9, 11]),
            (7, [0, 1], [20, 7, 9, 3, 11]),
            (4_000_000_000, [0, 1], [7, 11, 3, 9, 20]),
            (5_000_000_000, [1, 0], [9, 11, 7, 3, 20]),
        ];
        for (base, two, five) in cases {
            assert_eq!(NpRng::from_entropy(&[13, base]).permutation(&[0, 1]), two, "{base}");
            assert_eq!(
                NpRng::from_entropy(&[13, base]).permutation(&[3, 7, 9, 11, 20]),
                five,
                "{base}"
            );
        }
    }
}
