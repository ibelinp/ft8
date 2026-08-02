//! LDPC(174,91): encoder, parity check, and the belief-propagation decoder.
//! Ported from ft8_lib's `ldpc.c` and the `encode174` half of `encode.c`.
//!
//! A codeword is 174 bits: 91 payload bits (77 message + 14 CRC) followed by
//! 83 parity bits.

use crate::constants::{
    LDPC_GENERATOR, LDPC_K, LDPC_K_BYTES, LDPC_M, LDPC_MN, LDPC_N, LDPC_NM, LDPC_NUM_ROWS,
};

/// Rational approximations, kept bit-for-bit from ft8_lib. They are not just an
/// optimisation: the decoder's convergence behaviour — and so which marginal
/// signals decode — depends on their exact shape, so "improving" them to
/// `f32::tanh` would silently change results.
fn fast_tanh(x: f32) -> f32 {
    if x < -4.97 {
        return -1.0;
    }
    if x > 4.97 {
        return 1.0;
    }
    let x2 = x * x;
    let a = x * (945.0 + x2 * (105.0 + x2));
    let b = 945.0 + x2 * (420.0 + x2 * 15.0);
    a / b
}

fn fast_atanh(x: f32) -> f32 {
    let x2 = x * x;
    let a = x * (945.0 + x2 * (-735.0 + x2 * 64.0));
    let b = 945.0 + x2 * (-1050.0 + x2 * 225.0);
    a / b
}

/// Number of failed parity checks; 0 means a valid codeword.
pub fn check(codeword: &[u8; LDPC_N]) -> usize {
    let mut errors = 0;
    for m in 0..LDPC_M {
        let mut x = 0u8;
        for i in 0..LDPC_NUM_ROWS[m] as usize {
            x ^= codeword[LDPC_NM[m][i] as usize - 1];
        }
        if x != 0 {
            errors += 1;
        }
    }
    errors
}

/// Encode 91 payload bits (12 bytes, MSB first) into a 174-bit codeword,
/// returned as one byte per bit.
pub fn encode(message: &[u8; LDPC_K_BYTES]) -> [u8; LDPC_N] {
    let mut out = [0u8; LDPC_N];
    for (i, slot) in out.iter_mut().enumerate().take(LDPC_K) {
        *slot = (message[i / 8] >> (7 - (i % 8))) & 1;
    }
    for i in 0..LDPC_M {
        let mut nsum = 0u32;
        for j in 0..LDPC_K_BYTES {
            nsum ^= (message[j] & LDPC_GENERATOR[i][j]).count_ones();
        }
        out[LDPC_K + i] = (nsum & 1) as u8;
    }
    out
}

/// Belief-propagation decode of 174 log-likelihoods (`log P(0)/P(1)`).
///
/// Returns the hard-decision bits and the smallest parity-error count seen;
/// 0 means a clean codeword. ft8_lib also ships a sum-product decoder, but it
/// allocates two 83×174 f32 planes (~115 kB of stack) and `decode.c` does not
/// use it — this one needs under 3 kB, which matters in wasm.
pub fn bp_decode(codeword: &[f32; LDPC_N], max_iters: usize) -> ([u8; LDPC_N], usize) {
    let mut tov = [[0.0f32; 3]; LDPC_N];
    let mut toc = [[0.0f32; 7]; LDPC_M];
    let mut plain = [0u8; LDPC_N];
    let mut min_errors = LDPC_M;

    for _ in 0..max_iters {
        // Hard decision from the current beliefs (tov is zero on the first pass).
        let mut plain_sum = 0u32;
        for n in 0..LDPC_N {
            plain[n] = ((codeword[n] + tov[n][0] + tov[n][1] + tov[n][2]) > 0.0) as u8;
            plain_sum += plain[n] as u32;
        }
        // All-zeros satisfies every parity check but is not a legal message, so
        // converging there means we have lost the signal, not found it.
        if plain_sum == 0 {
            break;
        }

        let errors = check(&plain);
        if errors < min_errors {
            min_errors = errors;
            if errors == 0 {
                break;
            }
        }

        // Bits → checks.
        for m in 0..LDPC_M {
            for n_idx in 0..LDPC_NUM_ROWS[m] as usize {
                let n = LDPC_NM[m][n_idx] as usize - 1;
                let mut tnm = codeword[n];
                for m_idx in 0..3 {
                    if LDPC_MN[n][m_idx] as usize - 1 != m {
                        tnm += tov[n][m_idx];
                    }
                }
                toc[m][n_idx] = fast_tanh(-tnm / 2.0);
            }
        }

        // Checks → bits.
        for n in 0..LDPC_N {
            for m_idx in 0..3 {
                let m = LDPC_MN[n][m_idx] as usize - 1;
                let mut tmn = 1.0f32;
                for n_idx in 0..LDPC_NUM_ROWS[m] as usize {
                    if LDPC_NM[m][n_idx] as usize - 1 != n {
                        tmn *= toc[m][n_idx];
                    }
                }
                tov[n][m_idx] = -2.0 * fast_atanh(tmn);
            }
        }
    }

    (plain, min_errors)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random payloads — no rand dependency, and a failure
    /// reproduces exactly.
    fn payload(seed: u64) -> [u8; LDPC_K_BYTES] {
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut m = [0u8; LDPC_K_BYTES];
        for b in m.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *b = (s >> 33) as u8;
        }
        m[11] &= 0xE0; // 91 bits used of 96; keep the pad clear
        m
    }

    /// The tables and the encoder have to agree before anything else can be
    /// trusted: an encoded word must satisfy every parity check by construction.
    #[test]
    fn encoded_words_are_valid_codewords() {
        for seed in 0..64 {
            let cw = encode(&payload(seed));
            assert_eq!(check(&cw), 0, "seed {seed} encoded to an invalid codeword");
        }
    }

    /// Soft-decision input with no noise: the decoder must return exactly the
    /// bits that went in.
    #[test]
    fn decodes_a_clean_codeword() {
        for seed in 0..32 {
            let cw = encode(&payload(seed));
            let mut llr = [0.0f32; LDPC_N];
            for i in 0..LDPC_N {
                llr[i] = if cw[i] == 1 { 4.0 } else { -4.0 };
            }
            let (plain, errors) = bp_decode(&llr, 20);
            assert_eq!(errors, 0, "seed {seed} failed to converge");
            assert_eq!(plain, cw, "seed {seed} decoded to the wrong codeword");
        }
    }

    /// The point of the code. LDPC(174,91) should shrug off a handful of
    /// flipped bits; if the Nm/Mn tables were transposed or off by one this is
    /// what would catch it, since clean decoding alone would still pass.
    #[test]
    fn corrects_flipped_bits() {
        let mut recovered = 0;
        for seed in 0..32 {
            let cw = encode(&payload(seed));
            let mut llr = [0.0f32; LDPC_N];
            for i in 0..LDPC_N {
                llr[i] = if cw[i] == 1 { 4.0 } else { -4.0 };
            }
            // Flip 6 bits, spread across the word.
            for k in 0..6 {
                let i = (seed as usize * 7 + k * 29) % LDPC_N;
                llr[i] = -llr[i];
            }
            let (plain, errors) = bp_decode(&llr, 30);
            if errors == 0 && plain == cw {
                recovered += 1;
            }
        }
        assert!(
            recovered >= 30,
            "only {recovered}/32 recovered from 6 bit errors"
        );
    }

    /// Noise with no signal must not produce a confident answer.
    #[test]
    fn rejects_garbage() {
        let mut llr = [0.0f32; LDPC_N];
        let mut s = 12345u64;
        for v in llr.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = ((s >> 40) as i32 as f32 % 200.0) / 100.0 - 1.0;
        }
        let (_, errors) = bp_decode(&llr, 30);
        assert!(errors > 0, "random LLRs decoded to a valid codeword");
    }
}
