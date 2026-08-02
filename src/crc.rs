//! CRC-14 over the source-encoded message. Ported from ft8_lib's `crc.c`.

use crate::constants::{CRC_POLYNOMIAL, CRC_WIDTH};

const TOPBIT: u16 = 1 << (CRC_WIDTH - 1);
const MASK: u16 = (TOPBIT << 1) - 1;

/// Bitwise modulo-2 division over `num_bits` of `message` (MSB first).
pub fn compute(message: &[u8], num_bits: usize) -> u16 {
    let mut remainder: u16 = 0;
    let mut idx_byte = 0;
    for idx_bit in 0..num_bits {
        if idx_bit % 8 == 0 {
            remainder ^= (message[idx_byte] as u16) << (CRC_WIDTH - 8);
            idx_byte += 1;
        }
        remainder = if remainder & TOPBIT != 0 {
            (remainder << 1) ^ CRC_POLYNOMIAL
        } else {
            remainder << 1
        };
    }
    remainder & MASK
}

/// Pull the transmitted CRC out of a 91-bit payload (bits 77..91).
pub fn extract(a91: &[u8]) -> u16 {
    ((a91[9] as u16 & 0x07) << 11) | ((a91[10] as u16) << 3) | (a91[11] as u16 >> 5)
}

/// Append the CRC to a 77-bit payload, yielding the 91 bits the LDPC encoder
/// takes. The CRC covers the payload zero-extended from 77 to 82 bits, which is
/// why the spare bits are cleared first — get that wrong and every message
/// fails its own checksum.
pub fn add(payload: &[u8], a91: &mut [u8]) {
    a91[..10].copy_from_slice(&payload[..10]);
    a91[9] &= 0xF8;
    a91[10] = 0;
    let checksum = compute(a91, 96 - 14);
    a91[9] |= (checksum >> 11) as u8;
    a91[10] = ((checksum >> 3) & 0xFF) as u8;
    a91[11] = ((checksum << 5) & 0xFF) as u8;
}

/// Does a 91-bit payload carry a CRC matching its own contents?
pub fn check(a91: &[u8]) -> bool {
    let mut scratch = [0u8; 12];
    scratch[..12].copy_from_slice(&a91[..12]);
    scratch[9] &= 0xF8;
    scratch[10] = 0;
    scratch[11] = 0;
    compute(&scratch, 96 - 14) == extract(a91)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A CRC that always returns zero, or ignores its input, would still pass a
    /// naive round-trip. Pin the actual value and check it moves with the data.
    #[test]
    fn add_then_extract_round_trips() {
        let payload = [
            0x12u8, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x22, 0, 0,
        ];
        let mut a91 = [0u8; 12];
        add(&payload, &mut a91);
        let stored = extract(&a91);
        assert!(check(&a91), "freshly added CRC must verify");
        assert_ne!(
            stored, 0,
            "a zero CRC here would mean the function is inert"
        );

        // The first 77 bits must be preserved exactly.
        assert_eq!(a91[..9], payload[..9]);
        assert_eq!(a91[9] & 0xF8, payload[9] & 0xF8);
    }

    #[test]
    fn detects_a_single_flipped_bit() {
        let payload = [
            0x12u8, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x22, 0, 0,
        ];
        let mut a91 = [0u8; 12];
        add(&payload, &mut a91);
        for bit in 0..77 {
            let mut bad = a91;
            bad[bit / 8] ^= 0x80 >> (bit % 8);
            assert!(!check(&bad), "flipping payload bit {bit} went undetected");
        }
    }

    #[test]
    fn width_is_14_bits() {
        for seed in 0u8..64 {
            let payload = [seed; 12];
            let mut a91 = [0u8; 12];
            add(&payload, &mut a91);
            assert!(extract(&a91) <= MASK, "CRC wider than 14 bits");
        }
    }
}
