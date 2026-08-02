//! FT8 and FT4 decoding, in pure Rust. Give it audio, get messages.
//!
//! ```
//! use ft8::decode;
//!
//! # let audio = vec![0.0f32; 12000 * 15];
//! // 15 seconds of 12 kHz mono audio, aligned to the UTC slot boundary.
//! for d in decode(&audio, 12000.0) {
//!     println!("{:+3.0} dB  {:+4.1} s  {:5.0} Hz  {}",
//!              d.status.snr_db, d.time_sec, d.freq_hz, d.text());
//! }
//! ```
//!
//! For a live receiver, feed it as the audio arrives instead — one symbol at a
//! time — so the FFT cost spreads across the slot rather than landing as one
//! stall at the end:
//!
//! ```
//! use ft8::{Monitor, MonitorConfig};
//!
//! let mut mon = Monitor::new(MonitorConfig::default());
//! # let audio = vec![0.0f32; 12000 * 15];
//! for chunk in audio.chunks(mon.block_size()) {
//!     mon.process(chunk);          // as each symbol arrives
//! }
//! let decodes = mon.decode_all(30, 10, 30);
//! # let _ = decodes;
//! ```
//!
//! # Scope
//!
//! A complete receive chain: FFT, Costas sync search, soft-bit extraction,
//! LDPC(174,91), CRC-14 and message unpacking. Encoding too, which exists
//! mostly so the decoder can be tested without off-air recordings.
//!
//! Not covered: the contest and nonstandard-callsign message types (0.1–0.4,
//! 0.6, 3, 4, 5) are *identified* by [`Message::message_type`] but not rendered
//! as text, and hashed callsigns show as `<...>` since no hash table is kept.
//!
//! # Provenance and licensing
//!
//! A port of [ft8_lib](https://github.com/kgoba/ft8_lib) by Kārlis Goba, used
//! under the MIT licence — see `LICENSE-ft8_lib`, which must travel with any
//! copy of this crate. This crate is likewise MIT.
//!
//! The LDPC matrices originate in WSJT-X (`ldpc_174_91_c_reordered_parity.f90`,
//! `bpdecode174.f90`). They are code *parameters* published in the FT8
//! specification rather than WSJT-X source, so nothing here is encumbered by
//! WSJT-X's GPL — which is the whole reason this port exists: every other
//! browser-capable FT8 decoder is GPL or unlicensed.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod constants;
pub mod crc;
pub mod decode;
pub mod ldpc;
pub mod message;
pub mod monitor;
pub mod text;

pub use decode::Protocol;
pub use message::{CallsignHash, Error, Field, Message, MessageType, NoHash, StdMessage};
pub use monitor::{Decode, Monitor, MonitorConfig};
pub use text::Str;

/// Decode a whole slot of audio in one call.
///
/// `audio` should be mono and slot-aligned — 15 s for FT8. Anything past a full
/// slot is ignored; anything short is decoded as far as it goes. For a live
/// receiver prefer [`Monitor`], which spreads the work across the slot.
pub fn decode(audio: &[f32], sample_rate: f32) -> Vec<Decode> {
    decode_with(
        audio,
        MonitorConfig {
            sample_rate,
            ..MonitorConfig::default()
        },
    )
}

/// [`decode`] with the analysis spelled out — a different slice of audio to
/// search, FT4 instead of FT8, or cheaper oversampling.
pub fn decode_with(audio: &[f32], cfg: MonitorConfig) -> Vec<Decode> {
    let mut mon = Monitor::new(cfg);
    for chunk in audio.chunks(mon.block_size()) {
        mon.process(chunk);
    }
    mon.decode_all(30, 10, 30)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::NoHash;

    /// The layers have to line up: a message packed at the top must survive
    /// CRC, LDPC encoding, a noisy channel, and come back out identical.
    /// Each piece is tested alone; this is the one that catches a seam.
    #[test]
    fn message_survives_crc_ldpc_and_bit_errors() {
        let msg = Message::encode_std("CQ", "K1ABC", "FN42").unwrap();

        let mut a91 = [0u8; 12];
        crc::add(&msg.payload, &mut a91);
        assert!(crc::check(&a91));

        let codeword = ldpc::encode(&a91);
        assert_eq!(ldpc::check(&codeword), 0);

        let mut llr = [0.0f32; constants::LDPC_N];
        for i in 0..constants::LDPC_N {
            llr[i] = if codeword[i] == 1 { 4.0 } else { -4.0 };
        }
        for k in 0..6 {
            llr[k * 23 % constants::LDPC_N] *= -1.0;
        }

        let (plain, errors) = ldpc::bp_decode(&llr, 30);
        assert_eq!(errors, 0, "LDPC failed to correct 6 bit errors");

        let mut recovered = [0u8; 12];
        for (i, &bit) in plain.iter().take(91).enumerate() {
            recovered[i / 8] |= bit << (7 - (i % 8));
        }
        assert!(
            crc::check(&recovered),
            "recovered payload failed its own CRC"
        );

        let mut payload = [0u8; message::PAYLOAD_BYTES];
        payload.copy_from_slice(&recovered[..message::PAYLOAD_BYTES]);
        let out = Message::from_payload(payload)
            .decode_std(&mut NoHash)
            .unwrap();
        assert_eq!(&*out.call_to, "CQ");
        assert_eq!(&*out.call_de, "K1ABC");
        assert_eq!(&*out.extra, "FN42");
    }
}
