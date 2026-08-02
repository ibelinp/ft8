//! Audio → spectrogram, and the convenience decode built on it.
//! Ported from ft8_lib's `common/monitor.c`.
//!
//! **Feed this incrementally.** [`Monitor::process`] takes exactly one symbol
//! of audio and does that symbol's FFTs immediately, so the cost is spread
//! across the 15-second slot instead of landing as one stall at the end. Same
//! total work either way; completely different responsiveness. Buffering the
//! whole slot and calling this 93 times in a row at the boundary works, and is
//! the wrong thing to do in anything with a UI.

use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::Arc;

use crate::constants::{FT4_SLOT_TIME, FT4_SYMBOL_PERIOD, FT8_SLOT_TIME, FT8_SYMBOL_PERIOD};
use crate::decode::{
    decode_candidate, find_candidates, Candidate, DecodeStatus, Protocol, Waterfall,
};
use crate::message::{Message, MessageType, NoHash};
use crate::text::Str;

/// How to analyse the audio.
#[derive(Clone, Copy, Debug)]
pub struct MonitorConfig {
    pub sample_rate: f32,
    /// Lowest audio frequency to look at, Hz.
    pub f_min: f32,
    /// Highest audio frequency to look at, Hz.
    pub f_max: f32,
    pub protocol: Protocol,
    /// Time subdivisions per symbol. 2 is the usual choice.
    pub time_osr: usize,
    /// Frequency subdivisions per tone spacing. 2 is the usual choice.
    pub freq_osr: usize,
}

impl Default for MonitorConfig {
    /// The standard setup: 12 kHz audio, the 200–3000 Hz slice FT8 lives in,
    /// and 2× oversampling on both axes.
    fn default() -> Self {
        Self {
            sample_rate: 12000.0,
            f_min: 200.0,
            f_max: 3000.0,
            protocol: Protocol::Ft8,
            time_osr: 2,
            freq_osr: 2,
        }
    }
}

/// One decoded transmission.
#[derive(Clone, Debug)]
pub struct Decode {
    pub message: Message,
    /// Audio frequency of the signal, Hz.
    pub freq_hz: f32,
    /// Offset of the signal from the start of the slot, seconds. Negative means
    /// the sender was early.
    pub time_sec: f32,
    /// Sync strength. Not an SNR — see `status.snr_db` for that.
    pub score: i16,
    pub status: DecodeStatus,
}

impl Decode {
    /// The message as you would see it in a decode list, e.g. `CQ K1ABC FN42`.
    ///
    /// Message types this crate does not render yet come back as `<type>` —
    /// visible as "something was here" rather than silently dropped, which is
    /// what makes a missing decode debuggable.
    pub fn text(&self) -> Str<40> {
        let mut out = Str::new();
        match self.message.message_type() {
            MessageType::Standard => {
                if let Ok(m) = self.message.decode_std(&mut NoHash) {
                    out.push_str(&m.call_to);
                    out.push(b' ');
                    out.push_str(&m.call_de);
                    if !m.extra.is_empty() {
                        out.push(b' ');
                        out.push_str(&m.extra);
                    }
                } else {
                    out.push_str("<undecodable>");
                }
            }
            MessageType::FreeText => out.push_str(&self.message.decode_free()),
            MessageType::Telemetry => out.push_str(&self.message.telemetry_hex()),
            other => {
                out.push_str("<");
                out.push_str(match other {
                    MessageType::DxPedition => "DXpedition",
                    MessageType::EuVhf => "EU VHF",
                    MessageType::ArrlFd => "ARRL FD",
                    MessageType::Contesting => "contest",
                    MessageType::ArrlRtty => "ARRL RTTY",
                    MessageType::NonstdCall => "nonstd call",
                    MessageType::Wwrof => "WWROF",
                    _ => "unknown",
                });
                out.push_str(">");
            }
        }
        out
    }
}

/// Accumulates audio into a spectrogram, then decodes it.
pub struct Monitor {
    block_size: usize,
    subblock_size: usize,
    nfft: usize,
    time_osr: usize,
    freq_osr: usize,
    min_bin: usize,
    num_bins: usize,
    max_blocks: usize,
    num_blocks: usize,
    symbol_period: f32,
    protocol: Protocol,
    window: Vec<f32>,
    last_frame: Vec<f32>,
    mag: Vec<u8>,
    scratch: Vec<Complex<f32>>,
    fft: Arc<dyn Fft<f32>>,
}

// rustfft's planned FFT is not Debug, and the buffers are large and
// uninteresting — print the parameters that actually identify a Monitor.
impl core::fmt::Debug for Monitor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Monitor")
            .field("protocol", &self.protocol)
            .field("block_size", &self.block_size)
            .field("nfft", &self.nfft)
            .field("time_osr", &self.time_osr)
            .field("freq_osr", &self.freq_osr)
            .field("num_bins", &self.num_bins)
            .field("num_blocks", &self.num_blocks)
            .field("max_blocks", &self.max_blocks)
            .finish()
    }
}

/// Hann window, as `sin²(πi/N)`.
fn hann(i: usize, n: usize) -> f32 {
    let x = (core::f32::consts::PI * i as f32 / n as f32).sin();
    x * x
}

impl Monitor {
    pub fn new(cfg: MonitorConfig) -> Self {
        let (slot_time, symbol_period) = match cfg.protocol {
            Protocol::Ft8 => (FT8_SLOT_TIME, FT8_SYMBOL_PERIOD),
            Protocol::Ft4 => (FT4_SLOT_TIME, FT4_SYMBOL_PERIOD),
        };
        let block_size = (cfg.sample_rate * symbol_period) as usize;
        let subblock_size = block_size / cfg.time_osr;
        let nfft = block_size * cfg.freq_osr;
        // Folded into the window so it costs nothing per frame.
        let fft_norm = 2.0 / nfft as f32;

        // A bin is 1/symbol_period Hz wide — 6.25 Hz for FT8 — which is exactly
        // the tone spacing. That identity is the whole reason these numbers
        // work out, and why the FFT size is not a power of two.
        let min_bin = (cfg.f_min * symbol_period) as usize;
        let max_bin = (cfg.f_max * symbol_period) as usize + 1;
        let num_bins = max_bin - min_bin;
        let max_blocks = (slot_time / symbol_period) as usize;

        let window: Vec<f32> = (0..nfft).map(|i| fft_norm * hann(i, nfft)).collect();
        let fft = FftPlanner::<f32>::new().plan_fft_forward(nfft);

        Self {
            block_size,
            subblock_size,
            nfft,
            time_osr: cfg.time_osr,
            freq_osr: cfg.freq_osr,
            min_bin,
            num_bins,
            max_blocks,
            num_blocks: 0,
            symbol_period,
            protocol: cfg.protocol,
            window,
            last_frame: vec![0.0; nfft],
            mag: vec![0; max_blocks * cfg.time_osr * cfg.freq_osr * num_bins],
            scratch: vec![Complex::new(0.0, 0.0); nfft],
            fft,
        }
    }

    /// Samples [`process`](Self::process) expects per call — one symbol's worth.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Symbols stored so far.
    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// True once a full slot has been collected; further `process` calls are
    /// ignored.
    pub fn is_full(&self) -> bool {
        self.num_blocks >= self.max_blocks
    }

    /// Start a new slot.
    pub fn reset(&mut self) {
        self.num_blocks = 0;
        self.last_frame.fill(0.0);
    }

    /// Analyse one symbol of audio. `frame` must be [`block_size`](Self::block_size)
    /// samples; anything else is ignored, since a short frame would silently
    /// shift every subsequent symbol.
    pub fn process(&mut self, frame: &[f32]) {
        if self.is_full() || frame.len() != self.block_size {
            return;
        }
        let block_stride = self.time_osr * self.freq_osr * self.num_bins;
        let mut offset = self.num_blocks * block_stride;
        let mut frame_pos = 0;
        let fft = self.fft.clone();

        for _ in 0..self.time_osr {
            // Slide the analysis window along by one subblock. The FFT spans
            // `freq_osr` symbols, so consecutive frames overlap heavily —
            // that overlap is what gives sub-symbol time resolution.
            self.last_frame.copy_within(self.subblock_size.., 0);
            let keep = self.nfft - self.subblock_size;
            self.last_frame[keep..]
                .copy_from_slice(&frame[frame_pos..frame_pos + self.subblock_size]);
            frame_pos += self.subblock_size;

            let (window, last_frame, scratch) = (&self.window, &self.last_frame, &mut self.scratch);
            for (i, s) in scratch.iter_mut().enumerate() {
                *s = Complex::new(window[i] * last_frame[i], 0.0);
            }
            fft.process(&mut self.scratch);

            for freq_sub in 0..self.freq_osr {
                for bin in self.min_bin..self.min_bin + self.num_bins {
                    let c = self.scratch[bin * self.freq_osr + freq_sub];
                    let mag2 = c.re * c.re + c.im * c.im;
                    let db = 10.0 * (1e-12 + mag2).log10();
                    // 0..240 spans -120..0 dB in half-dB steps.
                    self.mag[offset] = ((2.0 * db + 240.0) as i32).clamp(0, 255) as u8;
                    offset += 1;
                }
            }
        }
        self.num_blocks += 1;
    }

    /// Borrow the spectrogram accumulated so far.
    pub fn waterfall(&self) -> Waterfall<'_> {
        Waterfall {
            mag: &self.mag,
            num_blocks: self.num_blocks,
            num_bins: self.num_bins,
            time_osr: self.time_osr,
            freq_osr: self.freq_osr,
            protocol: self.protocol,
        }
    }

    /// Find and decode every message in the slot.
    ///
    /// Adjacent candidates routinely resolve to the same transmission, so
    /// results are de-duplicated by payload — otherwise one strong station
    /// appears half a dozen times.
    pub fn decode_all(
        &self,
        max_candidates: usize,
        min_score: i16,
        max_iterations: usize,
    ) -> Vec<Decode> {
        let wf = self.waterfall();
        let mut heap = vec![Candidate::default(); max_candidates];
        let n = find_candidates(&wf, &mut heap, min_score);

        let mut out: Vec<Decode> = Vec::new();
        for cand in &heap[..n] {
            let (msg, status) = decode_candidate(&wf, cand, max_iterations);
            let Some(message) = msg else { continue };
            if out.iter().any(|d| d.message.payload == message.payload) {
                continue;
            }
            let bin = self.min_bin as f32
                + cand.freq_offset as f32
                + cand.freq_sub as f32 / self.freq_osr as f32;
            out.push(Decode {
                message,
                freq_hz: bin / self.symbol_period,
                time_sec: (cand.time_offset as f32 + cand.time_sub as f32 / self.time_osr as f32)
                    * self.symbol_period,
                score: cand.score,
                status,
            });
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{FT8_COSTAS_PATTERN, FT8_GRAY_MAP, FT8_ND, LDPC_K_BYTES};
    use crate::message::NoHash;
    use crate::{crc, ldpc};

    /// Synthesise the audio a real transmitter would send: 79 tones, each one
    /// symbol long, at 6.25 Hz spacing, phase-continuous.
    fn synth_audio(payload: &[u8; 10], base_hz: f32, sample_rate: f32, noise: f32) -> Vec<f32> {
        let mut a91 = [0u8; LDPC_K_BYTES];
        crc::add(payload, &mut a91);
        let codeword = ldpc::encode(&a91);

        let mut tones = [0u8; 79];
        for (i, &t) in FT8_COSTAS_PATTERN.iter().enumerate() {
            tones[i] = t;
            tones[36 + i] = t;
            tones[72 + i] = t;
        }
        for k in 0..FT8_ND {
            let bits = (codeword[3 * k] << 2) | (codeword[3 * k + 1] << 1) | codeword[3 * k + 2];
            tones[k + if k < 29 { 7 } else { 14 }] = FT8_GRAY_MAP[bits as usize];
        }

        let spacing = 1.0 / FT8_SYMBOL_PERIOD; // 6.25 Hz
        let block = (sample_rate * FT8_SYMBOL_PERIOD) as usize;
        let mut out = Vec::with_capacity(79 * block);
        let mut phase = 0.0f32;
        let mut rng = 0x2545F491_4F6CDD1Du64;
        for &t in tones.iter() {
            let f = base_hz + t as f32 * spacing;
            let dphi = 2.0 * core::f32::consts::PI * f / sample_rate;
            for _ in 0..block {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let n = ((rng >> 40) as i32 as f32 / 8_388_608.0 - 1.0) * noise;
                out.push(phase.sin() + n);
                phase += dphi;
                if phase > core::f32::consts::TAU {
                    phase -= core::f32::consts::TAU;
                }
            }
        }
        out
    }

    /// The whole chain, from a waveform: FFT, sync search, soft bits, LDPC,
    /// CRC, message. This is the test that would catch a wrong window, a wrong
    /// FFT size, or a bin-indexing error — none of which the synthetic
    /// waterfall tests can see.
    #[test]
    fn decodes_a_synthesised_transmission() {
        let msg = Message::encode_std("CQ", "K1ABC", "FN42").unwrap();
        let cfg = MonitorConfig::default();
        let audio = synth_audio(&msg.payload, 1000.0, cfg.sample_rate, 0.0);

        let mut mon = Monitor::new(cfg);
        for chunk in audio.chunks(mon.block_size()) {
            mon.process(chunk);
        }
        assert!(
            mon.num_blocks() >= 79,
            "only got {} blocks",
            mon.num_blocks()
        );

        let decodes = mon.decode_all(30, 10, 30);
        assert_eq!(
            decodes.len(),
            1,
            "expected exactly one decode, got {decodes:?}"
        );

        let d = &decodes[0];
        let m = d.message.decode_std(&mut NoHash).unwrap();
        assert_eq!(&*m.call_to, "CQ");
        assert_eq!(&*m.call_de, "K1ABC");
        assert_eq!(&*m.extra, "FN42");
        // Frequency should land within a bin of where we put it.
        assert!((d.freq_hz - 1000.0).abs() < 7.0, "freq {} Hz", d.freq_hz);
        assert!(d.time_sec.abs() < 0.2, "time {} s", d.time_sec);
    }

    /// Two stations at once, which is the normal state of an FT8 slot and the
    /// thing a single-channel demodulator could not do.
    #[test]
    fn decodes_two_overlapping_signals() {
        let a = Message::encode_std("CQ", "K1ABC", "FN42").unwrap();
        let b = Message::encode_std("CQ", "W9XYZ", "EN37").unwrap();
        let cfg = MonitorConfig::default();
        let sig_a = synth_audio(&a.payload, 800.0, cfg.sample_rate, 0.0);
        let sig_b = synth_audio(&b.payload, 1500.0, cfg.sample_rate, 0.0);
        let mixed: Vec<f32> = sig_a.iter().zip(&sig_b).map(|(x, y)| x + y).collect();

        let mut mon = Monitor::new(cfg);
        for chunk in mixed.chunks(mon.block_size()) {
            mon.process(chunk);
        }
        let decodes = mon.decode_all(40, 10, 30);
        assert_eq!(decodes.len(), 2, "got {decodes:?}");

        let mut calls: Vec<_> = decodes
            .iter()
            .map(|d| {
                d.message
                    .decode_std(&mut NoHash)
                    .unwrap()
                    .call_de
                    .as_str()
                    .to_string()
            })
            .collect();
        calls.sort();
        assert_eq!(calls, ["K1ABC", "W9XYZ"]);
    }

    /// Survives noise well above the signal — FT8's whole point is decoding
    /// below the noise floor.
    #[test]
    fn decodes_under_noise() {
        let msg = Message::encode_std("K1ABC", "W9XYZ", "-15").unwrap();
        let cfg = MonitorConfig::default();
        let audio = synth_audio(&msg.payload, 1200.0, cfg.sample_rate, 3.0);

        let mut mon = Monitor::new(cfg);
        for chunk in audio.chunks(mon.block_size()) {
            mon.process(chunk);
        }
        let decodes = mon.decode_all(40, 10, 40);
        assert!(!decodes.is_empty(), "nothing decoded under noise");
        let m = decodes[0].message.decode_std(&mut NoHash).unwrap();
        assert_eq!(&*m.extra, "-15");
    }

    /// SNR has to track reality, not just produce a number: more noise must
    /// read as lower SNR, monotonically, and the values must land in the range
    /// an operator recognises (WSJT-X spans about -24..+15 dB).
    #[test]
    fn snr_falls_as_noise_rises() {
        let msg = Message::encode_std("CQ", "K1ABC", "FN42").unwrap();
        let cfg = MonitorConfig::default();
        let mut snrs = Vec::new();
        for noise in [0.05f32, 0.3, 1.0, 3.0] {
            let audio = synth_audio(&msg.payload, 1200.0, cfg.sample_rate, noise);
            let mut mon = Monitor::new(cfg);
            for chunk in audio.chunks(mon.block_size()) {
                mon.process(chunk);
            }
            let d = mon.decode_all(40, 10, 40);
            assert!(!d.is_empty(), "no decode at noise {noise}");
            snrs.push(d[0].status.snr_db);
        }
        for w in snrs.windows(2) {
            assert!(w[0] > w[1], "SNR did not fall with added noise: {snrs:?}");
        }
        // The noisiest case is a genuinely weak signal; it should read like one.
        assert!(
            snrs[3] < snrs[0] - 10.0,
            "SNR barely moved across 36 dB of noise: {snrs:?}"
        );
        assert!(
            snrs.iter().all(|s| (-30.0..40.0).contains(s)),
            "implausible SNR: {snrs:?}"
        );
    }

    /// Silence must produce nothing at all.
    #[test]
    fn silence_decodes_to_nothing() {
        let cfg = MonitorConfig::default();
        let mut mon = Monitor::new(cfg);
        let silence = vec![0.0f32; mon.block_size()];
        for _ in 0..93 {
            mon.process(&silence);
        }
        assert!(mon.decode_all(30, 10, 20).is_empty());
    }

    /// A short frame would shift every symbol after it, so it is refused
    /// rather than padded.
    #[test]
    fn rejects_a_short_frame() {
        let mut mon = Monitor::new(MonitorConfig::default());
        mon.process(&[0.0; 100]);
        assert_eq!(mon.num_blocks(), 0);
    }
}
