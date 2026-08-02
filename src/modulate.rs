//! Message → audio. Ported from ft8_lib's `gen_ft8` demo.
//!
//! The naive way to transmit 8-FSK is to hold each tone for its symbol period
//! and step to the next. That decodes fine, and it is what this crate's own
//! tests used to synthesise signals — but it is not fit to put on the air. An
//! instantaneous frequency step has infinite bandwidth in principle and very
//! wide sidebands in practice, and on FT8 the nearest other station is 6.25 Hz
//! away.
//!
//! So the frequency is *shaped*: the symbol sequence is convolved with a
//! Gaussian pulse, giving a frequency curve that glides between tones over
//! about a symbol instead of jumping. Integrate that curve to get phase, take
//! the sine. The BT parameter sets how gentle the glide is — 2.0 for FT8, 1.0
//! for FT4, which is tighter because its symbols are shorter.

use crate::constants::{FT4_SYMBOL_PERIOD, FT8_SYMBOL_PERIOD};
use crate::decode::{ft4_tones, ft8_tones, Protocol};
use crate::message::PAYLOAD_BYTES;

/// π·√(2/ln 2), the constant relating BT to the Gaussian's width.
const GFSK_K: f32 = 5.336_446;
const FT8_BT: f32 = 2.0;
const FT4_BT: f32 = 1.0;

/// Error function, Abramowitz & Stegun 7.1.26. Accurate to ~1.5e-7, which is
/// far beyond what a pulse shape needs, and avoids taking a dependency on
/// `libm` for one call.
fn erf(x: f32) -> f32 {
    const P: f32 = 0.327_591_1;
    const A: [f32; 5] = [
        0.254_829_6,
        -0.284_496_74,
        1.421_413_7,
        -1.453_152,
        1.061_405_4,
    ];
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + P * x);
    let poly = t * (A[0] + t * (A[1] + t * (A[2] + t * (A[3] + t * A[4]))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// The Gaussian smoothing pulse, truncated at three symbol periods — beyond
/// that it contributes nothing measurable.
fn gfsk_pulse(n_spsym: usize, bt: f32) -> Vec<f32> {
    (0..3 * n_spsym)
        .map(|i| {
            let t = i as f32 / n_spsym as f32 - 1.5;
            (erf(GFSK_K * bt * (t + 0.5)) - erf(GFSK_K * bt * (t - 0.5))) / 2.0
        })
        .collect()
}

/// Synthesise the audio for a tone sequence.
///
/// `f0_hz` is the audio frequency of tone 0 — the bottom of the signal. On a
/// radio in USB this lands at `dial + f0_hz`, so 1500 Hz is the conventional
/// middle-of-the-passband choice.
fn synth_gfsk(
    symbols: &[u8],
    f0_hz: f32,
    bt: f32,
    symbol_period: f32,
    sample_rate: f32,
) -> Vec<f32> {
    let n_spsym = (0.5 + sample_rate * symbol_period) as usize;
    let n_wave = symbols.len() * n_spsym;
    let dphi_peak = 2.0 * core::f32::consts::PI / n_spsym as f32;

    // Frequency curve, with a symbol of margin at each end for the pulse tails.
    let mut dphi = vec![2.0 * core::f32::consts::PI * f0_hz / sample_rate; n_wave + 2 * n_spsym];
    let pulse = gfsk_pulse(n_spsym, bt);

    for (i, &sym) in symbols.iter().enumerate() {
        let ib = i * n_spsym;
        for (j, &p) in pulse.iter().enumerate() {
            dphi[ib + j] += dphi_peak * sym as f32 * p;
        }
    }
    // Extend the first and last symbols outward, so the curve starts and ends
    // settled rather than sliding in from zero.
    let last = *symbols.last().unwrap_or(&0) as f32;
    let first = *symbols.first().unwrap_or(&0) as f32;
    for j in 0..2 * n_spsym {
        dphi[j] += dphi_peak * pulse[j + n_spsym] * first;
        dphi[j + symbols.len() * n_spsym] += dphi_peak * pulse[j] * last;
    }

    let mut signal = Vec::with_capacity(n_wave);
    let mut phi = 0.0f32;
    for k in 0..n_wave {
        signal.push(phi.sin());
        phi = (phi + dphi[k + n_spsym]) % (2.0 * core::f32::consts::PI);
    }

    // Ramp the envelope over an eighth of a symbol at each end. Without it the
    // transmitter starts and stops on a step, which is a click across the band
    // — the same reason a key-click filter exists on CW.
    let n_ramp = n_spsym / 8;
    for i in 0..n_ramp {
        let env =
            (1.0 - (2.0 * core::f32::consts::PI * i as f32 / (2.0 * n_ramp as f32)).cos()) / 2.0;
        signal[i] *= env;
        signal[n_wave - 1 - i] *= env;
    }
    signal
}

/// Turn a message payload into transmittable audio.
///
/// Returns 12.64 s of samples for FT8, 5.04 s for FT4 — the transmission
/// itself, not the whole slot. The caller is responsible for starting it at
/// the UTC slot boundary and for keying the radio; both are station-specific
/// and do not belong in a library.
///
/// ```
/// use ft8::{modulate, Message, Protocol};
///
/// let msg = Message::encode_std("CQ", "K1ABC", "FN42").unwrap();
/// let audio = modulate(&msg.payload, Protocol::Ft8, 1500.0, 12000.0);
/// assert_eq!(audio.len(), 151_680); // 79 symbols x 0.16 s at 12 kHz
/// ```
pub fn modulate(
    payload: &[u8; PAYLOAD_BYTES],
    protocol: Protocol,
    f0_hz: f32,
    sample_rate: f32,
) -> Vec<f32> {
    match protocol {
        Protocol::Ft8 => synth_gfsk(
            &ft8_tones(payload),
            f0_hz,
            FT8_BT,
            FT8_SYMBOL_PERIOD,
            sample_rate,
        ),
        Protocol::Ft4 => synth_gfsk(
            &ft4_tones(payload),
            f0_hz,
            FT4_BT,
            FT4_SYMBOL_PERIOD,
            sample_rate,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Message;
    use crate::monitor::{Monitor, MonitorConfig};
    use rustfft::{num_complex::Complex, FftPlanner};

    fn decode_back(audio: &[f32], protocol: Protocol) -> Vec<String> {
        let cfg = MonitorConfig {
            protocol,
            ..MonitorConfig::default()
        };
        let mut mon = Monitor::new(cfg);
        for chunk in audio.chunks(mon.block_size()) {
            mon.process(chunk);
        }
        mon.decode_all(40, 10, 40)
            .iter()
            .map(|d| d.text().to_string())
            .collect()
    }

    /// The only test that matters: what we transmit, we can receive.
    #[test]
    fn modulated_ft8_decodes_back() {
        for (to, de, extra) in [
            ("CQ", "K1ABC", "FN42"),
            ("K1ABC", "W9XYZ", "-11"),
            ("K1ABC", "W9XYZ", "RR73"),
        ] {
            let msg = Message::encode_std(to, de, extra).unwrap();
            let audio = modulate(&msg.payload, Protocol::Ft8, 1500.0, 12000.0);
            let got = decode_back(&audio, Protocol::Ft8);
            let want = format!("{to} {de} {extra}").trim().to_string();
            assert!(
                got.iter().any(|g| g == &want),
                "wanted {want:?}, got {got:?}"
            );
        }
    }

    #[test]
    fn modulated_ft4_decodes_back() {
        let msg = Message::encode_std("CQ", "K1ABC", "FN42").unwrap();
        let audio = modulate(&msg.payload, Protocol::Ft4, 1500.0, 12000.0);
        let got = decode_back(&audio, Protocol::Ft4);
        assert!(got.iter().any(|g| g == "CQ K1ABC FN42"), "got {got:?}");
    }

    /// Occupied bandwidth: energy more than 100 Hz outside the signal, relative
    /// to the total. This is the entire reason for GFSK, so it is worth
    /// measuring rather than assuming — and worth measuring against the naive
    /// modulator to show the shaping is doing something.
    fn splatter_db(audio: &[f32], f0: f32, rate: f32) -> f32 {
        let n = 1 << 15;
        let mut buf: Vec<Complex<f32>> = audio
            .iter()
            .take(n)
            .enumerate()
            .map(|(i, &s)| {
                // Hann window, or the analysis leaks worse than the signal.
                let w = 0.5 * (1.0 - (2.0 * core::f32::consts::PI * i as f32 / n as f32).cos());
                Complex::new(s * w, 0.0)
            })
            .collect();
        buf.resize(n, Complex::new(0.0, 0.0));
        FftPlanner::<f32>::new()
            .plan_fft_forward(n)
            .process(&mut buf);

        let bin_hz = rate / n as f32;
        let (lo, hi) = (f0 - 100.0, f0 + 8.0 * 6.25 + 100.0);
        let (mut inside, mut outside) = (0.0f64, 0.0f64);
        for (i, c) in buf.iter().take(n / 2).enumerate() {
            let f = i as f32 * bin_hz;
            if f < 50.0 {
                continue; // ignore DC
            }
            let p = (c.re * c.re + c.im * c.im) as f64;
            if f >= lo && f <= hi {
                inside += p;
            } else {
                outside += p;
            }
        }
        10.0 * (outside / inside.max(1e-30)).log10() as f32
    }

    #[test]
    fn gfsk_is_far_cleaner_than_hard_switching() {
        let msg = Message::encode_std("CQ", "K1ABC", "FN42").unwrap();
        let rate = 12000.0;
        let shaped = modulate(&msg.payload, Protocol::Ft8, 1500.0, rate);

        // The naive version: hold each tone, step to the next, phase-continuous.
        let tones = ft8_tones(&msg.payload);
        let n_spsym = (rate * FT8_SYMBOL_PERIOD) as usize;
        let mut naive = Vec::with_capacity(tones.len() * n_spsym);
        let mut phase = 0.0f32;
        for &t in tones.iter() {
            let dphi = 2.0 * core::f32::consts::PI * (1500.0 + t as f32 * 6.25) / rate;
            for _ in 0..n_spsym {
                naive.push(phase.sin());
                phase = (phase + dphi) % (2.0 * core::f32::consts::PI);
            }
        }

        let shaped_db = splatter_db(&shaped, 1500.0, rate);
        let naive_db = splatter_db(&naive, 1500.0, rate);
        println!("out-of-band energy: GFSK {shaped_db:.1} dB, hard-switched {naive_db:.1} dB");
        assert!(
            shaped_db < naive_db - 10.0,
            "GFSK should be far cleaner: shaped {shaped_db:.1} dB vs naive {naive_db:.1} dB"
        );
        // And in absolute terms it should be well down, not merely better.
        assert!(
            shaped_db < -40.0,
            "shaped splatter {shaped_db:.1} dB is too high"
        );
    }

    #[test]
    fn starts_and_ends_at_silence() {
        let msg = Message::encode_std("CQ", "K1ABC", "FN42").unwrap();
        let a = modulate(&msg.payload, Protocol::Ft8, 1500.0, 12000.0);
        // The envelope ramp means no step into or out of the transmission.
        assert!(a[0].abs() < 0.01, "starts at {}", a[0]);
        assert!(a[a.len() - 1].abs() < 0.05, "ends at {}", a[a.len() - 1]);
        assert!(a.iter().all(|s| s.abs() <= 1.0), "clipped");
    }
}
