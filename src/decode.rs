//! Sync search and soft-bit extraction over a spectrogram. Ported from
//! ft8_lib's `decode.c`.
//!
//! The input is a "waterfall": FFT magnitudes for the whole slot, laid out as
//! `[block][time_sub][freq_sub][bin]`. The `monitor` module builds one from
//! audio; this module never touches audio itself.
//!
//! Bins are spaced 6.25 Hz apart — exactly the FT8 tone spacing — and there is
//! one block per 0.16 s symbol, so the grid *is* the FSK demodulator: each cell
//! is "how much energy was in this tone during this symbol". Magnitudes are
//! stored as `dB * 2 + 240` in a byte, a quarter the size of `f32` at no useful
//! loss, since the search only ever compares neighbouring cells.

use crate::constants::{
    FT4_COSTAS_PATTERN, FT4_GRAY_MAP, FT4_LENGTH_SYNC, FT4_ND, FT4_NN, FT4_NUM_SYNC,
    FT4_SYMBOL_PERIOD, FT4_SYNC_OFFSET, FT4_XOR_SEQUENCE, FT8_COSTAS_PATTERN, FT8_GRAY_MAP,
    FT8_LENGTH_SYNC, FT8_ND, FT8_NN, FT8_NUM_SYNC, FT8_SYMBOL_PERIOD, FT8_SYNC_OFFSET, LDPC_K,
    LDPC_K_BYTES, LDPC_N,
};
use crate::message::{Message, PAYLOAD_BYTES};
use crate::{crc, ldpc};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Protocol {
    Ft8,
    Ft4,
}

/// FFT magnitudes for one slot, `[block][time_sub][freq_sub][bin]`.
///
/// Each stored byte is a magnitude in half-decibels offset by 120, so
/// `dB = byte * 0.5 - 120`. That is ft8_lib's packing and it is kept because it
/// makes the whole spectrogram a quarter the size of `f32` at no useful loss —
/// the sync search only ever compares neighbouring bins.
#[derive(Clone, Copy, Debug)]
pub struct Waterfall<'a> {
    pub mag: &'a [u8],
    /// Symbols stored.
    pub num_blocks: usize,
    /// Bins per symbol, in units of the 6.25 Hz tone spacing.
    pub num_bins: usize,
    /// Time subdivisions per symbol.
    pub time_osr: usize,
    /// Frequency subdivisions per tone spacing.
    pub freq_osr: usize,
    pub protocol: Protocol,
}

impl Waterfall<'_> {
    #[inline]
    pub fn block_stride(&self) -> usize {
        self.time_osr * self.freq_osr * self.num_bins
    }
}

/// A possible message start, in time and frequency.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Candidate {
    pub score: i16,
    pub time_offset: i16,
    pub freq_offset: i16,
    pub time_sub: u8,
    pub freq_sub: u8,
}

/// How a decode attempt went, whether or not it succeeded.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct DecodeStatus {
    pub ldpc_errors: usize,
    pub crc_extracted: u16,
    pub crc_calculated: u16,
    /// Estimated SNR in dB over a 2500 Hz reference bandwidth, as WSJT-X
    /// reports. Only meaningful once the decode succeeded, since the estimate
    /// works backwards from the recovered message. NaN otherwise.
    pub snr_db: f32,
}

impl Default for DecodeStatus {
    fn default() -> Self {
        Self {
            ldpc_errors: 0,
            crc_extracted: 0,
            crc_calculated: 0,
            snr_db: f32::NAN,
        }
    }
}

/// Index of the candidate's symbol 0. Signed because time offsets run negative:
/// a message whose first Costas group fell before the recording still has all
/// its data bits, so it is worth scoring.
#[inline]
fn cand_base(wf: &Waterfall, c: &Candidate) -> isize {
    let mut off = c.time_offset as isize;
    off = off * wf.time_osr as isize + c.time_sub as isize;
    off = off * wf.freq_osr as isize + c.freq_sub as isize;
    off * wf.num_bins as isize + c.freq_offset as isize
}

/// Sync score: how much the expected Costas tone stands out from its immediate
/// neighbours in frequency and time.
///
/// ft8_lib also ships a "weighted difference against all 8 tones" variant,
/// commented out with the note that it does not work as well. Comparing only
/// neighbours is what makes the score robust to a sloping noise floor, which is
/// the normal state of an HF band.
// `m` and `k` index two different Costas tables depending on protocol, and are
// also used in the block arithmetic, so iterating the tables directly does not
// express this.
#[allow(clippy::needless_range_loop)]
fn sync_score(wf: &Waterfall, c: &Candidate) -> i32 {
    let (num_sync, length_sync, sync_offset, max_tone) = match wf.protocol {
        Protocol::Ft8 => (FT8_NUM_SYNC, FT8_LENGTH_SYNC, FT8_SYNC_OFFSET, 7usize),
        Protocol::Ft4 => (FT4_NUM_SYNC, FT4_LENGTH_SYNC, FT4_SYNC_OFFSET, 3usize),
    };
    let stride = wf.block_stride() as isize;
    let base = cand_base(wf, c);
    let mut score = 0i32;
    let mut num_average = 0i32;

    for m in 0..num_sync {
        for k in 0..length_sync {
            let block = match wf.protocol {
                Protocol::Ft8 => sync_offset * m + k,
                // FT4's first symbol is a ramp, so its sync groups start at 1.
                Protocol::Ft4 => 1 + sync_offset * m + k,
            };
            let block_abs = c.time_offset as isize + block as isize;
            if block_abs < 0 {
                continue;
            }
            if block_abs >= wf.num_blocks as isize {
                break;
            }
            let p = base + block as isize * stride;
            let sm = match wf.protocol {
                Protocol::Ft8 => FT8_COSTAS_PATTERN[k] as isize,
                Protocol::Ft4 => FT4_COSTAS_PATTERN[m][k] as isize,
            };
            // The bounds above guarantee p + sm is inside `mag`; the neighbour
            // reads are each additionally guarded.
            let at = |i: isize| -> i32 { wf.mag[i as usize] as i32 };
            let here = at(p + sm);

            if sm > 0 {
                score += here - at(p + sm - 1);
                num_average += 1;
            }
            if (sm as usize) < max_tone {
                score += here - at(p + sm + 1);
                num_average += 1;
            }
            if k > 0 && block_abs > 0 {
                score += here - at(p + sm - stride);
                num_average += 1;
            }
            if k + 1 < length_sync && block_abs + 1 < wf.num_blocks as isize {
                score += here - at(p + sm + stride);
                num_average += 1;
            }
        }
    }

    if num_average > 0 {
        score /= num_average;
    }
    score
}

fn heapify_down(heap: &mut [Candidate]) {
    let n = heap.len();
    let mut current = 0;
    loop {
        let (left, right) = (2 * current + 1, 2 * current + 2);
        let mut smallest = current;
        if left < n && heap[left].score < heap[smallest].score {
            smallest = left;
        }
        if right < n && heap[right].score < heap[smallest].score {
            smallest = right;
        }
        if smallest == current {
            break;
        }
        heap.swap(smallest, current);
        current = smallest;
    }
}

fn heapify_up(heap: &mut [Candidate]) {
    let mut current = heap.len().saturating_sub(1);
    while current > 0 {
        let parent = (current - 1) / 2;
        if heap[current].score >= heap[parent].score {
            break;
        }
        heap.swap(parent, current);
        current = parent;
    }
}

/// Score every (time, frequency) offset and keep the best `heap.len()`.
///
/// Returns how many candidates were found; on return `heap[..n]` is sorted by
/// descending score. A min-heap is used so displacing the worst survivor is
/// O(log n) rather than a rescan — this loop runs over roughly
/// `30 × num_bins × time_osr × freq_osr` positions and is the bulk of decode
/// time before LDPC.
pub fn find_candidates(wf: &Waterfall, heap: &mut [Candidate], min_score: i16) -> usize {
    if heap.is_empty() || wf.num_bins == 0 {
        return 0;
    }
    let num_tones = match wf.protocol {
        Protocol::Ft8 => 8,
        Protocol::Ft4 => 4,
    };
    let capacity = heap.len();
    let mut heap_size = 0usize;

    for time_sub in 0..wf.time_osr as u8 {
        for freq_sub in 0..wf.freq_osr as u8 {
            for time_offset in -10i16..20 {
                for freq_offset in 0..(wf.num_bins as i16 - num_tones + 1) {
                    let mut cand = Candidate {
                        score: 0,
                        time_offset,
                        freq_offset,
                        time_sub,
                        freq_sub,
                    };
                    cand.score =
                        sync_score(wf, &cand).clamp(i16::MIN as i32, i16::MAX as i32) as i16;

                    if cand.score < min_score {
                        continue;
                    }
                    if heap_size == capacity && cand.score > heap[0].score {
                        heap_size -= 1;
                        heap[0] = heap[heap_size];
                        heapify_down(&mut heap[..heap_size]);
                    }
                    if heap_size < capacity {
                        heap[heap_size] = cand;
                        heap_size += 1;
                        heapify_up(&mut heap[..heap_size]);
                    }
                }
            }
        }
    }

    // Heapsort in place: repeatedly move the smallest to the end, leaving the
    // array in descending order.
    let mut len_unsorted = heap_size;
    while len_unsorted > 1 {
        heap.swap(len_unsorted - 1, 0);
        len_unsorted -= 1;
        heapify_down(&mut heap[..len_unsorted]);
    }
    heap_size
}

#[inline]
fn mag_db(v: u8) -> f32 {
    v as f32 * 0.5 - 120.0
}

#[inline]
fn max2(a: f32, b: f32) -> f32 {
    if a >= b {
        a
    } else {
        b
    }
}

#[inline]
fn max4(a: f32, b: f32, c: f32, d: f32) -> f32 {
    max2(max2(a, b), max2(c, d))
}

/// Soft bits for one 8-FSK symbol: for each of the 3 bits, the strongest tone
/// that would set it minus the strongest that would clear it.
fn ft8_extract_symbol(mag: &[u8], out: &mut [f32]) {
    let mut s2 = [0.0f32; 8];
    for (j, s) in s2.iter_mut().enumerate() {
        *s = mag_db(mag[FT8_GRAY_MAP[j] as usize]);
    }
    out[0] = max4(s2[4], s2[5], s2[6], s2[7]) - max4(s2[0], s2[1], s2[2], s2[3]);
    out[1] = max4(s2[2], s2[3], s2[6], s2[7]) - max4(s2[0], s2[1], s2[4], s2[5]);
    out[2] = max4(s2[1], s2[3], s2[5], s2[7]) - max4(s2[0], s2[2], s2[4], s2[6]);
}

/// Soft bits for one 4-FSK symbol (FT4).
fn ft4_extract_symbol(mag: &[u8], out: &mut [f32]) {
    let mut s2 = [0.0f32; 4];
    for (j, s) in s2.iter_mut().enumerate() {
        *s = mag_db(mag[FT4_GRAY_MAP[j] as usize]);
    }
    out[0] = max2(s2[2], s2[3]) - max2(s2[0], s2[1]);
    out[1] = max2(s2[1], s2[3]) - max2(s2[0], s2[2]);
}

fn extract_likelihood(wf: &Waterfall, c: &Candidate, log174: &mut [f32; LDPC_N]) {
    log174.fill(0.0);
    let stride = wf.block_stride() as isize;
    let base = cand_base(wf, c);
    let (nd, bits_per_sym) = match wf.protocol {
        Protocol::Ft8 => (FT8_ND, 3),
        Protocol::Ft4 => (FT4_ND, 2),
    };

    for k in 0..nd {
        // Data symbols are interleaved with the sync groups, so symbol k of the
        // payload sits at a channel-symbol index that jumps past each one.
        let sym_idx = match wf.protocol {
            Protocol::Ft8 => k + if k < 29 { 7 } else { 14 },
            Protocol::Ft4 => {
                k + if k < 29 {
                    5
                } else if k < 58 {
                    9
                } else {
                    13
                }
            }
        };
        let bit_idx = bits_per_sym * k;
        let block = c.time_offset as isize + sym_idx as isize;
        if block < 0 || block >= wf.num_blocks as isize {
            continue; // leave these bits at zero: "no information"
        }
        let p = (base + sym_idx as isize * stride) as usize;
        let window = &wf.mag[p..];
        match wf.protocol {
            Protocol::Ft8 => ft8_extract_symbol(window, &mut log174[bit_idx..bit_idx + 3]),
            Protocol::Ft4 => ft4_extract_symbol(window, &mut log174[bit_idx..bit_idx + 2]),
        }
    }
}

/// Scale the soft bits to the variance the LDPC decoder expects.
///
/// The 24.0 is ft8_lib's, found experimentally. It matters: the decoder's
/// tanh/atanh approximations are only well-conditioned over a particular range,
/// so feeding raw dB differences converges far worse than feeding normalised
/// ones, even though the sign pattern is identical.
fn normalize_logl(log174: &mut [f32; LDPC_N]) {
    let mut sum = 0.0f32;
    let mut sum2 = 0.0f32;
    for &v in log174.iter() {
        sum += v;
        sum2 += v * v;
    }
    let inv_n = 1.0 / LDPC_N as f32;
    let variance = (sum2 - sum * sum * inv_n) * inv_n;
    if variance <= 0.0 {
        return;
    }
    let norm = sqrtf(24.0 / variance);
    for v in log174.iter_mut() {
        *v *= norm;
    }
}

/// Newton–Raphson square root.
///
/// `f32::sqrt` lives in `std`, and pulling in `libm` for one call would put a
/// dependency on a crate that otherwise has none.
///
/// The seed matters more than the iteration count: halving the IEEE-754
/// exponent lands within a factor of ~1.4 of the answer, which four Newton
/// steps polish to the limit of `f32`. Seeding with `x` itself instead needs
/// well over six steps once `x` is large — sqrt(100) was still wrong in the
/// fifth digit.
fn sqrtf(x: f32) -> f32 {
    if x <= 0.0 || !x.is_finite() {
        return 0.0;
    }
    let mut g = f32::from_bits((x.to_bits() + (127 << 23)) >> 1);
    for _ in 0..4 {
        g = 0.5 * (g + x / g);
    }
    g
}

fn pack_bits(bits: &[u8], num_bits: usize, packed: &mut [u8]) {
    packed.fill(0);
    for i in 0..num_bits {
        if bits[i] != 0 {
            packed[i / 8] |= 0x80 >> (i % 8);
        }
    }
}

/// The 79 channel symbols a payload transmits as: three Costas sync groups
/// interleaved with two 29-symbol data blocks.
///
/// Public because it is what a transmitter needs, and because the SNR estimate
/// below works by re-deriving what *should* have been on the air and comparing.
pub fn ft8_tones(payload: &[u8; PAYLOAD_BYTES]) -> [u8; FT8_NN] {
    let mut a91 = [0u8; LDPC_K_BYTES];
    crc::add(payload, &mut a91);
    let codeword = ldpc::encode(&a91);

    let mut tones = [0u8; FT8_NN];
    for (i, &t) in FT8_COSTAS_PATTERN.iter().enumerate() {
        tones[i] = t;
        tones[36 + i] = t;
        tones[72 + i] = t;
    }
    for k in 0..FT8_ND {
        let bits = (codeword[3 * k] << 2) | (codeword[3 * k + 1] << 1) | codeword[3 * k + 2];
        tones[k + if k < 29 { 7 } else { 14 }] = FT8_GRAY_MAP[bits as usize];
    }
    tones
}

/// The 105 channel symbols an FT4 payload transmits as: a ramp, four 4-symbol
/// Costas groups interleaved with three data blocks, and a closing ramp.
///
/// Note the payload is XORed with a fixed sequence *before* the CRC and FEC are
/// computed — FT4 does this so that CQ, which is mostly zeros, does not put a
/// long run of one tone on the air.
pub fn ft4_tones(payload: &[u8; PAYLOAD_BYTES]) -> [u8; FT4_NN] {
    let mut scrambled = *payload;
    for (p, x) in scrambled.iter_mut().zip(FT4_XOR_SEQUENCE.iter()) {
        *p ^= x;
    }
    let mut a91 = [0u8; LDPC_K_BYTES];
    crc::add(&scrambled, &mut a91);
    let codeword = ldpc::encode(&a91);

    let mut tones = [0u8; FT4_NN];
    for m in 0..FT4_NUM_SYNC {
        for k in 0..FT4_LENGTH_SYNC {
            tones[1 + FT4_SYNC_OFFSET * m + k] = FT4_COSTAS_PATTERN[m][k];
        }
    }
    for k in 0..FT4_ND {
        let bits = (codeword[2 * k] << 1) | codeword[2 * k + 1];
        let sym = k + if k < 29 {
            5
        } else if k < 58 {
            9
        } else {
            13
        };
        tones[sym] = FT4_GRAY_MAP[bits as usize];
    }
    tones
}

/// Signal-to-noise in dB, referenced to a 2500 Hz noise bandwidth — the
/// convention WSJT-X reports and everyone reads.
///
/// Once a message has decoded we know exactly which tone was sent in every
/// symbol, so the signal level is the mean power in those bins.
///
/// The noise floor is measured from a guard band either side of the signal's
/// eight tones — NOT from the other seven tones of each symbol, which is the
/// obvious choice and is wrong: those bins sit 6.25 Hz away and are full of
/// the signal's own spectral leakage. Leakage scales *with* the signal, so
/// that estimator returns nearly the same SNR across 36 dB of added noise.
///
/// The floor is a median rather than a mean, so a neighbouring station in the
/// guard band raises it barely at all — and on FT8 there is nearly always a
/// neighbouring station.
///
/// The −26 dB converts from the 6.25 Hz bin the measurement is made in to the
/// 2500 Hz reference WSJT-X reports against: `10·log10(6.25/2500)`. Treat the
/// result as good to a dB or two rather than calibrated; it comes from
/// magnitudes already quantised to half a dB.
fn estimate_snr(wf: &Waterfall, cand: &Candidate, payload: &[u8; PAYLOAD_BYTES]) -> f32 {
    let (tones, num_tones, symbol_period): (&[u8], i16, f32) = match wf.protocol {
        Protocol::Ft8 => (&ft8_tones(payload)[..], 8, FT8_SYMBOL_PERIOD),
        Protocol::Ft4 => (&ft4_tones(payload)[..], 4, FT4_SYMBOL_PERIOD),
    };
    let stride = wf.block_stride() as isize;
    let base = cand_base(wf, cand);
    // The tones span bins 0..num_tones-1 from the candidate; 4 bins of guard
    // clears Hann leakage, and 20 bins of window either side is enough to
    // median over without wandering into unrelated spectrum.
    const GUARD: i16 = 4;
    const WINDOW: i16 = 20;
    let last = num_tones - 1;

    let mut sig = 0.0f64;
    let mut n_sig = 0usize;
    let mut hist = [0u32; 256];
    let mut n_noise = 0usize;

    for (block, &tone) in tones.iter().enumerate() {
        let abs = cand.time_offset as isize + block as isize;
        if abs < 0 || abs >= wf.num_blocks as isize {
            continue;
        }
        let p = base + block as isize * stride;
        let at = |off: isize| -> Option<u8> {
            let i = p + off;
            if i < 0 || i as usize >= wf.mag.len() {
                None
            } else {
                Some(wf.mag[i as usize])
            }
        };
        if let Some(v) = at(tone as isize) {
            sig += 10f64.powf(mag_db(v) as f64 / 10.0);
            n_sig += 1;
        }
        for off in -(GUARD + WINDOW)..=(last + GUARD + WINDOW) {
            if (-GUARD..=(last + GUARD)).contains(&off) {
                continue; // the signal and its guard
            }
            let bin = cand.freq_offset + off;
            if bin < 0 || bin >= wf.num_bins as i16 {
                continue;
            }
            if let Some(v) = at(off as isize) {
                hist[v as usize] += 1;
                n_noise += 1;
            }
        }
    }
    if n_sig == 0 || n_noise == 0 {
        return f32::NAN;
    }
    let want = n_noise / 2;
    let mut acc = 0usize;
    let mut median = 0u8;
    for (v, &c) in hist.iter().enumerate() {
        acc += c as usize;
        if acc >= want {
            median = v as u8;
            break;
        }
    }
    let n_mean = 10f64.powf(mag_db(median) as f64 / 10.0);
    let s_mean = sig / n_sig as f64;
    // The signal bin holds signal *plus* noise.
    let signal = (s_mean - n_mean).max(n_mean * 1e-4);
    // A bin is 1/symbol_period Hz wide — 6.25 for FT8, 20.83 for FT4 — so the
    // correction to the 2500 Hz reference is not the same constant for both.
    let bw_correction = 10.0 * ((1.0 / symbol_period) / 2500.0).log10();
    let snr = 10.0 * (signal / n_mean).log10() + bw_correction as f64;
    (snr as f32).clamp(-30.0, 40.0)
}

/// Try to turn one candidate into a message.
///
/// Returns `None` when the LDPC decoder fails to converge or the CRC does not
/// match — both are normal and common, since most candidates are noise. The
/// status is returned either way so a caller can report near-misses.
pub fn decode_candidate(
    wf: &Waterfall,
    cand: &Candidate,
    max_iterations: usize,
) -> (Option<Message>, DecodeStatus) {
    let mut status = DecodeStatus::default();
    let mut log174 = [0.0f32; LDPC_N];
    extract_likelihood(wf, cand, &mut log174);
    normalize_logl(&mut log174);

    let (plain, errors) = ldpc::bp_decode(&log174, max_iterations);
    status.ldpc_errors = errors;
    if errors > 0 {
        return (None, status);
    }

    let mut a91 = [0u8; LDPC_K_BYTES];
    pack_bits(&plain, LDPC_K, &mut a91);

    status.crc_extracted = crc::extract(&a91);
    // The CRC covers the payload zero-extended from 77 to 82 bits.
    a91[9] &= 0xF8;
    a91[10] = 0;
    status.crc_calculated = crc::compute(&a91, 96 - 14);
    if status.crc_extracted != status.crc_calculated {
        return (None, status);
    }

    let mut payload = [0u8; PAYLOAD_BYTES];
    payload.copy_from_slice(&a91[..PAYLOAD_BYTES]);
    if wf.protocol == Protocol::Ft4 {
        // FT4 XORs the message with a fixed sequence so that CQ, which is
        // mostly zeros, does not transmit a long run of one tone.
        for (p, x) in payload.iter_mut().zip(FT4_XOR_SEQUENCE.iter()) {
            *p ^= x;
        }
    }
    status.snr_db = estimate_snr(wf, cand, &payload);
    (Some(Message::from_payload(payload)), status)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::NoHash;

    use super::ft8_tones as ft8_symbols;

    /// Render those symbols as a waterfall, with the transmitted tone loud and
    /// everything else at a floor. `offset_bin` places the signal away from bin
    /// zero so the sync search has to actually find it.
    fn synth_waterfall(tones: &[u8; 79], num_bins: usize, offset_bin: usize, snr: u8) -> Vec<u8> {
        let mut mag = vec![100u8; 79 * num_bins]; // ~-70 dB floor
        for (block, &t) in tones.iter().enumerate() {
            mag[block * num_bins + offset_bin + t as usize] = 100u8.saturating_add(snr);
        }
        mag
    }

    /// The whole receive chain below the FFT: find the signal in time and
    /// frequency, pull soft bits out, correct them, check the CRC, unpack.
    #[test]
    fn finds_and_decodes_a_synthetic_signal() {
        let msg = Message::encode_std("CQ", "K1ABC", "FN42").unwrap();
        let tones = ft8_symbols(&msg.payload);
        let num_bins = 40;
        let offset_bin = 11;
        let mag = synth_waterfall(&tones, num_bins, offset_bin, 60);

        let wf = Waterfall {
            mag: &mag,
            num_blocks: 79,
            num_bins,
            time_osr: 1,
            freq_osr: 1,
            protocol: Protocol::Ft8,
        };

        let mut heap = [Candidate::default(); 20];
        let n = find_candidates(&wf, &mut heap, 10);
        assert!(n > 0, "sync search found nothing");

        // The best candidate must be the signal we planted.
        assert_eq!(heap[0].freq_offset as usize, offset_bin, "wrong frequency");
        assert_eq!(heap[0].time_offset, 0, "wrong time");

        let (decoded, status) = decode_candidate(&wf, &heap[0], 30);
        let decoded = decoded.unwrap_or_else(|| panic!("decode failed: {status:?}"));
        assert_eq!(status.ldpc_errors, 0);
        assert_eq!(status.crc_extracted, status.crc_calculated);

        let std_msg = decoded.decode_std(&mut NoHash).unwrap();
        assert_eq!(&*std_msg.call_to, "CQ");
        assert_eq!(&*std_msg.call_de, "K1ABC");
        assert_eq!(&*std_msg.extra, "FN42");
    }

    /// Candidates come back strongest first — callers decode in that order and
    /// stop when they run out of budget.
    #[test]
    fn candidates_are_sorted_by_descending_score() {
        let msg = Message::encode_std("CQ", "W9XYZ", "EN37").unwrap();
        let tones = ft8_symbols(&msg.payload);
        let mag = synth_waterfall(&tones, 40, 5, 60);
        let wf = Waterfall {
            mag: &mag,
            num_blocks: 79,
            num_bins: 40,
            time_osr: 1,
            freq_osr: 1,
            protocol: Protocol::Ft8,
        };
        let mut heap = [Candidate::default(); 16];
        let n = find_candidates(&wf, &mut heap, 0);
        assert!(n > 1);
        for w in heap[..n].windows(2) {
            assert!(
                w[0].score >= w[1].score,
                "heap not sorted: {:?}",
                &heap[..n]
            );
        }
    }

    /// Pure noise must not produce a message. A decoder that "finds" signals in
    /// noise is worse than one that finds nothing.
    #[test]
    fn noise_decodes_to_nothing() {
        let mut mag = vec![0u8; 79 * 40];
        let mut s = 987654321u64;
        for m in mag.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *m = (s >> 56) as u8;
        }
        let wf = Waterfall {
            mag: &mag,
            num_blocks: 79,
            num_bins: 40,
            time_osr: 1,
            freq_osr: 1,
            protocol: Protocol::Ft8,
        };
        let mut heap = [Candidate::default(); 20];
        let n = find_candidates(&wf, &mut heap, 0);
        for cand in &heap[..n] {
            assert!(decode_candidate(&wf, cand, 20).0.is_none(), "decoded noise");
        }
    }

    /// A signal starting late still decodes: the search covers -10..+20 symbol
    /// offsets, which is what absorbs clock error between stations.
    #[test]
    fn finds_a_time_shifted_signal() {
        let msg = Message::encode_std("K1ABC", "W9XYZ", "-11").unwrap();
        let tones = ft8_symbols(&msg.payload);
        let num_bins = 32;
        let shift = 3usize;
        // Pad the front so the message starts `shift` symbols in.
        let mut mag = vec![100u8; (79 + shift) * num_bins];
        for (block, &t) in tones.iter().enumerate() {
            mag[(block + shift) * num_bins + 7 + t as usize] = 160;
        }
        let wf = Waterfall {
            mag: &mag,
            num_blocks: 79 + shift,
            num_bins,
            time_osr: 1,
            freq_osr: 1,
            protocol: Protocol::Ft8,
        };
        let mut heap = [Candidate::default(); 20];
        let n = find_candidates(&wf, &mut heap, 10);
        assert!(n > 0);
        assert_eq!(heap[0].time_offset as usize, shift);
        let (decoded, status) = decode_candidate(&wf, &heap[0], 30);
        let decoded = decoded.unwrap_or_else(|| panic!("decode failed: {status:?}"));
        let m = decoded.decode_std(&mut NoHash).unwrap();
        assert_eq!(&*m.extra, "-11");
    }

    /// Checked against the real thing across the range `normalize_logl` can
    /// produce, which is 24/variance for any variance a spectrogram yields.
    #[test]
    fn sqrtf_matches_std() {
        for x in [1e-6f32, 1e-3, 0.5, 1.0, 2.0, 24.0, 100.0, 1e4, 1e7] {
            let got = sqrtf(x);
            let want = std::primitive::f32::sqrt(x);
            assert!(
                (got - want).abs() / want < 1e-6,
                "sqrt({x}) = {got}, want {want}"
            );
        }
        assert_eq!(sqrtf(0.0), 0.0);
        assert_eq!(sqrtf(-1.0), 0.0);
    }
}
