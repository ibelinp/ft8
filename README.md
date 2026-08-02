# ft8

FT8 and FT4 decoding, in pure Rust. Give it audio, get messages.

```rust
use ft8::decode;

// 15 seconds of 12 kHz mono audio, aligned to the UTC slot boundary.
for d in decode(&audio, 12000.0) {
    println!("{:+3.0} dB  {:+4.1} s  {:5.0} Hz  {}",
             d.status.snr_db, d.time_sec, d.freq_hz, d.text());
}
```

```
 -3 dB  +0.0 s   1653 KM8C K1JLB EM12
 -9 dB  +0.1 s    916 KF8FWH K0MVB R+00
-15 dB  -0.1 s    813 CQ K6IL EM13
```

For a live receiver, feed it as the audio arrives instead — one symbol at a time
— so the FFT cost spreads across the slot rather than landing as one stall at
the end:

```rust
use ft8::{Monitor, MonitorConfig};

let mut mon = Monitor::new(MonitorConfig::default());
for chunk in audio.chunks(mon.block_size()) {
    mon.process(chunk);          // as each symbol arrives
}
let decodes = mon.decode_all(30, 10, 30);
```

One pass finds **every** station in the passband, not one at a time: the sync
search is a 2-D scan over time and frequency, so twenty simultaneous
transmissions come out of twenty different cells of the same spectrogram.

## Why this exists

Every FT8 decoder that runs in a browser today is either GPL (WSJT-X-derived, or
`ft8ts`) or carries no licence at all. That rules them out for anything you
cannot open-source — and "no licence" is the worse of the two, since it means
all rights reserved, not public domain.

The one permissively-licensed implementation is
[ft8_lib](https://github.com/kgoba/ft8_lib) (MIT) in C. This is a port of it to
Rust, so the same work is available to Rust projects — on the desktop and on
`wasm32` in a browser — with one dependency and no C toolchain.

## Status

A complete receive chain:

- **FFT and spectrogram** — Hann-windowed, 6.25 Hz bins, one block per 0.16 s
  symbol, with optional 2× oversampling on both axes
- **Sync search** — Costas correlation across time and frequency, candidate
  ranking
- **Soft-bit extraction**, **LDPC(174,91)**, **CRC-14**
- **Messages** — standard type 1/2 (CQ, grid exchanges, reports,
  `RRR`/`RR73`/`73`, `/P` and `/R`, `CQ nnn` / `CQ ABCD`, and the Swaziland and
  Guinea prefix work-arounds), free text, telemetry
- **SNR** in dB over a 2500 Hz reference bandwidth, the WSJT-X convention
- **Encoding**, enough to build a valid waveform — it exists mainly so the
  decoder can be tested without off-air recordings, but it is what a
  transmitter would need

Not yet:

- Contest and nonstandard-callsign message types (0.1–0.4, 0.6, 3, 4, 5) are
  *identified* by `Message::message_type` but render as `<contest>` rather than
  as text
- Hashed callsigns show as `<...>`, because no callsign hash table is kept —
  which is what WSJT-X shows too, before it has heard the full call. The
  `CallsignHash` trait is there for a caller that wants to supply one.
- FT4 shares the message and LDPC layers and has its sync patterns wired up, but
  is far less exercised than FT8

The SNR estimate has not been calibrated against a reference receiver. Its
spread and ordering are right; treat the absolute figure as good to a few dB.

## Testing

The tests synthesise real waveforms — phase-continuous 8-FSK at 6.25 Hz spacing
— and require them to come back as the original text. That covers the whole
chain, including the parts a bit-level test cannot see: a wrong window, a wrong
FFT size, an off-by-one in bin indexing. Two overlapping stations must both
decode; noise three times the signal amplitude must still decode; silence must
decode to nothing; SNR must fall monotonically as noise is added.

It has also been run against live 20 m off the air, decoding 14–17 stations per
slot at 3–8 ms per slot.

The LDPC tables are generated from ft8_lib's `constants.c` by
`tools/gen_ft8_constants.py` rather than transcribed — they are ~1300 numbers
and a single wrong digit surfaces much later as "some messages just don't
decode". The tests pin byte sums plus two structural invariants: `Nm`'s padding
must be zero past `Num_rows`, and `Mn` must be the exact transpose of `Nm`.

That transpose check matters more than it looks. A transposed or off-by-one
matrix still decodes clean signals perfectly; it only shows up as degraded
weak-signal performance. So the test that actually catches it is the one
requiring recovery from six flipped bits.

## Licence

MIT. A port of [ft8_lib](https://github.com/kgoba/ft8_lib) by Kārlis Goba, also
MIT — its licence is included as `LICENSE-ft8_lib` and must travel with any copy.

The LDPC matrices originate in WSJT-X (`ldpc_174_91_c_reordered_parity.f90`,
`bpdecode174.f90`). They are code *parameters* published in the FT8
specification rather than WSJT-X source, so no GPL obligation attaches.
