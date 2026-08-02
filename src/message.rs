//! The 77-bit FTx message payload: type dispatch, callsign and grid packing.
//! Ported from ft8_lib's `message.c`.
//!
//! Covered so far: type 1/2 (standard — the overwhelming majority of on-air
//! traffic: CQ, grid exchanges, reports, RRR/RR73/73), type 0.0 (free text) and
//! type 0.5 (telemetry). The contest and nonstandard-callsign types (0.1–0.4,
//! 0.6, 3, 4, 5) parse far enough to be *identified* but not yet rendered.

use crate::text::{charn, dd_to_int, int_to_dd, nchar, CharTable, Str};

/// Bytes holding the 77-bit payload.
pub const PAYLOAD_BYTES: usize = 10;

const MAX22: u32 = 4_194_304;
const NTOKENS: u32 = 2_063_592;
const MAXGRID4: u16 = 32_400;

/// A callsign as rendered: up to 11 characters, plus `/P` or `/R`, plus the
/// angle brackets a hash-recovered call is shown in.
pub type Callsign = Str<16>;
/// The third field: a grid, a report, `RRR`/`RR73`/`73`, or nothing.
pub type Extra = Str<8>;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MessageType {
    FreeText,
    DxPedition,
    EuVhf,
    ArrlFd,
    Telemetry,
    Contesting,
    Standard,
    ArrlRtty,
    NonstdCall,
    Wwrof,
    Unknown,
}

/// What a rendered field turned out to be — enough for a UI to colour a
/// callsign differently from a report, or to make a decode clickable.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Field {
    #[default]
    Unknown,
    None,
    /// `RRR`, `RR73`, `73`, `DE`, `QRZ`, `CQ`
    Token,
    /// `CQ nnn`, `CQ ABCD`
    TokenWithArg,
    Call,
    Grid,
    Rst,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Error {
    Callsign1,
    Callsign2,
    Suffix,
    Grid,
    /// The message is well-formed but of a type this crate does not render yet.
    UnsupportedType(MessageType),
}

/// A decoded standard message: `call_to`, `call_de` and the third field.
#[derive(Clone, Copy, Debug)]
pub struct StdMessage {
    pub call_to: Callsign,
    pub call_de: Callsign,
    pub extra: Extra,
    pub fields: [Field; 3],
}

/// Resolves the 22-bit hashes used for nonstandard callsigns.
///
/// FT8 sends some callsigns as a hash, on the assumption the receiver has heard
/// the full call earlier in the QSO. Without a table those decode to `<...>`,
/// which is exactly what WSJT-X shows in the same situation.
pub trait CallsignHash {
    fn lookup22(&self, _hash: u32) -> Option<Callsign> {
        None
    }
    fn save(&mut self, _callsign: &str) {}
}

/// The do-nothing implementation: hashed calls render as `<...>`.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoHash;
impl CallsignHash for NoHash {}

/// The 77-bit payload.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Message {
    pub payload: [u8; PAYLOAD_BYTES],
}

impl Message {
    pub fn from_payload(payload: [u8; PAYLOAD_BYTES]) -> Self {
        Self { payload }
    }

    /// Message type selector, bits 74..76.
    pub fn i3(&self) -> u8 {
        (self.payload[9] >> 3) & 0x07
    }

    /// Sub-type selector for `i3 == 0`, bits 71..73.
    pub fn n3(&self) -> u8 {
        ((self.payload[8] << 2) & 0x04) | ((self.payload[9] >> 6) & 0x03)
    }

    pub fn message_type(&self) -> MessageType {
        match self.i3() {
            0 => match self.n3() {
                0 => MessageType::FreeText,
                1 => MessageType::DxPedition,
                2 => MessageType::EuVhf,
                3 | 4 => MessageType::ArrlFd,
                5 => MessageType::Telemetry,
                6 => MessageType::Contesting,
                _ => MessageType::Unknown,
            },
            1 | 2 => MessageType::Standard,
            3 => MessageType::ArrlRtty,
            4 => MessageType::NonstdCall,
            5 => MessageType::Wwrof,
            _ => MessageType::Unknown,
        }
    }

    /// Decode a type 1/2 standard message.
    pub fn decode_std<H: CallsignHash>(&self, hash: &mut H) -> Result<StdMessage, Error> {
        let p = &self.payload;
        let n29a = ((p[0] as u32) << 21)
            | ((p[1] as u32) << 13)
            | ((p[2] as u32) << 5)
            | (p[3] as u32 >> 3);
        let n29b = ((p[3] as u32 & 0x07) << 26)
            | ((p[4] as u32) << 18)
            | ((p[5] as u32) << 10)
            | ((p[6] as u32) << 2)
            | (p[7] as u32 >> 6);
        let ir = (p[7] & 0x20) >> 5;
        let igrid4 = ((p[7] as u16 & 0x1F) << 10) | ((p[8] as u16) << 2) | (p[9] as u16 >> 6);
        let i3 = self.i3();

        let mut fields = [Field::Unknown; 3];
        let call_to = unpack28(n29a >> 1, (n29a & 1) as u8, i3, hash, &mut fields[0])
            .ok_or(Error::Callsign1)?;
        let call_de = unpack28(n29b >> 1, (n29b & 1) as u8, i3, hash, &mut fields[1])
            .ok_or(Error::Callsign2)?;
        let extra = unpack_grid(igrid4, ir, &mut fields[2]).ok_or(Error::Grid)?;

        Ok(StdMessage {
            call_to,
            call_de,
            extra,
            fields,
        })
    }

    /// Encode a type 1/2 standard message. Present mostly so the decoder can be
    /// tested without off-air recordings, but it is also what a transmitter
    /// would need.
    pub fn encode_std(call_to: &str, call_de: &str, extra: &str) -> Result<Self, Error> {
        let (n28a, ipa) = pack28(call_to).ok_or(Error::Callsign1)?;
        let (n28b, ipb) = pack28(call_de).ok_or(Error::Callsign2)?;

        let mut i3 = 1u8;
        if call_to.ends_with("/P") || call_de.ends_with("/P") {
            i3 = 2;
            if call_to.ends_with("/R") || call_de.ends_with("/R") {
                return Err(Error::Suffix);
            }
        }

        let igrid4 = pack_grid(extra);
        let mut n29a = (n28a << 1) | ipa as u32;
        let n29b = (n28b << 1) | ipb as u32;
        if call_to.ends_with("/R") {
            n29a |= 1;
        } else if call_to.ends_with("/P") {
            n29a |= 1;
            i3 = 2;
        }

        let mut payload = [0u8; PAYLOAD_BYTES];
        payload[0] = (n29a >> 21) as u8;
        payload[1] = (n29a >> 13) as u8;
        payload[2] = (n29a >> 5) as u8;
        payload[3] = ((n29a << 3) as u8) | ((n29b >> 26) as u8);
        payload[4] = (n29b >> 18) as u8;
        payload[5] = (n29b >> 10) as u8;
        payload[6] = (n29b >> 2) as u8;
        payload[7] = ((n29b << 6) as u8) | ((igrid4 >> 10) as u8);
        payload[8] = (igrid4 >> 2) as u8;
        payload[9] = ((igrid4 << 6) as u8) | (i3 << 3);
        Ok(Self { payload })
    }

    /// The 71 payload bits, right-aligned into 9 bytes. Shared by free text and
    /// telemetry, which are the same bits read two different ways.
    pub fn telemetry(&self) -> [u8; 9] {
        let mut out = [0u8; 9];
        let mut carry = 0u8;
        for (o, &p) in out.iter_mut().zip(self.payload.iter()) {
            *o = (carry << 7) | (p >> 1);
            carry = p & 0x01;
        }
        out
    }

    pub fn telemetry_hex(&self) -> Str<19> {
        const HEX: &[u8; 16] = b"0123456789ABCDEF";
        let b71 = self.telemetry();
        let mut out = Str::new();
        for b in b71 {
            out.push(HEX[(b >> 4) as usize]);
            out.push(HEX[(b & 0x0F) as usize]);
        }
        out
    }

    /// Decode type 0.0 free text: a 71-bit integer in base 42, most significant
    /// character first.
    pub fn decode_free(&self) -> Str<14> {
        let mut b71 = self.telemetry();
        let mut chars = [b' '; 13];
        for idx in (0..13).rev() {
            let mut rem = 0u16;
            for b in b71.iter_mut() {
                rem = (rem << 8) | *b as u16;
                *b = (rem / 42) as u8;
                rem %= 42;
            }
            chars[idx] = charn(rem as i32, CharTable::Full);
        }
        let mut out = Str::new();
        for c in chars {
            out.push(c);
        }
        out.trim_spaces();
        out
    }

    pub fn encode_free(text: &str) -> Self {
        // Base-42, most significant character first, then shifted left one bit
        // to sit in the 77-bit payload with i3 = n3 = 0.
        let mut b71 = [0u8; 9];
        for c in text
            .bytes()
            .take(13)
            .chain(core::iter::repeat(b' '))
            .take(13)
        {
            let v = nchar(c, CharTable::Full).unwrap_or(0) as u32;
            let mut carry = v;
            for b in b71.iter_mut().rev() {
                let acc = (*b as u32) * 42 + carry;
                *b = (acc & 0xFF) as u8;
                carry = acc >> 8;
            }
        }
        // Undo the right-shift `telemetry()` applies. Shifting a big-endian
        // number left by one bit means each byte takes the TOP bit of the byte
        // after it — not the bottom bit of the one before, which is the same
        // mistake in reverse and produces plausible-looking garbage.
        let mut payload = [0u8; PAYLOAD_BYTES];
        for i in 0..9 {
            let next = if i + 1 < 9 { b71[i + 1] >> 7 } else { 0 };
            payload[i] = (b71[i] << 1) | next;
        }
        // payload[8]'s LSB is n3's top bit and payload[9] carries the rest of
        // n3 plus i3; free text is n3 = i3 = 0, so both stay clear.
        payload[9] = 0;
        Self { payload }
    }
}

/// Pack a base callsign into its 28-bit numeric form, or `None` if it is not a
/// standard call. The two prefix work-arounds are not decoration: Swaziland and
/// Guinea calls do not fit the 1-2 char prefix / digit / 1-3 char suffix shape,
/// so the protocol rewrites them.
pub fn pack_basecall(callsign: &str, length: usize) -> Option<u32> {
    if length <= 2 {
        return None;
    }
    let cs = callsign.as_bytes();
    let mut c6 = [b' '; 6];

    if callsign.starts_with("3DA0") && length > 4 && length <= 7 {
        // 3DA0XYZ -> 3D0XYZ
        c6[..3].copy_from_slice(b"3D0");
        c6[3..3 + length - 4].copy_from_slice(&cs[4..length]);
    } else if callsign.starts_with("3X")
        && cs.get(2).is_some_and(|c| c.is_ascii_alphabetic())
        && length <= 7
    {
        // 3XA0XYZ -> QA0XYZ
        c6[0] = b'Q';
        c6[1..1 + length - 2].copy_from_slice(&cs[2..length]);
    } else if cs.get(2).is_some_and(|c| c.is_ascii_digit()) && length <= 6 {
        c6[..length].copy_from_slice(&cs[..length]);
    } else if cs.get(1).is_some_and(|c| c.is_ascii_digit()) && length <= 5 {
        c6[1..1 + length].copy_from_slice(&cs[..length]);
    }

    let i0 = nchar(c6[0], CharTable::AlphanumSpace)?;
    let i1 = nchar(c6[1], CharTable::Alphanum)?;
    let i2 = nchar(c6[2], CharTable::Numeric)?;
    let i3 = nchar(c6[3], CharTable::LettersSpace)?;
    let i4 = nchar(c6[4], CharTable::LettersSpace)?;
    let i5 = nchar(c6[5], CharTable::LettersSpace)?;

    let mut n = i0 as u32;
    n = n * 36 + i1 as u32;
    n = n * 10 + i2 as u32;
    n = n * 27 + i3 as u32;
    n = n * 27 + i4 as u32;
    n = n * 27 + i5 as u32;
    Some(n)
}

fn parse_cq_modifier(s: &str) -> Option<i32> {
    let b = s.as_bytes();
    let (mut nnum, mut nlet, mut m) = (0, 0, 0i32);
    for i in 3..8 {
        match b.get(i) {
            None | Some(b' ') => break,
            Some(c) if c.is_ascii_digit() => nnum += 1,
            Some(c) if c.is_ascii_uppercase() => {
                nlet += 1;
                m = 27 * m + (*c - b'A' + 1) as i32;
            }
            _ => return None, // '/' and friends are not allowed here
        }
    }
    if nnum == 3 && nlet == 0 {
        Some(dd_to_int(&b[3..], 3))
    } else if nnum == 0 && nlet <= 4 {
        Some(1000 + m)
    } else {
        None
    }
}

/// Returns `(n28, ip)` where `ip` flags a `/P` or `/R` suffix.
fn pack28(callsign: &str) -> Option<(u32, u8)> {
    match callsign {
        "DE" => return Some((0, 0)),
        "QRZ" => return Some((1, 0)),
        "CQ" => return Some((2, 0)),
        _ => {}
    }
    let length = callsign.len();
    if callsign.starts_with("CQ ") && length < 8 {
        return Some(((3 + parse_cq_modifier(callsign)?) as u32, 0));
    }

    let mut ip = 0u8;
    let mut length_base = length;
    if callsign.ends_with("/P") || callsign.ends_with("/R") {
        ip = 1;
        length_base = length - 2;
    }
    if let Some(n28) = pack_basecall(callsign, length_base) {
        return Some((NTOKENS + MAX22 + n28, ip));
    }
    // Nonstandard: would need a 22-bit hash, which requires a hash table and a
    // type 4 message. Not yet supported.
    None
}

fn unpack28<H: CallsignHash>(
    n28: u32,
    ip: u8,
    i3: u8,
    hash: &mut H,
    field: &mut Field,
) -> Option<Callsign> {
    let mut out = Callsign::new();

    if n28 < NTOKENS {
        if n28 <= 2 {
            out.push_str(match n28 {
                0 => "DE",
                1 => "QRZ",
                _ => "CQ",
            });
            *field = Field::Token;
            return Some(out);
        }
        if n28 <= 1002 {
            out.push_str("CQ ");
            int_to_dd(&mut out, (n28 - 3) as i32, 3, false);
            *field = Field::TokenWithArg;
            return Some(out);
        }
        if n28 <= 532_443 {
            let mut n = n28 - 1003;
            let mut aaaa = [b' '; 4];
            for i in (0..4).rev() {
                aaaa[i] = charn((n % 27) as i32, CharTable::LettersSpace);
                n /= 27;
            }
            out.push_str("CQ ");
            for &c in aaaa.iter().skip_while(|&&c| c == b' ') {
                out.push(c);
            }
            *field = Field::TokenWithArg;
            return Some(out);
        }
        return None;
    }

    let n28 = n28 - NTOKENS;
    if n28 < MAX22 {
        // A hashed nonstandard call. WSJT-X shows <...> when it has not heard
        // the full call yet, and so do we.
        *field = Field::Call;
        return Some(
            hash.lookup22(n28)
                .unwrap_or_else(|| Callsign::from_str_lossy("<...>")),
        );
    }

    let mut n = n28 - MAX22;
    let mut callsign = [0u8; 6];
    callsign[5] = charn((n % 27) as i32, CharTable::LettersSpace);
    n /= 27;
    callsign[4] = charn((n % 27) as i32, CharTable::LettersSpace);
    n /= 27;
    callsign[3] = charn((n % 27) as i32, CharTable::LettersSpace);
    n /= 27;
    callsign[2] = charn((n % 10) as i32, CharTable::Numeric);
    n /= 10;
    callsign[1] = charn((n % 36) as i32, CharTable::Alphanum);
    n /= 36;
    callsign[0] = charn((n % 37) as i32, CharTable::AlphanumSpace);

    // Undo the prefix work-arounds applied when packing.
    if callsign.starts_with(b"3D0") && callsign[3] != b' ' {
        out.push_str("3DA0");
        for &c in &callsign[3..] {
            out.push(c);
        }
    } else if callsign[0] == b'Q' && callsign[1].is_ascii_alphabetic() {
        out.push_str("3X");
        for &c in &callsign[1..] {
            out.push(c);
        }
    } else {
        for &c in &callsign {
            out.push(c);
        }
    }
    out.trim_spaces();

    if out.len() < 3 {
        return None;
    }
    if ip != 0 {
        match i3 {
            1 => out.push_str("/R"),
            2 => out.push_str("/P"),
            _ => return None,
        }
    }
    hash.save(out.as_str());
    *field = Field::Call;
    Some(out)
}

fn pack_grid(grid4: &str) -> u16 {
    if grid4.is_empty() {
        return MAXGRID4 + 1;
    }
    match grid4 {
        "RRR" => return MAXGRID4 + 2,
        "RR73" => return MAXGRID4 + 3,
        "73" => return MAXGRID4 + 4,
        _ => {}
    }
    let b = grid4.as_bytes();
    if b.len() >= 4
        && (b'A'..=b'R').contains(&b[0])
        && (b'A'..=b'R').contains(&b[1])
        && b[2].is_ascii_digit()
        && b[3].is_ascii_digit()
    {
        let mut igrid4 = (b[0] - b'A') as u16;
        igrid4 = igrid4 * 18 + (b[1] - b'A') as u16;
        igrid4 = igrid4 * 10 + (b[2] - b'0') as u16;
        igrid4 = igrid4 * 10 + (b[3] - b'0') as u16;
        return igrid4;
    }
    if b[0] == b'R' {
        let dd = dd_to_int(&b[1..], 3);
        ((MAXGRID4 as i32 + 35 + dd) as u16) | 0x8000 // ir = 1
    } else {
        let dd = dd_to_int(b, 3);
        (MAXGRID4 as i32 + 35 + dd) as u16
    }
}

fn unpack_grid(igrid4: u16, ir: u8, field: &mut Field) -> Option<Extra> {
    let mut out = Extra::new();
    if igrid4 <= MAXGRID4 {
        if ir > 0 {
            out.push_str("R ");
        }
        let n = igrid4;
        out.push(b'A' + (n / 10 / 10 / 18 % 18) as u8);
        out.push(b'A' + (n / 10 / 10 % 18) as u8);
        out.push(b'0' + (n / 10 % 10) as u8);
        out.push(b'0' + (n % 10) as u8);
        *field = Field::Grid;
    } else {
        let irpt = igrid4 - MAXGRID4;
        match irpt {
            1 => *field = Field::None,
            2 => {
                out.push_str("RRR");
                *field = Field::Token;
            }
            3 => {
                out.push_str("RR73");
                *field = Field::Token;
            }
            4 => {
                out.push_str("73");
                *field = Field::Token;
            }
            _ => {
                if ir > 0 {
                    out.push(b'R');
                }
                int_to_dd(&mut out, irpt as i32 - 35, 2, true);
                *field = Field::Rst;
            }
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(to: &str, de: &str, extra: &str) -> StdMessage {
        let msg = Message::encode_std(to, de, extra)
            .unwrap_or_else(|e| panic!("encode {to:?} {de:?} {extra:?} failed: {e:?}"));
        assert_eq!(msg.message_type(), MessageType::Standard);
        msg.decode_std(&mut NoHash)
            .unwrap_or_else(|e| panic!("decode {to:?} {de:?} {extra:?} failed: {e:?}"))
    }

    /// The traffic that actually fills a band: CQ, grid exchange, reports,
    /// acknowledgements, sign-off.
    #[test]
    fn standard_qso_round_trips() {
        for (to, de, extra) in [
            ("CQ", "K1ABC", "FN42"),
            ("CQ", "W9XYZ", "EN37"),
            ("K1ABC", "W9XYZ", "EN37"),
            ("K1ABC", "W9XYZ", "-11"),
            ("K1ABC", "W9XYZ", "R-09"),
            ("K1ABC", "W9XYZ", "RRR"),
            ("K1ABC", "W9XYZ", "RR73"),
            ("K1ABC", "W9XYZ", "73"),
            ("K1ABC", "W9XYZ", ""),
            ("DE", "K1ABC", "FN42"),
            ("QRZ", "K1ABC", "FN42"),
        ] {
            let d = round_trip(to, de, extra);
            assert_eq!(&*d.call_to, to, "call_to");
            assert_eq!(&*d.call_de, de, "call_de");
            assert_eq!(&*d.extra, extra, "extra for {to}/{de}");
        }
    }

    /// Grids span the whole A..R / 0..9 space; an off-by-one in the base-18/10
    /// unpacking would only show at the edges.
    #[test]
    fn every_grid_field_round_trips() {
        for &g in &["AA00", "RR99", "FN42", "JO22", "IO91", "AR09", "RA90"] {
            let d = round_trip("CQ", "K1ABC", g);
            assert_eq!(&*d.extra, g);
        }
    }

    /// Reports run -30..+49 with an optional R prefix.
    #[test]
    fn reports_round_trip_across_their_range() {
        for db in -30..=49 {
            let mut s = Str::<8>::new();
            int_to_dd(&mut s, db, 2, true);
            let d = round_trip("K1ABC", "W9XYZ", &s);
            assert_eq!(&*d.extra, &*s, "report {db}");

            let mut r = Str::<8>::from_str_lossy("R");
            int_to_dd(&mut r, db, 2, true);
            let d = round_trip("K1ABC", "W9XYZ", &r);
            assert_eq!(&*d.extra, &*r, "report R{db}");
        }
    }

    /// The prefix work-arounds are the fiddliest part of callsign packing and
    /// affect real countries (Swaziland, Guinea).
    #[test]
    fn prefix_workarounds_round_trip() {
        for call in ["3DA0XYZ", "3XA0ABC"] {
            let d = round_trip("CQ", call, "FN42");
            assert_eq!(&*d.call_de, call);
        }
    }

    #[test]
    fn portable_and_rover_suffixes_round_trip() {
        let d = round_trip("K1ABC", "W9XYZ/R", "FN42");
        assert_eq!(&*d.call_de, "W9XYZ/R");
        let d = round_trip("K1ABC", "W9XYZ/P", "FN42");
        assert_eq!(&*d.call_de, "W9XYZ/P");
    }

    #[test]
    fn cq_with_modifier_round_trips() {
        for cq in ["CQ 123", "CQ DX", "CQ POTA"] {
            let d = round_trip(cq, "K1ABC", "FN42");
            assert_eq!(&*d.call_to, cq);
        }
    }

    #[test]
    fn field_types_are_reported() {
        let d = round_trip("CQ", "K1ABC", "FN42");
        assert_eq!(d.fields, [Field::Token, Field::Call, Field::Grid]);
        let d = round_trip("K1ABC", "W9XYZ", "-11");
        assert_eq!(d.fields, [Field::Call, Field::Call, Field::Rst]);
        let d = round_trip("K1ABC", "W9XYZ", "RR73");
        assert_eq!(d.fields, [Field::Call, Field::Call, Field::Token]);
    }

    #[test]
    fn free_text_round_trips() {
        for t in ["HELLO WORLD", "TNX 73 GL", "A", "0123456789ABC"] {
            let msg = Message::encode_free(t);
            assert_eq!(msg.message_type(), MessageType::FreeText);
            assert_eq!(&*msg.decode_free(), t);
        }
    }

    #[test]
    fn telemetry_hex_is_18_digits() {
        let msg =
            Message::from_payload([0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x11, 0x2A]);
        assert_eq!(msg.telemetry_hex().len(), 18);
    }

    /// A hashed nonstandard call has no table behind it here, and must render
    /// the same placeholder WSJT-X uses rather than inventing a callsign.
    #[test]
    fn hashed_callsigns_render_as_placeholder() {
        let mut fields = [Field::Unknown; 3];
        let got = unpack28(NTOKENS + 12345, 0, 1, &mut NoHash, &mut fields[0]).unwrap();
        assert_eq!(&*got, "<...>");
        assert_eq!(fields[0], Field::Call);
    }
}
