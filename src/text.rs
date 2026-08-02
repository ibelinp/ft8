//! Character tables and a fixed-capacity string, ported from ft8_lib's `text.c`.
//!
//! FT8 packs text against several different alphabets depending on the field,
//! and the alphabets are nested: each is the previous one minus a group. The
//! `CharTable` variants encode which groups are present rather than listing the
//! characters, exactly as the C does, so the moduli used by the message packer
//! (37, 36, 27, 10) line up with the table sizes by construction.

/// The alphabets FT8 packs against. Sizes: 42, 38, 37, 36, 27, 10.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CharTable {
    /// ` 0-9A-Z+-./?` — free text.
    Full,
    /// ` 0-9A-Z/`
    AlphanumSpaceSlash,
    /// ` 0-9A-Z`
    AlphanumSpace,
    /// `0-9A-Z`
    Alphanum,
    /// ` A-Z`
    LettersSpace,
    /// `0-9`
    Numeric,
}

impl CharTable {
    const fn has_space(self) -> bool {
        !matches!(self, CharTable::Alphanum | CharTable::Numeric)
    }
    const fn has_digits(self) -> bool {
        !matches!(self, CharTable::LettersSpace)
    }
    const fn has_letters(self) -> bool {
        !matches!(self, CharTable::Numeric)
    }
}

/// Index → character.
pub fn charn(mut c: i32, table: CharTable) -> u8 {
    if table.has_space() {
        if c == 0 {
            return b' ';
        }
        c -= 1;
    }
    if table.has_digits() {
        if c < 10 {
            return b'0' + c as u8;
        }
        c -= 10;
    }
    if table.has_letters() {
        if c < 26 {
            return b'A' + c as u8;
        }
        c -= 26;
    }
    match table {
        CharTable::Full if (0..5).contains(&c) => b"+-./?"[c as usize],
        CharTable::AlphanumSpaceSlash if c == 0 => b'/',
        _ => b'_', // unreachable for well-formed input
    }
}

/// Character → index, or `None` if the character is not in `table`.
pub fn nchar(c: u8, table: CharTable) -> Option<i32> {
    let mut n = 0;
    if table.has_space() {
        if c == b' ' {
            return Some(n);
        }
        n += 1;
    }
    if table.has_digits() {
        if c.is_ascii_digit() {
            return Some(n + (c - b'0') as i32);
        }
        n += 10;
    }
    if table.has_letters() {
        if c.is_ascii_uppercase() {
            return Some(n + (c - b'A') as i32);
        }
        n += 26;
    }
    match table {
        CharTable::Full => match c {
            b'+' => Some(n),
            b'-' => Some(n + 1),
            b'.' => Some(n + 2),
            b'/' => Some(n + 3),
            b'?' => Some(n + 4),
            _ => None,
        },
        CharTable::AlphanumSpaceSlash if c == b'/' => Some(n),
        _ => None,
    }
}

/// A fixed-capacity ASCII string.
///
/// The crate is `no_std` and allocation-free, which matters for the embedded
/// users ft8_lib was written for — so decoded callsigns and messages come back
/// in one of these rather than a `String`. It derefs to `&str`, so most callers
/// never have to think about it.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Str<const N: usize> {
    buf: [u8; N],
    len: usize,
}

impl<const N: usize> Default for Str<N> {
    fn default() -> Self {
        Self {
            buf: [0; N],
            len: 0,
        }
    }
}

impl<const N: usize> Str<N> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Silently ignores anything past capacity: every caller here sizes its
    /// buffer from the protocol's own field limits, so an overflow would mean a
    /// bug rather than untrusted input.
    pub fn push(&mut self, c: u8) {
        if self.len < N {
            self.buf[self.len] = c;
            self.len += 1;
        }
    }

    pub fn push_str(&mut self, s: &str) {
        for &b in s.as_bytes() {
            self.push(b);
        }
    }

    pub fn as_str(&self) -> &str {
        // Only ASCII is ever pushed, from the tables above.
        core::str::from_utf8(&self.buf[..self.len]).unwrap_or("")
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Drop leading and trailing spaces in place.
    pub fn trim_spaces(&mut self) {
        let mut start = 0;
        while start < self.len && self.buf[start] == b' ' {
            start += 1;
        }
        let mut end = self.len;
        while end > start && self.buf[end - 1] == b' ' {
            end -= 1;
        }
        self.buf.copy_within(start..end, 0);
        self.len = end - start;
    }

    pub fn from_str_lossy(s: &str) -> Self {
        let mut out = Self::new();
        out.push_str(s);
        out
    }
}

impl<const N: usize> core::ops::Deref for Str<N> {
    type Target = str;
    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl<const N: usize> core::fmt::Debug for Str<N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(self.as_str(), f)
    }
}

impl<const N: usize> core::fmt::Display for Str<N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Render `value` right-aligned in `width` digits, with a sign if negative or
/// if `full_sign`. Ported from `int_to_dd`.
pub fn int_to_dd<const N: usize>(out: &mut Str<N>, mut value: i32, width: u32, full_sign: bool) {
    if value < 0 {
        out.push(b'-');
        value = -value;
    } else if full_sign {
        out.push(b'+');
    }
    let mut divisor = 10i32.pow(width.saturating_sub(1));
    while divisor >= 1 {
        let digit = value / divisor;
        out.push(b'0' + digit as u8);
        value -= digit * divisor;
        divisor /= 10;
    }
}

/// Parse a signed integer of at most `len` digits. Ported from `dd_to_int`.
pub fn dd_to_int(s: &[u8], len: usize) -> i32 {
    let (negative, start) = match s.first() {
        Some(b'-') => (true, 1),
        Some(b'+') => (false, 1),
        _ => (false, 0),
    };
    let mut result = 0i32;
    for &c in s.iter().take(len).skip(start) {
        if !c.is_ascii_digit() {
            break;
        }
        result = result * 10 + (c - b'0') as i32;
    }
    if negative {
        -result
    } else {
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// charn and nchar must be exact inverses across every table, or callsigns
    /// round-trip to something subtly different.
    #[test]
    fn char_tables_round_trip() {
        for (table, size) in [
            (CharTable::Full, 42),
            (CharTable::AlphanumSpaceSlash, 38),
            (CharTable::AlphanumSpace, 37),
            (CharTable::Alphanum, 36),
            (CharTable::LettersSpace, 27),
            (CharTable::Numeric, 10),
        ] {
            for i in 0..size {
                let c = charn(i, table);
                assert_ne!(c, b'_', "{table:?} index {i} produced no character");
                assert_eq!(
                    nchar(c, table),
                    Some(i),
                    "{table:?} index {i} -> {}",
                    c as char
                );
            }
            // One past the end must not silently alias a valid character.
            assert_eq!(
                charn(size, table),
                b'_',
                "{table:?} accepted an out-of-range index"
            );
        }
    }

    #[test]
    fn int_to_dd_matches_ft8_conventions() {
        let mut s = Str::<8>::new();
        int_to_dd(&mut s, -7, 2, true);
        assert_eq!(&*s, "-07"); // reports are always two digits
        let mut s = Str::<8>::new();
        int_to_dd(&mut s, 12, 2, true);
        assert_eq!(&*s, "+12");
        let mut s = Str::<8>::new();
        int_to_dd(&mut s, 5, 3, false);
        assert_eq!(&*s, "005"); // CQ nnn
    }

    #[test]
    fn dd_to_int_handles_signs() {
        assert_eq!(dd_to_int(b"-07", 3), -7);
        assert_eq!(dd_to_int(b"+12", 3), 12);
        assert_eq!(dd_to_int(b"590", 3), 590);
        assert_eq!(dd_to_int(b"7", 3), 7);
    }

    #[test]
    fn str_trims_and_derefs() {
        let mut s = Str::<16>::from_str_lossy("  K1ABC  ");
        s.trim_spaces();
        assert_eq!(&*s, "K1ABC");
        assert_eq!(s.len(), 5);
    }
}
