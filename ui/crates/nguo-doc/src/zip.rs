// SPDX-License-Identifier: GPL-2.0-or-later
//! The minimum zip a `.nguo` needs: write a few members, read them back.
//!
//! # Why not the `zip` crate
//! Nothing in the tree can write a zip container, but `flate2` (and its
//! `crc32fast`) already ship inside the app — `png` -> `tiny-skia`/`vello` ->
//! `iced` -> `nguo-app` — so deflate and CRC-32 cost **zero new crates and zero
//! attribution churn**. Only the ~40 bytes of header per member are ours. The
//! reader accepts both `stored` and `deflate` precisely so a project a user
//! unzipped and re-zipped with `/usr/bin/zip` still opens.
//!
//! # Trust boundary
//! [`read_all`] parses a hostile byte slice: every field is a bounds-checked
//! `get`, every member is capped by its own declared size *and* by
//! [`MAX_MEMBER`]/[`MAX_TOTAL`], every CRC is verified, and every name is
//! rejected unless it is a plain relative path. It returns `Err`, never panics,
//! and never touches the filesystem — decompression is to memory only, so
//! nothing a `../` name could name can be written.

use std::collections::BTreeMap;
use std::io::{Read, Write};

const LOCAL_SIG: u32 = 0x0403_4b50;
const CENTRAL_SIG: u32 = 0x0201_4b50;
const EOCD_SIG: u32 = 0x0605_4b50;
const STORED: u16 = 0;
const DEFLATE: u16 = 8;
/// Set when sizes/CRC live in a trailing data descriptor rather than the local
/// header. We read sizes from the *central* directory, so it is harmless there,
/// but a set bit 0 (encryption) is not.
const FLAG_ENCRYPTED: u16 = 1 << 0;

/// ponytail: a project bundle is a manifest, a JSON document and a small
/// preview. These ceilings are ~100x the largest plausible one and exist only
/// to stop a zip bomb; raise them the day a `.nguo` legitimately embeds a mesh.
pub const MAX_MEMBER: u64 = 64 << 20;
pub const MAX_TOTAL: u64 = 256 << 20;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ZipError {
    #[error(
        "not a zip archive: no end-of-central-directory record (truncated or wrong file type?)"
    )]
    NotAZip,
    #[error("the archive is truncated or corrupt: {0}")]
    Corrupt(&'static str),
    #[error("member {name:?} uses unsupported zip feature: {what}")]
    Unsupported { name: String, what: &'static str },
    #[error("member {name:?} failed its CRC check — the archive is damaged")]
    BadCrc { name: String },
    #[error("refusing member {name:?}: a member path must be relative, with no `..` component")]
    UnsafeName { name: String },
    #[error("member {name:?} is larger than the {MAX_MEMBER} byte limit")]
    TooBig { name: String },
    #[error("the archive expands past the {MAX_TOTAL} byte limit")]
    TooBigTotal,
}

/// Reject anything that is not a plain relative path.
///
/// A `.nguo` is unzipped by third-party tools as well as by us; a member called
/// `../../.ssh/authorized_keys` must never reach a filesystem, ours or theirs.
pub fn check_name(name: &str) -> Result<(), ZipError> {
    let bad = name.is_empty()
        || name.len() > 1024
        || name.starts_with('/')
        || name.contains('\\')
        || name.contains('\0')
        || name.split('/').any(|c| c == ".." || c == ".")
        || name.chars().nth(1) == Some(':');
    if bad {
        return Err(ZipError::UnsafeName {
            name: name.to_owned(),
        });
    }
    Ok(())
}

/// A member's `(name, bytes)`, deflated on the way in.
pub struct Writer {
    out: Vec<u8>,
    dir: Vec<u8>,
    count: u16,
    dos: (u16, u16),
}

impl Writer {
    /// `dos` is `(time, date)` in MS-DOS form — see [`crate::bundle::dos_datetime`].
    /// It is what `unzip -l` prints, nothing else depends on it.
    pub fn new(dos: (u16, u16)) -> Writer {
        Writer {
            out: Vec::new(),
            dir: Vec::new(),
            count: 0,
            dos,
        }
    }

    pub fn add(&mut self, name: &str, data: &[u8]) -> Result<(), ZipError> {
        check_name(name)?;
        let mut crc = flate2::Crc::new();
        crc.update(data);
        let crc = crc.sum();

        // Measured on the 14-piece dress in `tests/bundle.rs` (release): level 6
        // gives 9 323 B in 1.2-3.9 ms, level 1 gives 19 950 B in 0.8-1.3 ms.
        // Half the file for a couple of ms on an explicit save is the right side
        // of that trade.
        let mut enc =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(data).expect("deflate into a Vec cannot fail");
        let packed = enc.finish().expect("deflate into a Vec cannot fail");

        // A tiny member can deflate larger than it started.
        let (method, body) = if packed.len() < data.len() {
            (DEFLATE, packed)
        } else {
            (STORED, data.to_vec())
        };
        let offset = self.out.len() as u32;

        self.out.extend_from_slice(&LOCAL_SIG.to_le_bytes());
        self.out.extend_from_slice(&20u16.to_le_bytes()); // version needed
        self.out.extend_from_slice(&0u16.to_le_bytes()); // flags
        self.out.extend_from_slice(&method.to_le_bytes());
        self.out.extend_from_slice(&self.dos.0.to_le_bytes());
        self.out.extend_from_slice(&self.dos.1.to_le_bytes());
        self.out.extend_from_slice(&crc.to_le_bytes());
        self.out
            .extend_from_slice(&(body.len() as u32).to_le_bytes());
        self.out
            .extend_from_slice(&(data.len() as u32).to_le_bytes());
        self.out
            .extend_from_slice(&(name.len() as u16).to_le_bytes());
        self.out.extend_from_slice(&0u16.to_le_bytes()); // extra len
        self.out.extend_from_slice(name.as_bytes());
        self.out.extend_from_slice(&body);

        self.dir.extend_from_slice(&CENTRAL_SIG.to_le_bytes());
        self.dir.extend_from_slice(&20u16.to_le_bytes()); // version made by
        self.dir.extend_from_slice(&20u16.to_le_bytes()); // version needed
        self.dir.extend_from_slice(&0u16.to_le_bytes()); // flags
        self.dir.extend_from_slice(&method.to_le_bytes());
        self.dir.extend_from_slice(&self.dos.0.to_le_bytes());
        self.dir.extend_from_slice(&self.dos.1.to_le_bytes());
        self.dir.extend_from_slice(&crc.to_le_bytes());
        self.dir
            .extend_from_slice(&(body.len() as u32).to_le_bytes());
        self.dir
            .extend_from_slice(&(data.len() as u32).to_le_bytes());
        self.dir
            .extend_from_slice(&(name.len() as u16).to_le_bytes());
        self.dir.extend_from_slice(&0u16.to_le_bytes()); // extra len
        self.dir.extend_from_slice(&0u16.to_le_bytes()); // comment len
        self.dir.extend_from_slice(&0u16.to_le_bytes()); // disk number
        self.dir.extend_from_slice(&0u16.to_le_bytes()); // internal attrs
        self.dir.extend_from_slice(&0u32.to_le_bytes()); // external attrs
        self.dir.extend_from_slice(&offset.to_le_bytes());
        self.dir.extend_from_slice(name.as_bytes());

        self.count += 1;
        Ok(())
    }

    pub fn finish(mut self) -> Vec<u8> {
        let dir_at = self.out.len() as u32;
        let dir_len = self.dir.len() as u32;
        self.out.append(&mut self.dir);
        self.out.extend_from_slice(&EOCD_SIG.to_le_bytes());
        self.out.extend_from_slice(&0u16.to_le_bytes()); // this disk
        self.out.extend_from_slice(&0u16.to_le_bytes()); // disk with the dir
        self.out.extend_from_slice(&self.count.to_le_bytes());
        self.out.extend_from_slice(&self.count.to_le_bytes());
        self.out.extend_from_slice(&dir_len.to_le_bytes());
        self.out.extend_from_slice(&dir_at.to_le_bytes());
        self.out.extend_from_slice(&0u16.to_le_bytes()); // comment len
        self.out
    }
}

fn le16(b: &[u8], at: usize) -> Result<u16, ZipError> {
    let s = b
        .get(at..at + 2)
        .ok_or(ZipError::Corrupt("a header runs past the end of the file"))?;
    Ok(u16::from_le_bytes([s[0], s[1]]))
}

fn le32(b: &[u8], at: usize) -> Result<u32, ZipError> {
    let s = b
        .get(at..at + 4)
        .ok_or(ZipError::Corrupt("a header runs past the end of the file"))?;
    Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

/// Every member of the archive, keyed by name, decompressed and CRC-checked.
///
/// Reads through the **central directory**, which is the authority: a file
/// re-zipped by another tool may use data descriptors, and only the directory
/// records the real sizes.
pub fn read_all(data: &[u8]) -> Result<BTreeMap<String, Vec<u8>>, ZipError> {
    // The EOCD is last, but a trailing comment can push it back up to 64 KiB.
    let scan_from = data.len().saturating_sub(22 + 0xffff);
    let eocd = (scan_from..=data.len().saturating_sub(22))
        .rev()
        .find(|&at| le32(data, at) == Ok(EOCD_SIG))
        .ok_or(ZipError::NotAZip)?;

    let count = le16(data, eocd + 10)?;
    let mut at = le32(data, eocd + 16)? as usize;
    if at == 0xffff_ffff || le16(data, eocd + 8)? == 0xffff {
        return Err(ZipError::Unsupported {
            name: String::new(),
            what: "ZIP64 (a .nguo is never 4 GB)",
        });
    }

    let mut out = BTreeMap::new();
    let mut total = 0u64;
    for _ in 0..count {
        if le32(data, at)? != CENTRAL_SIG {
            return Err(ZipError::Corrupt("the central directory is malformed"));
        }
        let flags = le16(data, at + 8)?;
        let method = le16(data, at + 10)?;
        let crc = le32(data, at + 16)?;
        let csize = le32(data, at + 20)? as u64;
        let usize_ = le32(data, at + 24)? as u64;
        let name_len = le16(data, at + 28)? as usize;
        let extra_len = le16(data, at + 30)? as usize;
        let comment_len = le16(data, at + 32)? as usize;
        let local = le32(data, at + 42)? as usize;
        let name = data
            .get(at + 46..at + 46 + name_len)
            .ok_or(ZipError::Corrupt("a member name runs past the end"))?;
        let name = String::from_utf8_lossy(name).into_owned();
        at += 46 + name_len + extra_len + comment_len;

        check_name(&name)?;
        if flags & FLAG_ENCRYPTED != 0 {
            return Err(ZipError::Unsupported {
                name,
                what: "encryption",
            });
        }
        if method != STORED && method != DEFLATE {
            return Err(ZipError::Unsupported {
                name,
                what: "a compression method other than stored/deflate",
            });
        }
        if usize_ > MAX_MEMBER || csize > MAX_MEMBER {
            return Err(ZipError::TooBig { name });
        }
        total += usize_;
        if total > MAX_TOTAL {
            return Err(ZipError::TooBigTotal);
        }

        // Skip the local header (its name/extra lengths may differ from the
        // directory's) and take exactly `csize` bytes.
        if le32(data, local)? != LOCAL_SIG {
            return Err(ZipError::Corrupt("a member's local header is missing"));
        }
        let body_at =
            local + 30 + le16(data, local + 26)? as usize + le16(data, local + 28)? as usize;
        let body = data
            .get(body_at..body_at + csize as usize)
            .ok_or(ZipError::Corrupt("a member's data is truncated"))?;

        let plain = if method == STORED {
            body.to_vec()
        } else {
            // Bounded by the declared size, so a bomb inflates to nothing worse.
            let mut plain = Vec::with_capacity(usize_ as usize);
            flate2::read::DeflateDecoder::new(body)
                .take(usize_ + 1)
                .read_to_end(&mut plain)
                .map_err(|_| ZipError::Corrupt("a member failed to decompress"))?;
            plain
        };
        if plain.len() as u64 != usize_ {
            return Err(ZipError::Corrupt(
                "a member's size does not match its header",
            ));
        }
        let mut check = flate2::Crc::new();
        check.update(&plain);
        if check.sum() != crc {
            return Err(ZipError::BadCrc { name });
        }
        out.insert(name, plain);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn archive() -> Vec<u8> {
        let mut w = Writer::new((0, 0x21));
        // Long and repetitive so it deflates; short so it does not.
        w.add("manifest.json", &b"{\"a\":1}".repeat(200)).unwrap();
        w.add("tiny", b"x").unwrap();
        w.finish()
    }

    #[test]
    fn members_round_trip_through_both_methods() {
        let out = read_all(&archive()).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out["manifest.json"], b"{\"a\":1}".repeat(200));
        assert_eq!(out["tiny"], b"x");
    }

    /// The central directory is data, not truth: a header that claims a member
    /// bigger than the file, or one living past its end, must be an error and
    /// not a slice out of bounds.
    #[test]
    fn a_lying_directory_cannot_read_past_the_end() {
        let full = archive();
        let central = full
            .windows(4)
            .position(|w| w == CENTRAL_SIG.to_le_bytes())
            .expect("a central directory");
        for (field, value) in [
            (20usize, 0x00ff_ffffu32),
            (24, 0x00ff_ffff),
            (42, 0x00ff_ffff),
        ] {
            let mut a = full.clone();
            a[central + field..central + field + 4].copy_from_slice(&value.to_le_bytes());
            assert!(
                matches!(read_all(&a), Err(ZipError::Corrupt(_))),
                "field at +{field} must be range-checked"
            );
        }
    }

    #[test]
    fn a_truncated_archive_is_an_error_not_a_panic() {
        let full = archive();
        for cut in [
            0,
            1,
            30,
            60,
            full.len() / 2,
            full.len() - 25,
            full.len() - 1,
        ] {
            assert!(read_all(&full[..cut]).is_err(), "cut at {cut} must fail");
        }
        // Losing bytes from the *middle* keeps the EOCD but breaks the data.
        let mut holed = full.clone();
        holed.drain(40..60);
        assert!(read_all(&holed).is_err());
    }

    #[test]
    fn a_flipped_byte_is_caught_by_the_crc() {
        let mut a = archive();
        let last = a.len();
        // The stored member's single byte, right before the central directory.
        let at = a[..last]
            .windows(4)
            .position(|w| w == CENTRAL_SIG.to_le_bytes())
            .unwrap()
            - 1;
        a[at] ^= 0xff;
        assert!(matches!(read_all(&a), Err(ZipError::BadCrc { .. })));
    }

    #[test]
    fn a_member_path_cannot_escape_the_archive() {
        for name in [
            "../evil",
            "/etc/passwd",
            "a/../../b",
            "C:/x",
            "back\\slash",
            "",
            "./x",
        ] {
            assert!(check_name(name).is_err(), "{name:?} must be rejected");
        }
        for name in ["manifest.json", "assets/mesh.bin", "a.b.c"] {
            check_name(name).unwrap();
        }
    }

    #[test]
    fn a_hostile_name_in_the_directory_is_refused_on_read() {
        // Hand-patch a written archive's name to `../x/` (same length as
        // `tiny/`), in the local header and the central directory both.
        let mut w = Writer::new((0, 0x21));
        w.add("tiny/", b"x").unwrap();
        let mut a = w.finish();
        let mut patched = 0;
        for at in 0..a.len() - 5 {
            if &a[at..at + 5] == b"tiny/" {
                a[at..at + 5].copy_from_slice(b"../x/");
                patched += 1;
            }
        }
        assert_eq!(patched, 2, "one name per header");
        assert!(matches!(read_all(&a), Err(ZipError::UnsafeName { .. })));
    }

    #[test]
    fn random_bytes_are_not_a_zip() {
        assert_eq!(read_all(b"not a zip at all"), Err(ZipError::NotAZip));
        assert_eq!(read_all(&[]), Err(ZipError::NotAZip));
    }
}
