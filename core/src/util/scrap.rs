//! Read-only Classic Mac OS resident Scrap Manager inspection, shared with clients.

/// A resident TEXT payload, without encoding or newline normalization.
#[derive(Debug, PartialEq, Eq)]
pub struct ScrapText<'a> {
    pub address: u32,
    pub bytes: &'a [u8],
}

/// Why resident TEXT could not be returned. No guest routine is called.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScrapError {
    Empty,
    Nonresident,
    NoText,
    LimitExceeded,
    Unreadable,
    Malformed,
}

/// Inspect backing RAM using an explicit 24- or 32-bit pointer mask.
///
/// The bound covers the entire encoded scrap, including unknown types and padding.
/// All entries are validated, even those after TEXT. The first TEXT entry wins.
pub fn inspect_scrap(mem: &[u8], limit: usize, mask: u32) -> Result<ScrapText<'_>, ScrapError> {
    let word = |address: usize| -> Result<u32, ScrapError> {
        let bytes = mem
            .get(address..address + 4)
            .ok_or(ScrapError::Unreadable)?;
        Ok(u32::from_be_bytes(
            bytes.try_into().map_err(|_| ScrapError::Unreadable)?,
        ))
    };
    let state = mem.get(0x96A..0x96C).ok_or(ScrapError::Unreadable)?;
    if i16::from_be_bytes([state[0], state[1]]) < 0 {
        return Err(ScrapError::Nonresident);
    }
    let size = word(0x960)? as usize;
    if size == 0 {
        return Err(ScrapError::Empty);
    }
    if size > limit {
        return Err(ScrapError::LimitExceeded);
    }
    let handle = word(0x964)? & mask;
    if handle == 0 {
        return Err(ScrapError::Malformed);
    }
    let base = word(handle as usize)? & mask;
    if base == 0 {
        return Err(ScrapError::Malformed);
    }
    let end = (base as usize)
        .checked_add(size)
        .ok_or(ScrapError::Malformed)?;
    let scrap = mem.get(base as usize..end).ok_or(ScrapError::Unreadable)?;
    let mut offset = 0;
    let mut text = None;
    while offset < size {
        let header = scrap.get(offset..offset + 8).ok_or(ScrapError::Malformed)?;
        let length = u32::from_be_bytes(header[4..8].try_into().map_err(|_| ScrapError::Malformed)?)
            as usize;
        let start = offset + 8;
        let end = start.checked_add(length).ok_or(ScrapError::Malformed)?;
        let payload = scrap.get(start..end).ok_or(ScrapError::Malformed)?;
        offset = end.checked_add(length & 1).ok_or(ScrapError::Malformed)?;
        if offset > size {
            return Err(ScrapError::Malformed);
        }
        if &header[..4] == b"TEXT" && text.is_none() {
            text = Some(ScrapText {
                address: base
                    .checked_add(u32::try_from(start).map_err(|_| ScrapError::Malformed)?)
                    .ok_or(ScrapError::Malformed)?,
                bytes: payload,
            });
        }
    }
    text.ok_or(ScrapError::NoText)
}

/// GUI convenience: normalize classic carriage returns when copying to the host.
pub fn read_scrap_text(mem: &[u8]) -> Option<String> {
    inspect_scrap(mem, 1024 * 1024, u32::MAX)
        .ok()
        .map(|text| super::mac::macroman_to_utf8(text.bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resident_scrap_validates_unknown_entries_padding_bounds_and_pointer_tags() {
        let mut ram = vec![0; 0x2000];
        assert_eq!(inspect_scrap(&ram, 100, u32::MAX), Err(ScrapError::Empty));
        ram[0x964..0x968].copy_from_slice(&0xAB001000u32.to_be_bytes());
        ram[0x1000..0x1004].copy_from_slice(&0xCD001100u32.to_be_bytes());
        let data = b"PICT\0\0\0\x01x\0TEXT\0\0\0\x03a\rb\0";
        ram[0x1100..0x1100 + data.len()].copy_from_slice(data);
        ram[0x960..0x964].copy_from_slice(&(data.len() as u32).to_be_bytes());
        let text = inspect_scrap(&ram, 100, 0xFFFFFF).unwrap();
        assert_eq!(text.address, 0x1112);
        assert_eq!(text.bytes, b"a\rb");
        assert_eq!(
            inspect_scrap(&ram, 100, u32::MAX),
            Err(ScrapError::Unreadable)
        );
        assert_eq!(
            inspect_scrap(&ram, 8, 0xFFFFFF),
            Err(ScrapError::LimitExceeded)
        );
        ram[0x960..0x964].copy_from_slice(&10u32.to_be_bytes());
        assert_eq!(inspect_scrap(&ram, 100, 0xFFFFFF), Err(ScrapError::NoText));
        ram[0x960..0x964].copy_from_slice(&21u32.to_be_bytes());
        assert_eq!(
            inspect_scrap(&ram, 100, 0xFFFFFF),
            Err(ScrapError::Malformed)
        );
        ram[0x96A..0x96C].copy_from_slice(&(-1i16).to_be_bytes());
        assert_eq!(
            inspect_scrap(&ram, 100, 0xFFFFFF),
            Err(ScrapError::Nonresident)
        );
        assert_eq!(
            inspect_scrap(&ram[..8], 100, 0xFFFFFF),
            Err(ScrapError::Unreadable)
        );
    }
}
