//! The program's icon as a Windows resource file (`.res`), so that `rustxmalab.exe` carries
//! it: Explorer, the Start-menu entry the installer makes, and the taskbar show it.
//!
//! The Microsoft linker accepts a `.res` file as an input and links its resources into the
//! program, so no resource compiler is needed: this module writes the file from the `.ico`
//! with the standard library alone. A `.res` file is a sequence of entries, each a 32-byte
//! header (data size, header size, type, name, version fields) followed by the data padded
//! to four bytes, after an empty first entry. An icon becomes one `RT_ICON` entry per image
//! of the `.ico` and one `RT_GROUP_ICON` entry (the `.ico` directory, with each image's
//! offset replaced by its `RT_ICON` number). Group 1 is the lowest-numbered group, which is
//! the one Windows shows for the program.

/// `RT_ICON`.
const RT_ICON: u16 = 3;
/// `RT_GROUP_ICON`.
const RT_GROUP_ICON: u16 = 14;
/// `MOVEABLE | PURE | DISCARDABLE` for the images, as resource compilers write them.
const ICON_FLAGS: u16 = 0x1010;
/// `MOVEABLE | PURE | DISCARDABLE` for the group.
const GROUP_FLAGS: u16 = 0x1030;
/// US English, the language resource compilers give resources by default.
const LANGUAGE: u16 = 0x0409;

/// The `.res` file for the icon in `ico` (the bytes of an `.ico` file).
pub fn res_from_ico(ico: &[u8]) -> Result<Vec<u8>, String> {
    let u16_at = |at: usize| -> Result<u16, String> {
        ico.get(at..at + 2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .ok_or_else(|| "the .ico file is cut short".to_string())
    };
    let u32_at = |at: usize| -> Result<u32, String> {
        ico.get(at..at + 4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .ok_or_else(|| "the .ico file is cut short".to_string())
    };
    if u16_at(0)? != 0 || u16_at(2)? != 1 {
        return Err("not an .ico file".into());
    }
    let count = u16_at(4)?;
    if count == 0 {
        return Err("the .ico file holds no image".into());
    }

    let mut res = Vec::new();
    // The empty first entry that marks a 32-bit resource file.
    entry(&mut res, 0, 0, 0, 0, &[]);

    let mut group = Vec::with_capacity(6 + 14 * usize::from(count));
    group.extend_from_slice(&0u16.to_le_bytes());
    group.extend_from_slice(&1u16.to_le_bytes());
    group.extend_from_slice(&count.to_le_bytes());
    for i in 0..count {
        let dir = 6 + 16 * usize::from(i);
        let size = u32_at(dir + 8)?;
        let offset = u32_at(dir + 12)?;
        let start = usize::try_from(offset).map_err(|e| e.to_string())?;
        let len = usize::try_from(size).map_err(|e| e.to_string())?;
        let image = ico
            .get(start..start.saturating_add(len))
            .ok_or_else(|| format!("image {i} of the .ico file lies outside it"))?;
        let id = i + 1;
        entry(&mut res, RT_ICON, id, ICON_FLAGS, LANGUAGE, image);
        // Width, height, colour count and reserved are the .ico's own four bytes; planes
        // (written as 1 where the .ico says 0, as resource compilers do) and bit count
        // follow, then the image's size, and its resource number replaces the offset.
        group.extend_from_slice(&ico[dir..dir + 4]);
        group.extend_from_slice(&u16_at(dir + 4)?.max(1).to_le_bytes());
        group.extend_from_slice(&u16_at(dir + 6)?.to_le_bytes());
        group.extend_from_slice(&size.to_le_bytes());
        group.extend_from_slice(&id.to_le_bytes());
    }
    entry(&mut res, RT_GROUP_ICON, 1, GROUP_FLAGS, LANGUAGE, &group);
    Ok(res)
}

/// Appends one entry with a numbered type and name.
fn entry(res: &mut Vec<u8>, kind: u16, name: u16, flags: u16, language: u16, data: &[u8]) {
    let data_size = u32::try_from(data.len()).expect("a resource is far below 4 GiB");
    res.extend_from_slice(&data_size.to_le_bytes());
    res.extend_from_slice(&32u32.to_le_bytes()); // header size
    res.extend_from_slice(&0xFFFFu16.to_le_bytes()); // type is a number
    res.extend_from_slice(&kind.to_le_bytes());
    res.extend_from_slice(&0xFFFFu16.to_le_bytes()); // name is a number
    res.extend_from_slice(&name.to_le_bytes());
    res.extend_from_slice(&0u32.to_le_bytes()); // data version
    res.extend_from_slice(&flags.to_le_bytes());
    res.extend_from_slice(&language.to_le_bytes());
    res.extend_from_slice(&0u32.to_le_bytes()); // version
    res.extend_from_slice(&0u32.to_le_bytes()); // characteristics
    res.extend_from_slice(data);
    while !res.len().is_multiple_of(4) {
        res.push(0);
    }
}
