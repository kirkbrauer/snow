//! Synchronous compact-Mac access for native debuggers and Rust fixtures.
//!
//! This owner runs no host event loop, bridge, audio device, or media writeback.
//! The existing CPU, bus, video and floppy implementations execute the guest.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Result, ensure};
use serde_json::Value;
use sha2::{Digest, Sha256};
use snow_floppy::{FloppyType, macformat::MacFormatEncoder, noise::Noise};

use crate::bus::InspectableBus;
use crate::cpu_m68k::{CpuM68000, regs::RegisterFile};
use crate::emulator::{MouseMode, comm::EmulatorSpeed};
use crate::keymap::KeyEvent;
use crate::mac::{MacModel, compact::bus::CompactMacBus, swim::Swim};
use crate::renderer::{DisplayBuffer, Renderer, channel::ChannelRenderer};

struct CaptureRenderer {
    inner: ChannelRenderer,
    sequence: Arc<AtomicU64>,
}

impl Renderer for CaptureRenderer {
    fn new(width: u16, height: u16) -> Result<Self> {
        Ok(Self {
            inner: ChannelRenderer::new(width, height)?,
            sequence: Arc::new(AtomicU64::new(0)),
        })
    }

    fn buffer_mut(&mut self) -> &mut DisplayBuffer {
        self.inner.buffer_mut()
    }

    fn update(&mut self) -> Result<()> {
        self.inner.update()?;
        self.sequence.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

/// A completed native frame, observed at the end of a CPU instruction.
#[derive(Clone)]
pub struct Frame {
    pub width: u16,
    pub height: u16,
    pub rgba: Vec<u8>,
    pub sequence: u64,
    pub observed_at_cycle: u64,
}

/// A directly owned Plus. All methods act synchronously at instruction boundaries.
pub struct HeadlessPlus {
    cpu: Box<CpuM68000<CompactMacBus<CaptureRenderer>>>,
    frames: Arc<Mutex<Option<DisplayBuffer>>>,
    frame_sequence: Arc<AtomicU64>,
    last_frame: Option<Frame>,
}

impl HeadlessPlus {
    /// Creates an isolated 4 MB Plus. Raw 400/800 KB sector media is supported.
    /// `seconds` is a Macintosh RTC value (seconds since 1904), independent of timezone.
    pub fn new(
        rom: &[u8],
        disk: Option<&[u8]>,
        seed: u64,
        seconds: u32,
        pram: &[u8],
        mouse: MouseMode,
    ) -> Result<Self> {
        ensure!(rom.len() == 128 * 1024, "Plus ROM must be 128 KB");
        let renderer = CaptureRenderer::new(512, 342)?;
        let frames = renderer.inner.get_receiver();
        let frame_sequence = renderer.sequence.clone();
        let mut bus = CompactMacBus::new(
            MacModel::Plus,
            rom,
            None,
            renderer,
            mouse,
            Some(4 * 1024 * 1024),
            None,
        );
        bus.swim = Swim::new_seeded(MacModel::Plus.fdd_drives(), false, 8_000_000, Some(seed));
        bus.rtc_mut().initialize(seconds, pram)?;
        bus.set_speed(EmulatorSpeed::Uncapped);

        if let Some(data) = disk {
            let format = match data.len() {
                409600 => FloppyType::Mac400K,
                819200 => FloppyType::Mac800K,
                _ => anyhow::bail!("Headless Plus requires raw 400/800 KB sector media"),
            };
            let mut image = MacFormatEncoder::encode_with_noise(
                format,
                data,
                None,
                "boot",
                Noise::seeded(seed ^ 0x4D45444941),
            )?;
            image.clear_dirty();
            bus.swim.disk_insert(0, image)?;
        }

        let mut cpu = Box::new(CpuM68000::new(bus));
        cpu.reset()?;
        cpu.sync_bus()?;
        Ok(Self {
            cpu,
            frames,
            frame_sequence,
            last_frame: None,
        })
    }

    pub fn registers(&self) -> &RegisterFile {
        &self.cpu.regs
    }
    pub fn cycles(&self) -> u64 {
        self.cpu.cycles
    }
    pub fn ram(&self) -> &[u8] {
        &self.cpu.bus.ram
    }
    pub fn peek(&mut self, address: u32) -> Option<u8> {
        self.cpu.bus.inspect_read(address)
    }
    pub fn frame(&self) -> Option<&Frame> {
        self.last_frame.as_ref()
    }

    /// One CPU instruction, with device clocks synchronized before returning.
    pub fn step(&mut self) -> Result<()> {
        self.cpu.step()?;
        self.cpu.sync_bus()?;
        let sequence = self.frame_sequence.load(Ordering::Relaxed);
        if sequence > self.last_frame.as_ref().map_or(0, |f| f.sequence) {
            let buffer = self
                .frames
                .lock()
                .map_err(|_| anyhow::anyhow!("Frame lock poisoned"))?
                .take();
            if let Some(buffer) = buffer {
                self.last_frame = Some(Frame {
                    width: buffer.width(),
                    height: buffer.height(),
                    rgba: buffer.into_inner(),
                    sequence,
                    observed_at_cycle: self.cycles(),
                });
            }
        }
        Ok(())
    }

    pub fn key(&mut self, event: KeyEvent) {
        self.cpu.bus.keyboard_event(event);
    }
    pub fn mouse_relative(&mut self, x: i16, y: i16, button: Option<bool>) {
        self.cpu.bus.mouse_update_rel(x, y, button);
    }

    /// Returns false when the OS has not initialized its absolute mouse globals.
    pub fn mouse_absolute(&mut self, x: u16, y: u16) -> bool {
        self.cpu.bus.try_mouse_update_abs(x, y)
    }

    /// Qualification counters: controller noise, drive noise, media-loading noise, dirty media.
    /// Reading these values performs no controller or device bus reads.
    pub fn media_activity(&self) -> (u64, [u64; 3], [u64; 3], [bool; 3]) {
        let controller = &self.cpu.bus.swim;
        (
            controller.noise_draws(),
            std::array::from_fn(|i| controller.drives[i].noise_draws()),
            std::array::from_fn(|i| controller.drives[i].floppy.noise_draws()),
            std::array::from_fn(|i| controller.drives[i].floppy.is_dirty()),
        )
    }

    /// Canonical guest-state components. JSON objects are sorted before hashing;
    /// paths, renderers, audio presentation and host RTC statistics are excluded.
    /// This is a comparison format, not a save-state API.
    pub fn digests(&self) -> Result<BTreeMap<String, String>> {
        let mut cpu = serde_json::to_value(&self.cpu)?;
        let mut bus = cpu.as_object_mut().unwrap().remove("bus").unwrap();
        let mut parts = BTreeMap::new();
        for key in ["ram", "rom", "swim", "via", "scc", "scsi", "video"] {
            if let Some(value) = bus.as_object_mut().unwrap().remove(key) {
                parts.insert(key.to_string(), value);
            }
        }
        parts.insert("cpu".to_string(), cpu);
        parts.insert("bus".to_string(), bus);
        parts
            .into_iter()
            .map(|(name, mut value)| {
                canonicalize(&mut value);
                Ok((
                    name,
                    hex::encode(Sha256::digest(serde_json::to_vec(&value)?)),
                ))
            })
            .collect()
    }
}

fn canonicalize(value: &mut Value) {
    match value {
        Value::Object(map) => {
            map.sort_keys();
            for value in map.values_mut() {
                canonicalize(value);
            }
        }
        Value::Array(values) => {
            for value in values {
                canonicalize(value);
            }
        }
        _ => (),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn machine(seed: u64) -> Result<HeadlessPlus> {
        let mut rom = vec![0; 128 * 1024];
        rom[..4].copy_from_slice(&0x003F_FFFCu32.to_be_bytes());
        rom[4..8].copy_from_slice(&0x0040_0008u32.to_be_bytes());
        rom[8..10].copy_from_slice(&[0x60, 0xFE]); // BRA.S to itself
        HeadlessPlus::new(&rom, None, seed, 0, &[0; 256], MouseMode::RelativeHw)
    }

    #[test]
    fn repeatable_step_and_safe_inspection() -> Result<()> {
        let mut a = machine(3)?;
        let mut b = machine(3)?;
        for _ in 0..100 {
            a.step()?;
            b.step()?;
        }
        assert_eq!(a.registers().pc, 0x400008);
        assert_eq!(a.digests()?, b.digests()?);
        let before = a.digests()?;
        assert_eq!(a.peek(0x580000), None);
        assert_eq!(a.peek(0xEFE1FF), None);
        assert_eq!(a.peek(0x400008), Some(0x60));
        assert_eq!(a.digests()?, before);
        assert_ne!(a.digests()?, machine(4)?.digests()?);
        Ok(())
    }
}
