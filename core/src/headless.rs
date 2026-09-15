//! Synchronous Macintosh access for native debuggers and Rust fixtures.
//!
//! This owner runs no host event loop, bridge, audio device, or media writeback.
//! The existing CPU, bus, video and floppy implementations execute the guest.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Result, ensure};
use serde_json::Value;
use sha2::{Digest, Sha256};
use snow_floppy::loaders::{Autodetect, FloppyImageLoader};
use snow_floppy::noise::Noise;
mod controls;

use crate::cpu_m68k::regs::RegisterFile;
use crate::emulator::{EmulatorConfig, MouseMode, comm::EmulatorSpeed, construct_config};
use crate::keymap::KeyEvent;
use crate::mac::{ExtraROMs, MacModel, swim::Swim};
use crate::renderer::{DisplayBuffer, Renderer, channel::ChannelRenderer};

/// A completed native frame, observed at the end of a CPU instruction.
#[derive(Clone)]
pub struct Frame {
    pub width: u16,
    pub height: u16,
    pub rgba: Vec<u8>,
    pub sequence: u64,
    pub observed_at_cycle: u64,
}

/// A directly owned Macintosh. All methods act synchronously at instruction boundaries.
pub struct HeadlessMachine {
    config: EmulatorConfig,
    cycle_offset: u64,
    scsi_media: Vec<(usize, std::sync::Arc<std::sync::Mutex<Vec<u8>>>)>,
    frames: Arc<Mutex<Option<DisplayBuffer>>>,
    frame_sequence: Arc<AtomicU64>,
    last_frame: Option<Frame>,
}

impl HeadlessMachine {
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
        Self::new_model(HeadlessConfig {
            rom,
            extra_roms: &[],
            model: MacModel::Plus,
            ram_bytes: 4 * 1024 * 1024,
            pmmu: false,
            disk,
            seed,
            seconds,
            pram,
            mouse,
        })
    }

    /// Constructs any supported Snow model using the same factory as its GUI runner.
    pub fn new_model(options: HeadlessConfig<'_>) -> Result<Self> {
        let HeadlessConfig {
            rom,
            extra_roms,
            model,
            ram_bytes,
            pmmu,
            disk,
            seed,
            seconds,
            pram,
            mouse,
        } = options;
        ensure!(
            rom.len().is_power_of_two() && (64 * 1024..=4 * 1024 * 1024).contains(&rom.len()),
            "ROM must have a power-of-two size from 64 KB through 4 MB"
        );
        ensure!(
            model.ram_size_options().contains(&ram_bytes),
            "RAM size is not supported by {model}"
        );
        ensure!(
            model.cpu_type() != crate::cpu_m68k::M68000 || !pmmu,
            "68000 models cannot fit a PMMU"
        );
        ensure!(
            model.cpu_type() != crate::cpu_m68k::M68030 || pmmu,
            "68030 models have an integrated PMMU"
        );
        ensure!(
            model != MacModel::Portable15MB || rom.len() >= 256 * 1024,
            "Portable 15 MB ROM patch requires at least 256 KB"
        );
        for extra in extra_roms {
            let data = match extra {
                ExtraROMs::MDC12(data)
                | ExtraROMs::Toby(data)
                | ExtraROMs::SE30Video(data)
                | ExtraROMs::ExtensionROM(data) => data,
            };
            ensure!(!data.is_empty(), "Extra ROMs must not be empty");
        }
        let renderer = ChannelRenderer::new(512, 342)?;
        let frames = renderer.get_receiver();
        let frame_sequence = renderer.frame_sequence();
        let mut config = construct_config(
            rom,
            extra_roms,
            model,
            None,
            mouse,
            Some(ram_bytes),
            None,
            pmmu,
            renderer,
        )?;
        let frequency = match model {
            MacModel::Early128K
            | MacModel::Early512K
            | MacModel::Early512Ke
            | MacModel::Plus
            | MacModel::SE
            | MacModel::SeFdhd
            | MacModel::Classic => 8_000_000,
            _ => 16_000_000,
        };
        *config.swim_mut() =
            Swim::new_seeded(model.fdd_drives(), model.fdd_hd(), frequency, Some(seed));
        config.rtc_mut().initialize(seconds, pram)?;
        if let EmulatorConfig::Portable(cpu) = &mut config {
            cpu.bus.pmgr.initialize_clock(seconds, pram)?;
        }
        config.set_speed(EmulatorSpeed::Uncapped);

        if let Some(data) = disk {
            let mut image = Autodetect::load_with_noise(
                data,
                Some("boot"),
                Noise::seeded(seed ^ 0x4D45444941),
            )?;
            image.clear_dirty();
            config.swim_mut().disk_insert(0, image)?;
        }
        config.cpu_reset()?;
        config.cpu_sync_bus()?;
        Ok(Self {
            config,
            cycle_offset: 0,
            frames,
            frame_sequence,
            last_frame: None,
            scsi_media: Vec::new(),
        })
    }

    pub fn registers(&self) -> &RegisterFile {
        self.config.cpu_regs()
    }
    pub fn cycles(&self) -> u64 {
        self.cycle_offset + self.config.cpu_cycles()
    }
    pub fn ram(&self) -> &[u8] {
        self.config.ram()
    }
    pub fn peek(&mut self, address: u32) -> Option<u8> {
        self.config.bus_inspect_read(address)
    }
    /// Explain translation and compare cached mappings with current tables without guest effects.
    pub fn inspect_translation(
        &mut self,
        query: crate::cpu_m68k::pmmu::inspect::TranslationQuery,
    ) -> Result<crate::cpu_m68k::pmmu::inspect::TranslationInspection> {
        self.config.cpu_inspect_translation(query)
    }
    /// Inspect a bounded page of valid and flushed software ATC slots.
    pub fn inspect_atc(
        &self,
        start: usize,
        scan_limit: usize,
        entry_limit: usize,
    ) -> Result<crate::cpu_m68k::pmmu::inspect::AtcPage> {
        self.config.cpu_inspect_atc(start, scan_limit, entry_limit)
    }
    pub fn frame(&self) -> Option<&Frame> {
        self.last_frame.as_ref()
    }

    /// One CPU instruction, with device clocks synchronized before returning.
    pub fn step(&mut self) -> Result<()> {
        let result = self.config.cpu_step();
        self.config.cpu_sync_bus()?;
        result?;
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
        self.config.keyboard_event(event);
    }
    pub fn mouse_relative(&mut self, x: i16, y: i16, button: Option<bool>) {
        self.config.mouse_update_rel(x, y, button);
    }

    /// Returns false when the OS has not initialized its absolute mouse globals.
    pub fn mouse_absolute(&mut self, x: u16, y: u16) -> bool {
        self.config.try_mouse_update_abs(x, y)
    }

    /// Qualification counters: controller noise, drive noise, media-loading noise, dirty media.
    /// Reading these values performs no controller or device bus reads.
    pub fn media_activity(&self) -> (u64, [u64; 3], [u64; 3], [bool; 3]) {
        let controller = self.config.swim();
        (
            controller.noise_draws(),
            std::array::from_fn(|i| controller.drives[i].noise_draws()),
            std::array::from_fn(|i| controller.drives[i].floppy.noise_draws()),
            std::array::from_fn(|i| controller.drives[i].floppy.is_dirty()),
        )
    }

    /// Attach a private block image; no host paths or writeback are retained.
    pub fn attach_scsi(&mut self, id: usize, bytes: Vec<u8>) -> Result<()> {
        ensure!(
            self.config.model().has_scsi(),
            "This model has no SCSI controller"
        );
        ensure!(
            id < 7 && !self.scsi_media.iter().any(|(existing, _)| *existing == id),
            "SCSI ID must be unique and within 0..6"
        );
        ensure!(
            !bytes.is_empty() && bytes.len().is_multiple_of(512),
            "SCSI disk must contain complete 512-byte blocks"
        );
        let bytes = Arc::new(Mutex::new(bytes));
        self.config
            .scsi_mut()
            .attach_disk_image_at(Box::new(MemoryDisk(Arc::clone(&bytes))), id)?;
        self.scsi_media.push((id, bytes));
        Ok(())
    }

    /// Native breakpoint matching retains Snow's trigger semantics.
    pub fn set_breakpoint(&mut self, bp: crate::cpu_m68k::cpu::Breakpoint, enabled: bool) {
        self.config.cpu_clear_breakpoint(bp);
        if enabled {
            self.config.cpu_set_breakpoint(bp);
        }
    }
    pub fn take_breakpoint_hits(&mut self) -> (Vec<crate::cpu_m68k::cpu::BreakpointHit>, u64) {
        self.config.cpu_get_clr_breakpoint_hit();
        self.config.cpu_take_breakpoint_hits()
    }
    pub fn step_over_target(&self) -> Option<u32> {
        self.config.cpu_get_step_over()
    }
    pub fn configure_history(&mut self, instructions: bool, traps: bool) {
        self.config.cpu_enable_history(instructions);
        self.config.cpu_enable_systrap_history(traps);
    }
    pub fn history_status(&self) -> (bool, u64, bool, u64, usize) {
        self.config.cpu_history_status()
    }
    pub fn instruction_history(&mut self) -> Option<&[crate::cpu_m68k::cpu::HistoryEntry]> {
        self.config.cpu_read_history()
    }
    pub fn trap_history(&mut self) -> Option<&[crate::cpu_m68k::cpu::SystrapHistoryEntry]> {
        self.config.cpu_read_systrap_history()
    }
    pub fn peripherals(&self) -> crate::debuggable::DebuggableProperties {
        self.config.debug_properties()
    }

    /// Canonical guest-state components. JSON objects are sorted before hashing;
    /// paths, renderers, audio presentation and host RTC statistics are excluded.
    /// This is a comparison format, not a save-state API.
    pub fn digests(&self) -> Result<BTreeMap<String, String>> {
        let mut tagged = serde_json::to_value(&self.config)?;
        let mut cpu = std::mem::take(tagged.as_object_mut().unwrap().values_mut().next().unwrap());
        // Debugger policy has no guest meaning and must not affect evidence hashes.
        cpu.as_object_mut().unwrap().remove("breakpoints");
        cpu.as_object_mut().unwrap().remove("step_over_addr");
        let mut bus = cpu.as_object_mut().unwrap().remove("bus").unwrap();
        let mut parts = BTreeMap::new();
        for key in ["ram", "rom", "swim", "via", "scc", "scsi", "video"] {
            if let Some(value) = bus.as_object_mut().unwrap().remove(key) {
                parts.insert(key.to_string(), value);
            }
        }
        parts.insert("cpu".to_string(), cpu);
        if self.cycle_offset != 0 {
            parts.insert("cycle_offset".into(), Value::from(self.cycle_offset));
        }
        parts.insert("bus".to_string(), bus);
        let mut result: BTreeMap<String, String> = parts
            .into_iter()
            .map(|(name, mut value)| {
                canonicalize(&mut value);
                Ok((
                    name,
                    hex::encode(Sha256::digest(serde_json::to_vec(&value)?)),
                ))
            })
            .collect::<Result<_>>()?;
        for (id, media) in &self.scsi_media {
            result.insert(
                format!("scsi_disk_{id}"),
                hex::encode(Sha256::digest(&*media.lock().unwrap())),
            );
        }
        Ok(result)
    }
}

/// Backwards-compatible name for the initial Plus constructor.
pub type HeadlessPlus = HeadlessMachine;

/// Independent model configuration. The PMMU flag describes fitted hardware, not guest enable state.
#[derive(Clone, Copy)]
pub struct HeadlessConfig<'a> {
    pub rom: &'a [u8],
    pub extra_roms: &'a [ExtraROMs<'a>],
    pub model: MacModel,
    pub ram_bytes: usize,
    pub pmmu: bool,
    pub disk: Option<&'a [u8]>,
    pub seed: u64,
    pub seconds: u32,
    pub pram: &'a [u8],
    pub mouse: MouseMode,
}

struct MemoryDisk(Arc<Mutex<Vec<u8>>>);
impl crate::mac::scsi::disk_image::DiskImage for MemoryDisk {
    fn byte_len(&self) -> usize {
        self.0.lock().unwrap().len()
    }
    fn read_bytes(&self, offset: usize, length: usize) -> Vec<u8> {
        self.0.lock().unwrap()[offset..offset + length].to_vec()
    }
    fn write_bytes(&mut self, offset: usize, data: &[u8]) {
        self.0.lock().unwrap()[offset..offset + data.len()].copy_from_slice(data);
    }
    fn media_bytes(&self) -> Option<&[u8]> {
        None
    }
    fn image_path(&self) -> Option<&std::path::Path> {
        None
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

    #[test]
    fn debugger_edits_use_ram_mapping_and_refill_prefetch() -> Result<()> {
        use crate::cpu_m68k::regs::Register;
        let mut machine = machine(3)?;
        let before = machine.digests()?;
        for address in [0, 0x400008, 0x580000, 0xEFE1FF] {
            assert_eq!(machine.ram_address(address), None);
            assert_eq!(machine.write_memory_byte(address, 0xFF, false), None);
        }
        assert_eq!(before, machine.digests()?);
        // With four MiB, Snow CPU reads $600000 from backing offset $200000.
        // Resolve the inspection/read mapping rather than assuming an alias starts at zero.
        let code = [0x70, 0x2A, 0x52, 0x80, 0x60, 0xFE];
        for (offset, byte) in code.iter().enumerate() {
            assert_eq!(
                machine.write_memory_byte(0x200100 + offset as u32, *byte, true),
                Some(())
            );
            assert_eq!(machine.peek(0x600100 + offset as u32), Some(*byte));
        }
        let cycle = machine.cycles();
        machine.write_register(Register::PC, 0x600100)?;
        assert!(machine.cycles() > cycle);
        machine.step()?;
        assert_eq!(machine.registers().d[0], 42);
        machine.step()?;
        assert_eq!(machine.registers().d[0], 43);
        let cycle = machine.cycles();
        machine.reset()?;
        assert!(machine.cycles() > cycle);
        assert_eq!(machine.registers().pc, 0x400008);
        Ok(())
    }

    #[test]
    fn seeded_media_controls_are_private_and_repeatable() -> Result<()> {
        let mut a = machine(3)?;
        let mut b = machine(3)?;
        let source = vec![0; 400 * 1024];
        for machine in [&mut a, &mut b] {
            machine.insert_floppy(0, &source, 42, false)?;
            let exported = machine.export_floppy(0)?;
            machine.eject_floppy(0)?;
            machine.insert_floppy(0, &exported, 43, true)?;
            machine.attach_scsi(2, vec![0x5A; 512])?;
            assert_eq!(machine.export_scsi(2)?, vec![0x5A; 512]);
            machine.detach_scsi(2)?;
            machine.attach_cdrom(3, Some(vec![0; 2048]))?;
            machine.floppy_rpm(0, 2)?;
        }
        assert_eq!(a.digests()?, b.digests()?);
        assert_eq!(a.export_floppy(0)?, b.export_floppy(0)?);
        assert!(source.iter().all(|byte| *byte == 0));
        Ok(())
    }

    #[test]
    fn debugger_observations_preserve_guest_state_and_bound_history() -> Result<()> {
        use crate::cpu_m68k::cpu::{Breakpoint, BusBreakpoint};
        let mut observed = machine(3)?;
        let mut control = machine(3)?;
        observed.configure_history(true, true);
        observed.set_breakpoint(Breakpoint::Bus(BusBreakpoint::Read, 0x400008), true);
        for _ in 0..10_003 {
            observed.step()?;
            control.step()?;
        }
        assert_eq!(observed.digests()?, control.digests()?);
        let (enabled, total, traps, _, capacity) = observed.history_status();
        assert!(enabled && traps);
        assert_eq!(total, 10_003);
        assert_eq!(observed.instruction_history().unwrap().len(), capacity);
        let before = observed.digests()?;
        let (hits, dropped) = observed.take_breakpoint_hits();
        assert_eq!(hits.len(), 256);
        assert!(dropped > 0);
        assert_eq!(hits[0].value, Some(0x60));
        assert!(!observed.peripherals().is_empty());
        assert!(observed.trap_history().unwrap().is_empty());
        assert_eq!(before, observed.digests()?);
        observed.attach_scsi(0, vec![0; 512])?;
        assert!(observed.digests()?.contains_key("scsi_disk_0"));
        assert!(observed.attach_scsi(0, vec![0; 512]).is_err());
        Ok(())
    }

    #[test]
    fn all_models_share_the_native_factory() -> Result<()> {
        use strum::IntoEnumIterator;
        let mut rom = vec![0; 256 * 1024];
        rom[..4].copy_from_slice(&0x0001_FFFCu32.to_be_bytes());
        rom[4..8].copy_from_slice(&8u32.to_be_bytes());
        rom[8..10].copy_from_slice(&[0x60, 0xFE]);
        rom[0x2A..0x2C].copy_from_slice(&[0x60, 0xFE]);
        let video = vec![0; 64 * 1024];
        for model in MacModel::iter() {
            let extra = if model == MacModel::SE30 {
                ExtraROMs::SE30Video(&video)
            } else {
                ExtraROMs::MDC12(&video)
            };
            let mut machine = HeadlessMachine::new_model(HeadlessConfig {
                rom: &rom,
                extra_roms: &[extra],
                model,
                ram_bytes: model.ram_size_options()[0],
                pmmu: model.cpu_type() == crate::cpu_m68k::M68030,
                disk: None,
                seed: 1,
                seconds: 0,
                pram: &[0; 256],
                mouse: MouseMode::RelativeHw,
            })?;
            assert_eq!(machine.ram().len(), model.ram_size_options()[0]);
            assert_eq!(
                machine.registers().pc,
                if model == MacModel::Portable15MB {
                    0x00F3_FF70
                } else {
                    8
                }
            );
            for _ in 0..4 {
                machine.step()?;
            }
            assert_eq!(
                machine.registers().pc,
                if model == MacModel::Portable15MB {
                    0x00F0_002A
                } else {
                    8
                }
            );
            let before = machine.digests()?;
            assert!(!machine.mouse_absolute(0, 0));
            machine.peek(0x00F9_0000);
            machine.peek(0x5001_0000);
            assert_eq!(machine.digests()?, before, "{model}");
        }
        Ok(())
    }

    #[test]
    fn mmu_inspection_rejects_device_descriptors_without_guest_changes() -> Result<()> {
        use crate::cpu_m68k::pmmu::inspect::{TranslationMode, TranslationQuery};
        use crate::cpu_m68k::pmmu::regs::{RootPointerReg, TcReg};
        use crate::cpu_m68k::pmmu::walk::WalkFailureKind;

        let mut rom = vec![0; 256 * 1024];
        rom[..4].copy_from_slice(&0x0001_FFFCu32.to_be_bytes());
        rom[4..8].copy_from_slice(&8u32.to_be_bytes());
        rom[8..10].copy_from_slice(&[0x60, 0xFE]);
        let video = vec![0; 64 * 1024];

        for model in [MacModel::MacII, MacModel::MacIIx] {
            let mut machine = HeadlessMachine::new_model(HeadlessConfig {
                rom: &rom,
                extra_roms: &[ExtraROMs::MDC12(&video)],
                model,
                ram_bytes: model.ram_size_options()[0],
                pmmu: true,
                disk: None,
                seed: 1,
                seconds: 0,
                pram: &[0; 256],
                mouse: MouseMode::RelativeHw,
            })?;
            machine.configure_history(true, true);
            let registers = machine.config.cpu_regs_mut();
            registers.pmmu.tc = TcReg(0x80C8_6600);
            registers.pmmu.crp = RootPointerReg(0x7FFF_0002_5000_0000);
            let before = machine.digests()?;
            let cycles = machine.cycles();
            let history = machine.history_status();

            let result = machine.inspect_translation(TranslationQuery {
                address: 0x403006,
                function_code: 5,
                writing: false,
                mode: TranslationMode::CacheAware,
            })?;

            assert_eq!(
                result.failure.unwrap().kind,
                WalkFailureKind::UnreadableDescriptor
            );
            assert!(machine.inspect_atc(0, 32, 16)?.entries.is_empty());
            assert_eq!(machine.digests()?, before);
            assert_eq!(machine.cycles(), cycles);
            assert_eq!(machine.history_status(), history);
        }

        Ok(())
    }
}
