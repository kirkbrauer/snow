use super::*;
use crate::cpu_m68k::regs::Register;
use crate::mac::scc::SccCh;
use snow_floppy::loaders::{FloppyImageSaver, Moof};

impl HeadlessMachine {
    /// The same register edits as Snow's GUI. Setting PC refills prefetch and consumes cycles.
    pub fn write_register(&mut self, register: Register, value: u32) -> Result<()> {
        if register == Register::PC {
            ensure!(value & 1 == 0, "PC must be aligned");
            self.config.cpu_set_pc(value)?;
            self.config.cpu_prefetch_refill()?;
            self.config.cpu_sync_bus()?;
        } else {
            self.config.cpu_regs_mut().write(register, value);
        }
        Ok(())
    }

    /// Resolve a physical CPU-bus address to a backing-RAM offset without side effects.
    pub fn ram_address(&mut self, address: u32) -> Option<u32> {
        self.config.bus_debugger_ram_address(address)
    }

    /// Writes one inspected byte; RAM mode avoids overlays and updates dirty/video bookkeeping.
    pub fn write_memory_byte(&mut self, address: u32, value: u8, backing_ram: bool) -> Option<()> {
        if backing_ram {
            self.config.bus_debugger_write_ram(address, value)
        } else {
            self.config.bus_inspect_write(address, value)
        }
    }

    /// Same hard bus/CPU reset as the UI, with a monotonic external cycle timeline.
    pub fn reset(&mut self) -> Result<()> {
        let previous = self.cycles();
        self.config.input_release_all();
        self.config.bus_reset()?;
        self.cycle_offset = previous;
        self.config.cpu_reset()?;
        self.config.cpu_sync_bus()?;
        self.last_frame = None;
        Ok(())
    }

    pub fn programmer_key(&mut self) {
        self.config.progkey();
    }
    pub fn mouse_mode(&mut self, mode: MouseMode) {
        self.config.set_mouse_mode(mode);
    }
    pub fn bus_frequency(&mut self, hz: u64) {
        self.config.set_bus_frequency(hz);
    }
    pub fn debug_framebuffers(&mut self, enabled: bool) -> Result<()> {
        if let EmulatorConfig::Compact(cpu) = &mut self.config {
            cpu.bus.video.debug_framebuffers = enabled;
            Ok(())
        } else {
            anyhow::bail!("Debug framebuffers are only available on compact models")
        }
    }

    /// All Snow floppy loaders use this session's explicit padding/weak-bit RNG.
    pub fn insert_floppy(
        &mut self,
        drive: usize,
        bytes: &[u8],
        seed: u64,
        write_protect: bool,
    ) -> Result<()> {
        ensure!(drive < 3, "Floppy drive must be 0..2");
        let mut image = Autodetect::load_with_noise(bytes, Some("inserted"), Noise::seeded(seed))?;
        if write_protect {
            image.set_force_wp();
        }
        image.clear_dirty();
        self.config.swim_mut().disk_insert(drive, image)
    }
    pub fn eject_floppy(&mut self, drive: usize) -> Result<()> {
        ensure!(drive < 3, "Floppy drive must be 0..2");
        self.config.swim_mut().drives[drive].eject();
        Ok(())
    }
    pub fn export_floppy(&self, drive: usize) -> Result<Vec<u8>> {
        ensure!(
            drive < 3 && self.config.swim().drives[drive].is_present(),
            "Floppy drive is unavailable"
        );
        Moof::save_vec(self.config.swim().get_active_image(drive))
    }
    pub fn floppy_rpm(&mut self, drive: usize, adjustment: i32) -> Result<()> {
        ensure!(drive < 3, "Floppy drive must be 0..2");
        self.config.swim_mut().drives[drive].rpm_adjustment = adjustment;
        Ok(())
    }
    pub fn detach_scsi(&mut self, id: usize) -> Result<()> {
        ensure!(id < 7, "SCSI ID must be 0..6");
        self.config.scsi_mut().detach_target(id);
        self.scsi_media.retain(|(existing, _)| *existing != id);
        Ok(())
    }
    pub fn attach_cdrom(&mut self, id: usize, bytes: Option<Vec<u8>>) -> Result<()> {
        ensure!(
            id < 7 && self.config.model().has_scsi(),
            "SCSI ID is unavailable"
        );
        if let Some(ref data) = bytes {
            ensure!(
                !data.is_empty() && data.len().is_multiple_of(2048),
                "CD image requires complete 2048-byte sectors"
            );
        }
        self.detach_scsi(id)?;
        self.config.scsi_mut().attach_cdrom_at(id, None);
        if let Some(bytes) = bytes {
            let data = Arc::new(Mutex::new(bytes));
            self.config
                .scsi_mut()
                .insert_cdrom_image_at(Box::new(MemoryDisk(Arc::clone(&data))), id)?;
            self.scsi_media.push((id, data));
        }
        Ok(())
    }
    pub fn export_scsi(&self, id: usize) -> Result<Vec<u8>> {
        let media = self
            .scsi_media
            .iter()
            .find(|(existing, _)| *existing == id)
            .ok_or_else(|| anyhow::anyhow!("No private block image at that ID"))?;
        Ok(media
            .1
            .lock()
            .map_err(|_| anyhow::anyhow!("Media lock poisoned"))?
            .clone())
    }
    /// Inject bytes or a complete SDLC frame at this stopped guest boundary.
    /// Returns receiver readiness before delivery, not an acceptance acknowledgement.
    /// Native SCC rules may queue SDLC frames while busy or drop data while disabled.
    pub fn serial_receive(&mut self, channel: SccCh, data: &[u8], frame: bool) -> bool {
        let ready = self.config.scc().is_rx_ready_for_data(channel);
        if frame {
            self.config.scc_mut().push_rx_frame(channel, data.to_vec());
        } else {
            self.config.scc_mut().push_rx(channel, data);
        }
        ready
    }
    pub fn serial_dcd(&mut self, channel: SccCh, asserted: bool) {
        self.config.scc_mut().set_dcd(channel, asserted);
    }
    /// Drain output explicitly; the harness records this operation for replay.
    pub fn serial_take(&mut self, channel: SccCh) -> (Vec<u8>, Vec<Vec<u8>>) {
        (
            self.config.scc_mut().take_tx(channel),
            self.config
                .scc_mut()
                .take_tx_frames(channel)
                .into_iter()
                .collect(),
        )
    }
}
