use crate::bus::{Address, Bus, IrqSource};
use crate::cpu_m68k::CpuM68kType;
use crate::cpu_m68k::FpuM68kType;
use crate::cpu_m68k::cpu::{CpuError, CpuM68k, Group0Details, HistoryEntry, PagefaultCause};
use crate::cpu_m68k::pmmu::regs::{RegisterPSR, RootPointerReg};
use crate::types::Long;

use anyhow::{Result, anyhow, bail};
use proc_bitfield::bitfield;
use serde::{Deserialize, Serialize};

use super::inspect::{FaultSignal, RootBank, TranslationFault, TranslationRoute};
use super::walk::{TableWalk, WalkFailure, WalkFailureKind, WalkMemory};

struct ExecutingWalk<
    'a,
    TBus,
    const MASK: Address,
    const CPU: CpuM68kType,
    const FPU: FpuM68kType,
    const MMU: bool,
> where
    TBus: Bus<Address, u8> + IrqSource,
{
    cpu: &'a mut CpuM68k<TBus, MASK, CPU, FPU, MMU>,
    mutate: bool,
    error: Option<anyhow::Error>,
}

impl<TBus, const MASK: Address, const CPU: CpuM68kType, const FPU: FpuM68kType, const MMU: bool>
    WalkMemory for ExecutingWalk<'_, TBus, MASK, CPU, FPU, MMU>
where
    TBus: Bus<Address, u8> + IrqSource,
{
    fn select_descriptor(&mut self, address: u32) {
        self.cpu.regs.pmmu.last_desc = address;
    }

    fn read_long(&mut self, address: u32) -> Result<u32, WalkFailure> {
        self.cpu
            .read_ticks_physical::<Long>(address)
            .map_err(|error| {
                let failure = WalkFailure::new(
                    WalkFailureKind::UnreadableDescriptor,
                    Some(address),
                    error.to_string(),
                );
                self.error = Some(error);
                failure
            })
    }

    fn mark_used(&mut self, address: u32, original: u32) -> Result<(), WalkFailure> {
        if self.mutate {
            self.cpu
                .write_ticks_physical::<Long>(address, original | 8)
                .map_err(|error| {
                    let failure = WalkFailure::new(
                        WalkFailureKind::DescriptorUpdate,
                        Some(address),
                        error.to_string(),
                    );
                    self.error = Some(error);
                    failure
                })?;
        }
        Ok(())
    }
}

/// Index in CpuM68k::pmmu_atc tables when URP is in use
pub(in crate::cpu_m68k) const PMMU_ATC_URP: usize = 0;
/// Index in CpuM68k::pmmu_atc tables when SRP is in use
pub(in crate::cpu_m68k) const PMMU_ATC_SRP: usize = 1;
/// Number of ATC tables in CpuM68k::pmmu_atc (one per root pointer)
pub(in crate::cpu_m68k) const PMMU_ATCS: usize = 2;

pub(in crate::cpu_m68k) fn atc_generation_default() -> u32 {
    // 0 is reserved for unused entries
    1
}

/// A resolved Address Translation Cache entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(in crate::cpu_m68k) struct PmmuAtcEntry {
    /// Full CPU function code. A colliding code replaces this software cache slot.
    pub function_code: u8,
    /// Physical page base address (low PS bits are zero).
    pub paddr: Address,
    /// Write-protect bit inherited from any table descriptor on the walk
    /// or from the leaf page descriptor.
    pub wp: bool,
    /// Supervisor-only bit inherited from any long-format descriptor on the walk.
    pub s: bool,
    /// Physical address of the leaf page descriptor (long-word aligned).
    /// Needed so writes can set the descriptor's M (modified) bit without
    /// replaying the full table walk.
    pub leaf_desc_addr: Address,
    /// Whether the M bit of the leaf descriptor is already set. When false,
    /// the next write through this entry must RMW the descriptor to set M.
    pub modified: bool,
    // If < current generation, then this entry has been flushed.
    // 0 means never used.
    #[serde(default)]
    pub generation: u32,
}

/// Custom (de)serialization for the PMMU ATC tables.
///
/// ATC is a linear table for O(1) lookup, which can get pretty large in serialized
/// form so this is stored as key/value + size instead.
pub(in crate::cpu_m68k) mod atc_serde {
    use super::{PMMU_ATCS, PmmuAtcEntry};
    use serde::de::Error as _;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    struct AtcTable {
        size: usize,
        entries: Vec<(usize, PmmuAtcEntry)>,
    }

    impl AtcTable {
        fn from_slots(slots: &[Option<PmmuAtcEntry>]) -> Self {
            Self {
                size: slots.len(),
                entries: slots
                    .iter()
                    .enumerate()
                    .filter_map(|(i, e)| e.map(|e| (i, e)))
                    .collect(),
            }
        }

        fn into_slots(self) -> Option<Vec<Option<PmmuAtcEntry>>> {
            let mut slots = vec![None; self.size];
            for (i, e) in self.entries {
                *slots.get_mut(i)? = Some(e);
            }
            Some(slots)
        }
    }

    pub(in crate::cpu_m68k) fn serialize<S>(
        atc: &[Vec<Option<PmmuAtcEntry>>; PMMU_ATCS],
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let tables = [AtcTable::from_slots(&atc[0]), AtcTable::from_slots(&atc[1])];
        tables.serialize(serializer)
    }

    pub(in crate::cpu_m68k) fn deserialize<'de, D>(
        deserializer: D,
    ) -> Result<[Vec<Option<PmmuAtcEntry>>; PMMU_ATCS], D::Error>
    where
        D: Deserializer<'de>,
    {
        let [t0, t1] = <[AtcTable; PMMU_ATCS]>::deserialize(deserializer)?;
        let err = || D::Error::custom("ATC entry index out of range for table size");
        Ok([
            t0.into_slots().ok_or_else(err)?,
            t1.into_slots().ok_or_else(err)?,
        ])
    }
}

bitfield! {
    /// Short format page descriptor
    #[derive(Clone, Copy, PartialEq, Eq, Default)]
    pub struct PmmuShortPageDescriptor(pub u32): Debug, FromStorage, IntoStorage, DerefStorage {
        /// Page address (physical address)
        pub page_addr: u32 @ 8..=31,

        pub dt: u8 @ 0..=1,
        pub wp: bool @ 2,
        pub u: bool @ 3,
        pub m: bool @ 4,
        pub l: bool @ 5,
        pub ci: bool @ 6,
        pub g: bool @ 7,
    }
}

bitfield! {
    /// Long format page descriptor (type 1 and 2)
    #[derive(Clone, Copy, PartialEq, Eq, Default)]
    pub struct PmmuLongPageDescriptor(pub u64): Debug, FromStorage, IntoStorage, DerefStorage {
        pub lsl: u32 @ 0..=31,
        pub msl: u32 @ 32..=63,

        /// Page address (physical address)
        pub page_addr: u32 @ 8..=31,

        pub dt: u8 @ 32..=33,
        pub wp: bool @ 34,
        pub u: bool @ 35,
        pub m: bool @ 36,
        pub l: bool @ 37,
        pub ci: bool @ 38,
        pub g: bool @ 39,
        pub s: bool @ 40,
        pub sg: bool @ 41,
        pub wal: u8 @ 42..=44,
        pub ral: u8 @ 45..=47,
        pub limit: u16 @ 48..=62,
        pub lu: bool @ 63,
    }
}

bitfield! {
    /// Short format table descriptor
    #[derive(Clone, Copy, PartialEq, Eq, Default)]
    pub struct PmmuShortTableDescriptor(pub u32): Debug, FromStorage, IntoStorage, DerefStorage {
        /// Table address (physical address)
        pub table_addr: u32 @ 4..=31,

        pub dt: u8 @ 0..=1,
        pub wp: bool @ 2,
        pub u: bool @ 3,
    }
}

bitfield! {
    /// Long format table descriptor
    #[derive(Clone, Copy, PartialEq, Eq, Default)]
    pub struct PmmuLongTableDescriptor(pub u64): Debug, FromStorage, IntoStorage, DerefStorage {
        pub lsl: u32 @ 0..=31,
        pub msl: u32 @ 32..=63,

        /// Table address (physical address)
        pub table_addr: u32 @ 4..=31,

        pub dt: u8 @ 32..=33,
        pub wp: bool @ 34,
        pub u: bool @ 35,
        pub s: bool @ 40,
        pub sg: bool @ 41,
        pub wal: u8 @ 42..=44,
        pub ral: u8 @ 45..=47,
        pub limit: u16 @ 48..=62,
        pub lu: bool @ 63,
    }
}

impl<
    TBus,
    const ADDRESS_MASK: Address,
    const CPU_TYPE: CpuM68kType,
    const FPU_TYPE: FpuM68kType,
    const PMMU: bool,
> CpuM68k<TBus, ADDRESS_MASK, CPU_TYPE, FPU_TYPE, PMMU>
where
    TBus: Bus<Address, u8> + IrqSource,
{
    /// Enlarges ATC size if needed by configuration
    pub(in crate::cpu_m68k) fn pmmu_cache_ensure(&mut self) {
        if !self.regs.pmmu.tc.enable() {
            return;
        }

        let cache_size =
            (Address::MAX >> (self.regs.pmmu.tc.is() + self.regs.pmmu.tc.ps() as Address)) as usize
                + 1;
        if self.pmmu_atc.iter().map(|atc| atc.len()).min().unwrap() < cache_size {
            log::debug!("Expanding cache size: {}", cache_size);
            self.pmmu_atc
                .iter_mut()
                .for_each(|atc| atc.resize(cache_size, None));
        }
    }

    /// Flushes complete ATC
    pub(in crate::cpu_m68k) fn pmmu_cache_invalidate(&mut self) {
        // Incrementing generation invalidates all entries with a lower generation,
        // making ATC flushes very cheap rather than zeroizing the entire cache.
        self.pmmu_atc_generation = self.pmmu_atc_generation.wrapping_add(1);

        if self.pmmu_atc_generation == 0 {
            // Handle wraparound with a full cache purge.
            self.pmmu_atc.iter_mut().for_each(|atc| atc.fill(None));
            self.pmmu_atc_generation = 1;
        }
    }

    /// Looks an address up in the ATC
    #[inline(always)]
    pub(in crate::cpu_m68k) fn pmmu_atc_lookup(
        &self,
        atc: usize,
        key: usize,
        function_code: u8,
    ) -> Option<PmmuAtcEntry> {
        match self.pmmu_atc[atc][key] {
            Some(e)
                if e.generation == self.pmmu_atc_generation && e.function_code == function_code =>
            {
                Some(e)
            }
            _ => None,
        }
    }

    #[inline(always)]
    pub(super) fn pmmu_rootptr(&self, fc: u8) -> RootPointerReg {
        // M68851 manual 5.1.4.2
        // + Table 3-1, M68000 Family Function Code Assignments
        //
        // FC3 is not output by the 68020 so we ignore DRP here
        if fc & (1 << 2) != 0 && self.regs.pmmu.tc.sre() {
            self.regs.pmmu.srp
        } else {
            self.regs.pmmu.crp
        }
    }

    #[inline(always)]
    pub(super) fn pmmu_atc_tableidx(&self, fc: u8) -> usize {
        // M68851 manual 5.1.4.2
        // + Table 3-1, M68000 Family Function Code Assignments
        //
        // FC3 is not output by the 68020 so we ignore DRP here
        if fc & (1 << 2) != 0 && self.regs.pmmu.tc.sre() {
            PMMU_ATC_SRP
        } else {
            PMMU_ATC_URP
        }
    }

    /// Returns true if any enabled TT register transparently maps this access.
    /// TT regions bypass the page tables and the ATC entirely.
    #[inline]
    pub(super) fn pmmu_tt_index(&self, fc: u8, vaddr: Address, writing: bool) -> Option<usize> {
        for (index, tt) in self.regs.pmmu.tt.iter().enumerate() {
            if !tt.e() {
                continue;
            }
            // FC: bits set in fc_mask are don't-cares.
            let fc_care = !tt.fc_mask() & 0b111;
            if (fc & fc_care) != (tt.fc_base() & fc_care) {
                continue;
            }
            // MC68030 UM 9.7.3: R/W=1 means read; RWM=1 ignores direction.
            if !tt.rwm() && writing == tt.rw() {
                continue;
            }
            // Address comparison uses only the top byte; low 24 bits are ignored.
            let addr_care = !tt.le_mask() & 0xFF;
            if ((vaddr >> 24) & addr_care) != (tt.le_base() & addr_care) {
                continue;
            }
            return Some(index);
        }
        None
    }

    pub(in crate::cpu_m68k) fn pmmu_translate(
        &mut self,
        fc: u8,
        vaddr: Address,
        writing: bool,
    ) -> Result<Address> {
        if !PMMU {
            return Ok(vaddr);
        }

        // CPU-space cycles do not use translation tables (MC68030 UM 9.2.1,
        // MC68851 UM 4.2.3.5). Coprocessor register decoding belongs to the CPU/bus.
        if fc == 7 {
            return Ok(vaddr);
        }

        // Transparent translation runs even when TC.E=0; TT regions are
        // identity-mapped and bypass the page tables and the ATC.
        if self.pmmu_tt_index(fc, vaddr, writing).is_some() {
            return Ok(vaddr);
        }

        if !self.regs.pmmu.tc.enable() {
            return Ok(vaddr);
        }

        // This is formally tested in PMOVE when translation is enabled
        debug_assert_eq!(
            self.regs.pmmu.tc.is()
                + self.regs.pmmu.tc.tia() as u32
                + self.regs.pmmu.tc.tib() as u32
                + self.regs.pmmu.tc.tic() as u32
                + self.regs.pmmu.tc.tid() as u32
                + self.regs.pmmu.tc.ps() as u32,
            32
        );

        let supervisor = fc & (1 << 2) != 0;
        let atc = self.pmmu_atc_tableidx(fc);
        let is_mask = Address::MAX.unbounded_shl(32 - self.regs.pmmu.tc.is());
        let page_mask = (1u32 << self.regs.pmmu.tc.ps()) - 1;
        let cache_key = ((vaddr & !is_mask) >> self.regs.pmmu.tc.ps()) as usize;
        if let Some(entry) = self.pmmu_atc_lookup(atc, cache_key, fc) {
            if !supervisor && entry.s {
                self.pmmu_record_atc_fault(
                    fc,
                    vaddr,
                    writing,
                    WalkFailureKind::SupervisorOnly,
                    entry.leaf_desc_addr,
                );
                return Err(Self::pmmu_pagefault_to_buserror(fc, vaddr, writing));
            }
            if writing && entry.wp {
                self.pmmu_record_atc_fault(
                    fc,
                    vaddr,
                    writing,
                    WalkFailureKind::WriteProtected,
                    entry.leaf_desc_addr,
                );
                return Err(Self::pmmu_pagefault_to_buserror(fc, vaddr, writing));
            }
            if writing && !entry.modified {
                // First write through an unmodified page: RMW the leaf
                // descriptor to set the M bit, then promote the ATC entry.
                self.mark_translation_modified(
                    fc,
                    vaddr,
                    entry.leaf_desc_addr,
                    TranslationRoute::Atc,
                    0,
                )?;
                self.pmmu_atc[atc][cache_key] = Some(PmmuAtcEntry {
                    modified: true,
                    ..entry
                });
            }
            return Ok(entry.paddr | (vaddr & page_mask));
        }

        let (paddr, wp, s, leaf_desc_addr, modified) =
            self.pmmu_translate_lookup::<false>(fc, vaddr, writing)?;
        let cache_key = ((vaddr & !is_mask) >> self.regs.pmmu.tc.ps()) as usize;
        self.pmmu_atc[atc][cache_key] = Some(PmmuAtcEntry {
            function_code: fc,
            paddr: paddr & !page_mask,
            wp,
            s,
            leaf_desc_addr,
            modified,
            generation: self.pmmu_atc_generation,
        });
        Ok(paddr)
    }

    /// Captures the original attempt before exception entry can replace its context.
    fn record_mmu_fault(
        &mut self,
        fc: u8,
        vaddr: u32,
        writing: bool,
        route: TranslationRoute,
        failure: WalkFailure,
        tables: Option<&TableWalk>,
    ) {
        if self.history_enabled {
            let detail = TranslationFault {
                signal: match failure.kind {
                    WalkFailureKind::InvalidConfiguration
                    | WalkFailureKind::UnsupportedRoot
                    | WalkFailureKind::DepthExceeded => FaultSignal::BackendFailure,
                    WalkFailureKind::UnreadableDescriptor | WalkFailureKind::DescriptorUpdate => {
                        FaultSignal::TableAccessFailure
                    }
                    _ => FaultSignal::PageFault,
                },
                address: vaddr,
                function_code: fc,
                writing,
                instruction_pc: self.debug_instruction_pc,
                pc_at_attempt: self.regs.pc,
                cycle: self.cycles,
                sr: self.regs.sr.0,
                status: self.regs.pmmu.psr.0,
                tc: self.regs.pmmu.tc.0,
                root: if self.pmmu_atc_tableidx(fc) == 0 {
                    RootBank::Cpu
                } else {
                    RootBank::Supervisor
                },
                root_pointer: self.pmmu_rootptr(fc).0,
                cache_generation: self.pmmu_atc_generation,
                route,
                failure,
                tables: tables.cloned(),
            };
            self.push_history(HistoryEntry::Pagefault {
                address: vaddr,
                write: writing,
                detail: Box::new(detail),
            });
        }
    }

    fn pmmu_record_atc_fault(
        &mut self,
        fc: u8,
        vaddr: u32,
        writing: bool,
        kind: WalkFailureKind,
        descriptor: u32,
    ) {
        let level = [
            self.regs.pmmu.tc.tia(),
            self.regs.pmmu.tc.tib(),
            self.regs.pmmu.tc.tic(),
            self.regs.pmmu.tc.tid(),
        ]
        .iter()
        .filter(|&&bits| bits > 0)
        .count() as u8
            + u8::from(self.regs.pmmu.tc.fcl());
        self.regs.pmmu.psr = RegisterPSR::default();

        if kind == WalkFailureKind::SupervisorOnly {
            self.regs.pmmu.psr.set_supervisor_violation(true);
        } else {
            self.regs.pmmu.psr.set_write_protected(true);
        }

        self.regs.pmmu.psr.set_level_number(level);
        self.regs.pmmu.psr.set_bus_error(true);
        let mut failure = WalkFailure::new(
            kind,
            Some(descriptor),
            "Cached page protection rejects the access",
        );
        failure.level = level;
        self.record_mmu_fault(fc, vaddr, writing, TranslationRoute::Atc, failure, None);
    }

    /// Builds the Group-0 BusError stack frame error value for a page fault.
    fn pmmu_pagefault_to_buserror(fc: u8, vaddr: Address, writing: bool) -> anyhow::Error {
        anyhow!(CpuError::BusError(Group0Details {
            function_code: fc,
            ir: 0,
            instruction: false,
            read: !writing,
            address: vaddr,
            start_pc: 0,
            size: 0,
        }))
    }

    fn mark_translation_modified(
        &mut self,
        fc: u8,
        vaddr: u32,
        descriptor: u32,
        route: TranslationRoute,
        level: u8,
    ) -> Result<()> {
        let result = self
            .read_ticks_physical::<Long>(descriptor)
            .and_then(|word| self.write_ticks_physical::<Long>(descriptor, word | 16));

        if let Err(error) = &result {
            let mut failure = WalkFailure::new(
                WalkFailureKind::DescriptorUpdate,
                Some(descriptor),
                error.to_string(),
            );
            failure.level = level;
            self.record_mmu_fault(fc, vaddr, true, route, failure, None);
        }

        result
    }

    /// Perform address translation by performing a page table lookup.
    /// Returns (physical address, wp, s, leaf descriptor address, M-bit), or error:
    ///  - bus error stack frame for translation,
    ///  - simple error on PTEST.
    pub(in crate::cpu_m68k) fn pmmu_translate_lookup<const PTEST: bool>(
        &mut self,
        fc: u8,
        vaddr: Address,
        writing: bool,
    ) -> Result<(Address, bool, bool, Address, bool)> {
        let rootptr = self.pmmu_rootptr(fc);

        if PTEST {
            self.regs.pmmu.psr = RegisterPSR::default();
            self.regs.pmmu.last_desc = 0;
        }

        let tc = self.regs.pmmu.tc;
        let mut reader = ExecutingWalk {
            cpu: self,
            mutate: !PTEST,
            error: None,
        };
        let walk = super::walk::walk_tables(&mut reader, tc, rootptr, vaddr, fc);
        let level = walk.level;

        if let Some(failure) = &walk.failure {
            if let Some(error) = reader.error.take() {
                if !PTEST {
                    self.record_mmu_fault(
                        fc,
                        vaddr,
                        writing,
                        TranslationRoute::Table,
                        failure.clone(),
                        Some(&walk),
                    );
                }
                return Err(error);
            }

            let cause = match failure.kind {
                super::walk::WalkFailureKind::InvalidDescriptor => PagefaultCause::Invalid,
                super::walk::WalkFailureKind::LimitViolation => PagefaultCause::LimitViolation,
                _ => {
                    if !PTEST {
                        self.record_mmu_fault(
                            fc,
                            vaddr,
                            writing,
                            TranslationRoute::Table,
                            failure.clone(),
                            Some(&walk),
                        );
                    }
                    bail!("{}", failure.detail);
                }
            };

            if !PTEST {
                self.regs.pmmu.psr = RegisterPSR::default();
            }

            match cause {
                PagefaultCause::Invalid => self.regs.pmmu.psr.set_invalid(true),
                PagefaultCause::LimitViolation => self.regs.pmmu.psr.set_limit_violation(true),
                _ => unreachable!(),
            }
            self.regs.pmmu.psr.set_level_number(level);

            if PTEST {
                return Err(anyhow!(CpuError::Pagefault(cause)));
            }

            self.regs.pmmu.psr.set_bus_error(true);
            self.record_mmu_fault(
                fc,
                vaddr,
                writing,
                TranslationRoute::Table,
                failure.clone(),
                Some(&walk),
            );
            return Err(Self::pmmu_pagefault_to_buserror(fc, vaddr, writing));
        }

        let page = walk.resolved.expect("successful walk contains a page");
        let wp = page.write_protected;
        let s = page.supervisor_only;
        let leaf_desc_addr = page.leaf_address;
        let mut modified = page.modified;

        // Enforce supervisor-only access on the resolved page
        let supervisor = fc & (1 << 2) != 0;
        if !supervisor && s {
            if !PTEST {
                self.regs.pmmu.psr = RegisterPSR::default();
            }
            self.regs.pmmu.psr.set_supervisor_violation(true);
            self.regs.pmmu.psr.set_level_number(level);
            if PTEST {
                return Err(anyhow!(CpuError::Pagefault(PagefaultCause::SupervisorOnly)));
            } else {
                self.regs.pmmu.psr.set_bus_error(true);
                let mut failure = WalkFailure::new(
                    WalkFailureKind::SupervisorOnly,
                    Some(leaf_desc_addr),
                    "Inherited supervisor protection rejects the access",
                );
                failure.level = level;
                self.record_mmu_fault(
                    fc,
                    vaddr,
                    writing,
                    TranslationRoute::Table,
                    failure,
                    Some(&walk),
                );
                return Err(Self::pmmu_pagefault_to_buserror(fc, vaddr, writing));
            }
        }

        // Enforce write-protect on the resolved page
        if writing && wp {
            if !PTEST {
                self.regs.pmmu.psr = RegisterPSR::default();
            }
            self.regs.pmmu.psr.set_write_protected(true);
            self.regs.pmmu.psr.set_level_number(level);
            if PTEST {
                return Err(anyhow!(CpuError::Pagefault(PagefaultCause::WriteProtected)));
            } else {
                self.regs.pmmu.psr.set_bus_error(true);
                let mut failure = WalkFailure::new(
                    WalkFailureKind::WriteProtected,
                    Some(leaf_desc_addr),
                    "Inherited write protection rejects the access",
                );
                failure.level = level;
                self.record_mmu_fault(
                    fc,
                    vaddr,
                    writing,
                    TranslationRoute::Table,
                    failure,
                    Some(&walk),
                );
                return Err(Self::pmmu_pagefault_to_buserror(fc, vaddr, writing));
            }
        }

        // Set M on the leaf descriptor on first write. PTEST never mutates the
        // tables; only real translations do.
        if !PTEST && writing && !modified {
            self.mark_translation_modified(
                fc,
                vaddr,
                leaf_desc_addr,
                TranslationRoute::Table,
                level,
            )?;
            modified = true;
        }

        let paddr = page.physical_address;

        if PTEST {
            self.regs.pmmu.psr.set_level_number(level);
        }
        Ok((paddr, wp, s, leaf_desc_addr, modified))
    }
}

#[cfg(all(test, feature = "savestates"))]
mod tests {
    use super::*;

    fn sample_entry(paddr: Address) -> PmmuAtcEntry {
        PmmuAtcEntry {
            function_code: 5,
            paddr,
            wp: true,
            s: false,
            leaf_desc_addr: paddr + 0x10,
            modified: true,
            generation: 1,
        }
    }

    #[test]
    fn atc_serde_roundtrip() {
        let mut table0 = vec![None; 1024];
        table0[3] = Some(sample_entry(0x1000));
        table0[1000] = Some(sample_entry(0x2000));
        let mut table1 = vec![None; 64];
        table1[0] = Some(sample_entry(0x3000));

        let atc = [table0, table1];

        #[derive(serde::Serialize, serde::Deserialize)]
        struct Wrap {
            #[serde(with = "super::atc_serde")]
            atc: [Vec<Option<PmmuAtcEntry>>; PMMU_ATCS],
        }

        let bytes = postcard::to_allocvec(&Wrap { atc: atc.clone() }).unwrap();
        let restored: Wrap = postcard::from_bytes(&bytes).unwrap();

        assert_eq!(restored.atc[0].len(), 1024);
        assert_eq!(restored.atc[1].len(), 64);
        assert_eq!(restored.atc, atc);
    }
}
