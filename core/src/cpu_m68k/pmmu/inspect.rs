//! Read-only prediction of Snow's current MMU behavior, including cache/table differences.

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

use crate::bus::{Address, Bus, InspectableBus, IrqSource};
use crate::cpu_m68k::cpu::CpuM68k;
use crate::cpu_m68k::{CpuM68kType, FpuM68kType};

use super::walk::{
    ResolvedPage, TableWalk, WalkFailure, WalkFailureKind, WalkMemory, protection_failure,
    walk_tables,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TranslationMode {
    CacheAware,
    TableOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TranslationRoute {
    NoMmu,
    Disabled,
    Transparent,
    Atc,
    Table,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RootBank {
    Cpu,
    Supervisor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranslationQuery {
    pub address: u32,
    pub function_code: u8,
    pub writing: bool,
    pub mode: TranslationMode,
}

/// A slot in Snow's linear software cache, not the physical chip's ATC layout.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AtcSlot {
    pub bank: RootBank,
    pub index: usize,
    /// Logical page interpreted using the current TC; IS aliases are not expanded.
    pub logical_page: u32,
    pub physical_page: u32,
    pub write_protected: bool,
    pub supervisor_only: bool,
    pub leaf_address: u32,
    pub modified: bool,
    pub generation: u32,
    /// Generation match only. A valid entry can disagree with edited tables.
    pub valid: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AtcPage {
    pub generation: u32,
    pub tc: u32,
    pub table_sizes: [usize; 2],
    pub start: usize,
    pub scanned: usize,
    pub next: Option<usize>,
    pub entries: Vec<AtcSlot>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranslationInspection {
    pub query: TranslationQuery,
    pub route: TranslationRoute,
    pub root: Option<RootBank>,
    pub root_pointer: Option<u64>,
    pub transparent_register: Option<u8>,
    pub cache_generation: u32,
    pub page_bits: u8,
    pub ignored_bits: u8,
    pub cached: Option<AtcSlot>,
    pub tables: Option<TableWalk>,
    /// Compares address and inherited protection; None means no valid cache candidate.
    pub cache_matches_tables: Option<bool>,
    pub physical_address: Option<u32>,
    /// Address presented to the machine bus after the CPU address-width mask.
    pub bus_address: Option<u32>,
    pub write_protected: bool,
    pub supervisor_only: bool,
    pub failure: Option<WalkFailure>,
    /// U/M writes execution would attempt; not a guarantee those bus writes succeed.
    pub descriptor_effects: Vec<DescriptorEffect>,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DescriptorEffect {
    pub address: u32,
    /// Bit mask ORed into the descriptor's first long word: U=8, M=16.
    pub set_bits: u32,
}

/// An observed failed translation attempt. Program-prefetch errors can be deferred;
/// this event alone does not establish that an exception was dispatched.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranslationFault {
    pub signal: FaultSignal,
    pub address: u32,
    pub function_code: u8,
    pub writing: bool,
    pub instruction_pc: Option<u32>,
    pub pc_at_attempt: u32,
    pub cycle: u64,
    pub sr: u16,
    pub status: u16,
    pub tc: u32,
    pub root: RootBank,
    pub root_pointer: u64,
    pub cache_generation: u32,
    pub route: TranslationRoute,
    pub failure: WalkFailure,
    pub tables: Option<TableWalk>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FaultSignal {
    PageFault,
    TableAccessFailure,
    BackendFailure,
}

struct InspectionMemory<'a, T> {
    bus: &'a mut T,
}

impl<T: InspectableBus<Address, u8>> WalkMemory for InspectionMemory<'_, T> {
    fn select_descriptor(&mut self, _address: u32) {}

    fn read_long(&mut self, address: u32) -> Result<u32, WalkFailure> {
        let mut bytes = [0; 4];
        let mut partial = Vec::new();

        for (offset, byte) in bytes.iter_mut().enumerate() {
            let current = address.wrapping_add(offset as u32);
            *byte = self.bus.inspect_read(current).ok_or_else(|| {
                let mut failure = WalkFailure::new(
                    WalkFailureKind::UnreadableDescriptor,
                    Some(current),
                    "Descriptor byte is outside safe physical RAM/ROM inspection",
                );
                failure.partial_bytes.clone_from(&partial);
                failure
            })?;
            partial.push(*byte);
        }

        Ok(u32::from_be_bytes(bytes))
    }

    fn mark_used(&mut self, _address: u32, _original: u32) -> Result<(), WalkFailure> {
        Ok(())
    }
}

impl<TBus, const MASK: Address, const CPU: CpuM68kType, const FPU: FpuM68kType, const MMU: bool>
    CpuM68k<TBus, MASK, CPU, FPU, MMU>
where
    TBus: Bus<Address, u8> + IrqSource + InspectableBus<Address, u8>,
{
    /// Predict one byte access without timed reads, status updates, cache fills or U/M writes.
    /// Table-only mode bypasses the ATC, but retains disabled and transparent bypass behavior.
    pub fn inspect_translation(
        &mut self,
        query: TranslationQuery,
    ) -> Result<TranslationInspection> {
        ensure!(
            query.function_code <= 7,
            "Function code must fit three bits"
        );
        let tc = self.regs.pmmu.tc;
        let bank_index = self.pmmu_atc_tableidx(query.function_code);
        let root_pointer = self.pmmu_rootptr(query.function_code);
        let mut result = TranslationInspection {
            query,
            route: TranslationRoute::NoMmu,
            root: MMU.then_some(if bank_index == 0 {
                RootBank::Cpu
            } else {
                RootBank::Supervisor
            }),
            root_pointer: MMU.then_some(root_pointer.0),
            transparent_register: None,
            cache_generation: self.pmmu_atc_generation,
            page_bits: tc.ps(),
            ignored_bits: tc.is() as u8,
            cached: None,
            tables: None,
            cache_matches_tables: None,
            physical_address: Some(query.address),
            bus_address: Some(query.address & MASK),
            write_protected: false,
            supervisor_only: false,
            failure: None,
            descriptor_effects: Vec::new(),
            limitations: Vec::new(),
        };

        if !MMU {
            return Ok(result);
        }

        if tc.fcl() {
            result.limitations.push(
                "Snow ignores TC.FCL and does not perform a function-code lookup level".into(),
            );
        }
        result.limitations.push("Snow's software ATC is indexed by root bank and logical page, not physical-chip ATC organization".into());

        if self.regs.pmmu.tt.iter().any(|tt| tt.e() && !tt.rwm()) {
            result.limitations.push(
                "Snow's transparent R/W match is inverted relative to hardware when RWM is clear"
                    .into(),
            );
        }

        if let Some(index) = self.pmmu_tt_index(query.function_code, query.address, query.writing) {
            result.route = TranslationRoute::Transparent;
            result.transparent_register = Some(index as u8);
            return Ok(result);
        }

        if !tc.enable() {
            result.route = TranslationRoute::Disabled;
            return Ok(result);
        }

        result.route = TranslationRoute::Table;
        result.physical_address = None;
        result.bus_address = None;
        let indices = [tc.tia(), tc.tib(), tc.tic(), tc.tid()];

        if tc.is() + u32::from(tc.ps()) + indices.iter().map(|&v| u32::from(v)).sum::<u32>() != 32 {
            result.failure = Some(WalkFailure::new(
                WalkFailureKind::InvalidConfiguration,
                None,
                "TC initial shift, table indices and page offset must total 32 bits",
            ));
            return Ok(result);
        }

        let ignored_mask = u32::MAX.unbounded_shl(32 - tc.is());
        let key = ((query.address & !ignored_mask) >> tc.ps()) as usize;
        result.cached = self.inspection_atc_slot(bank_index, key);
        let mut reader = InspectionMemory { bus: &mut self.bus };
        let tables = walk_tables(&mut reader, tc, root_pointer, query.address);
        let cached_page = result
            .cached
            .as_ref()
            .filter(|slot| slot.valid)
            .map(|slot| ResolvedPage {
                physical_address: slot.physical_page | (query.address & ((1 << tc.ps()) - 1)),
                offset_bits: tc.ps(),
                write_protected: slot.write_protected,
                supervisor_only: slot.supervisor_only,
                leaf_address: slot.leaf_address,
                modified: slot.modified,
            });

        result.cache_matches_tables = cached_page.map(|cached| {
            tables.resolved.is_some_and(|table| {
                cached.physical_address == table.physical_address
                    && cached.write_protected == table.write_protected
                    && cached.supervisor_only == table.supervisor_only
            })
        });

        let selected = if query.mode == TranslationMode::CacheAware && cached_page.is_some() {
            result.route = TranslationRoute::Atc;
            cached_page
        } else {
            result.failure = tables.failure.clone();
            tables.resolved
        };

        if let Some(page) = selected {
            result.write_protected = page.write_protected;
            result.supervisor_only = page.supervisor_only;

            if let Some(kind) = protection_failure(page, query.function_code, query.writing) {
                let mut failure = WalkFailure::new(
                    kind,
                    Some(page.leaf_address),
                    "Requested access is rejected by inherited page protection",
                );
                failure.level = if result.route == TranslationRoute::Atc {
                    indices.iter().filter(|&&bits| bits > 0).count() as u8
                } else {
                    tables.level
                };
                result.failure = Some(failure);
            } else {
                result.physical_address = Some(page.physical_address);
                result.bus_address = Some(page.physical_address & MASK);

                if query.writing && !page.modified {
                    result.descriptor_effects.push(DescriptorEffect {
                        address: page.leaf_address,
                        set_bits: 16,
                    });
                }
            }
        }

        if result.route == TranslationRoute::Table {
            let used = tables
                .descriptors
                .iter()
                .filter(|step| !step.used && step.descriptor_type != 0)
                .map(|step| DescriptorEffect {
                    address: step.address,
                    set_bits: 8,
                });
            result.descriptor_effects.splice(0..0, used);
        }

        result.tables = Some(tables);
        Ok(result)
    }

    fn inspection_atc_slot(&self, bank: usize, index: usize) -> Option<AtcSlot> {
        let entry = self.pmmu_atc.get(bank)?.get(index)?.as_ref()?;
        Some(AtcSlot {
            bank: if bank == 0 {
                RootBank::Cpu
            } else {
                RootBank::Supervisor
            },
            index,
            logical_page: (index as u32) << self.regs.pmmu.tc.ps(),
            physical_page: entry.paddr,
            write_protected: entry.wp,
            supervisor_only: entry.s,
            leaf_address: entry.leaf_desc_addr,
            modified: entry.modified,
            generation: entry.generation,
            valid: entry.generation == self.pmmu_atc_generation,
        })
    }

    /// Bounded sparse cache enumeration. A cursor spans CPU slots then supervisor slots.
    pub fn inspect_atc(
        &self,
        start: usize,
        scan_limit: usize,
        entry_limit: usize,
    ) -> Result<AtcPage> {
        ensure!(
            (1..=65536).contains(&scan_limit),
            "ATC scan limit must be 1..65536"
        );
        ensure!(
            (1..=4096).contains(&entry_limit),
            "ATC entry limit must be 1..4096"
        );
        let sizes = [self.pmmu_atc[0].len(), self.pmmu_atc[1].len()];
        let total = sizes[0] + sizes[1];
        ensure!(start <= total, "ATC cursor exceeds cache storage");
        let end = total.min(start.saturating_add(scan_limit));
        let mut entries = Vec::new();
        let mut cursor = start;

        while cursor < end && entries.len() < entry_limit {
            let (bank, index) = if cursor < sizes[0] {
                (0, cursor)
            } else {
                (1, cursor - sizes[0])
            };

            if let Some(slot) = self.inspection_atc_slot(bank, index) {
                entries.push(slot);
            }

            cursor += 1;
        }

        Ok(AtcPage {
            generation: self.pmmu_atc_generation,
            tc: self.regs.pmmu.tc.0,
            table_sizes: sizes,
            start,
            scanned: cursor - start,
            next: (cursor < total).then_some(cursor),
            entries,
        })
    }
}
