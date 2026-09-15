//! Independently laid-out translation tables; no Apple assets or GUI owner.

use crate::bus::{InspectableBus, testbus::Testbus};
use crate::cpu_m68k::cpu::CpuM68k;
use crate::cpu_m68k::pmmu::inspect::{
    RootBank, TranslationMode, TranslationQuery, TranslationRoute,
};
use crate::cpu_m68k::pmmu::regs::{RootPointerReg, TcReg, TrTranslationReg};
use crate::cpu_m68k::pmmu::walk::WalkFailureKind;
use crate::cpu_m68k::{FPU_M68881, FPU_M68882, M68020, M68030};

type Cpu<const MODEL: usize, const FPU: usize> =
    CpuM68k<Testbus<u32, u8>, { u32::MAX }, MODEL, FPU, true>;

// Missing bytes are deliberate inspector holes; execution's Testbus still supplies zero.
impl InspectableBus<u32, u8> for Testbus<u32, u8> {
    fn inspect_read(&mut self, address: u32) -> Option<u8> {
        self.mem.get(&address).copied()
    }
    fn inspect_write(&mut self, address: u32, value: u8) -> Option<()> {
        self.mem.insert(address, value);
        Some(())
    }
}

fn put(bus: &mut Testbus<u32, u8>, address: u32, value: u32) {
    for (index, byte) in value.to_be_bytes().into_iter().enumerate() {
        bus.mem.insert(address + index as u32, byte);
    }
}

fn query(fc: u8, writing: bool) -> TranslationQuery {
    TranslationQuery {
        address: 0x0040_3006,
        function_code: fc,
        writing,
        mode: TranslationMode::CacheAware,
    }
}

fn machine<const MODEL: usize, const FPU: usize>() -> Cpu<MODEL, FPU> {
    let mut cpu = Cpu::new(Testbus::new(u32::MAX));
    // Ignore upper byte, 6-bit A/B indices, 12-bit page offset. A=16 and B=3.
    cpu.regs.pmmu.tc = TcReg(0x80C8_6600);
    cpu.regs.pmmu.crp = RootPointerReg(0x7FFF_0002_0000_1000);
    put(&mut cpu.bus, 0x1040, 0x2002);
    put(&mut cpu.bus, 0x200C, 0x5001);
    cpu.bus.mem.insert(0x5006, 0xA5);
    cpu.bus.mem.insert(0x9006, 0x5A);
    cpu.pmmu_cache_ensure();
    cpu.enable_history(true);
    cpu.bus.reset_trace();
    cpu
}

fn assert_pure<const MODEL: usize, const FPU: usize>(
    cpu: &mut Cpu<MODEL, FPU>,
    action: impl FnOnce(&mut Cpu<MODEL, FPU>),
) {
    let registers = serde_json::to_value(&cpu.regs).unwrap();
    let memory = cpu.bus.mem.clone();
    let cycles = cpu.cycles;
    let atc = cpu.pmmu_atc.clone();
    let generation = cpu.pmmu_atc_generation;
    let history = cpu.read_history().map(<[_]>::to_vec);
    let trace = cpu.bus.get_trace().len();
    let prefetch = cpu.prefetch.clone();

    action(cpu);

    assert_eq!(serde_json::to_value(&cpu.regs).unwrap(), registers);
    assert_eq!(cpu.bus.mem, memory);
    assert_eq!(cpu.cycles, cycles);
    assert_eq!(cpu.pmmu_atc, atc);
    assert_eq!(cpu.pmmu_atc_generation, generation);
    assert!(cpu.read_history().map(<[_]>::to_vec) == history);
    assert_eq!(cpu.bus.get_trace().len(), trace);
    assert_eq!(cpu.prefetch, prefetch);
}

fn stale_cache<const MODEL: usize, const FPU: usize>() {
    let mut cpu = machine::<MODEL, FPU>();

    assert_pure(&mut cpu, |cpu| {
        let result = cpu.inspect_translation(query(5, false)).unwrap();
        assert_eq!(result.route, TranslationRoute::Table);
        assert_eq!(result.physical_address, Some(0x5006));
        let tables = result.tables.unwrap();
        assert_eq!(
            tables
                .descriptors
                .iter()
                .map(|step| step.address)
                .collect::<Vec<_>>(),
            [0x1040, 0x200C]
        );
        assert_eq!(tables.descriptors[0].words, [0x2002]);
        assert_eq!(tables.descriptors[1].words, [0x5001]);
        assert_eq!(tables.resolved.unwrap().offset_bits, 12);
    });

    assert_eq!(
        cpu.read_ticks_generic::<u8, false>(5, query(5, false).address)
            .unwrap(),
        0xA5
    );
    assert_eq!(cpu.bus.mem[&0x200F], 9); // Real execution marks the leaf used.
    put(&mut cpu.bus, 0x200C, 0x9001);

    assert_pure(&mut cpu, |cpu| {
        let result = cpu.inspect_translation(query(5, false)).unwrap();
        assert_eq!(result.route, TranslationRoute::Atc);
        assert_eq!(result.physical_address, Some(0x5006));
        assert_eq!(
            result.tables.unwrap().resolved.unwrap().physical_address,
            0x9006
        );
        assert_eq!(result.cache_matches_tables, Some(false));

        let uncached = cpu
            .inspect_translation(TranslationQuery {
                mode: TranslationMode::TableOnly,
                ..query(5, false)
            })
            .unwrap();
        assert_eq!(uncached.route, TranslationRoute::Table);
        assert_eq!(uncached.physical_address, Some(0x9006));

        let alias = cpu
            .inspect_translation(TranslationQuery {
                address: 0xAA40_3006,
                ..query(5, false)
            })
            .unwrap();
        assert_eq!(alias.physical_address, Some(0x5006));
    });

    assert_eq!(
        cpu.read_ticks_generic::<u8, false>(5, query(5, false).address)
            .unwrap(),
        0xA5
    );
    cpu.pmmu_cache_invalidate();

    assert_pure(&mut cpu, |cpu| {
        let result = cpu.inspect_translation(query(5, false)).unwrap();
        assert_eq!(result.physical_address, Some(0x9006));
        assert!(!result.cached.unwrap().valid);
        assert_eq!(result.cache_matches_tables, None);
    });

    assert_eq!(
        cpu.read_ticks_generic::<u8, false>(5, query(5, false).address)
            .unwrap(),
        0x5A
    );
}

#[test]
fn pmmu_inspection_preserves_state_and_explains_stale_cache() {
    stale_cache::<M68020, FPU_M68881>();
    stale_cache::<M68030, FPU_M68882>();
}

fn protections<const MODEL: usize, const FPU: usize>() {
    let mut cpu = machine::<MODEL, FPU>();
    cpu.regs.pmmu.crp = RootPointerReg(0x7FFF_0003_0000_1000);
    // Long parent descriptor: supervisor-only, write-protected, upper child index limit 3.
    put(&mut cpu.bus, 0x1080, 0x0003_0106);
    put(&mut cpu.bus, 0x1084, 0x2000);

    for (fc, writing, expected) in [
        (1, false, Some(WalkFailureKind::SupervisorOnly)),
        (5, true, Some(WalkFailureKind::WriteProtected)),
        (5, false, None),
    ] {
        assert_pure(&mut cpu, |cpu| {
            let result = cpu.inspect_translation(query(fc, writing)).unwrap();
            assert_eq!(result.failure.map(|failure| failure.kind), expected);
            assert!(result.supervisor_only && result.write_protected);
            assert_eq!(result.tables.unwrap().descriptors[0].width, 8);
        });

        assert_eq!(
            cpu.pmmu_translate(fc, query(fc, writing).address, writing)
                .is_err(),
            expected.is_some()
        );
        cpu.pmmu_cache_invalidate();
    }

    put(&mut cpu.bus, 0x1080, 0x0002_0002); // Child B=3 now exceeds limit 2.

    assert_pure(&mut cpu, |cpu| {
        let result = cpu.inspect_translation(query(5, false)).unwrap();
        let failure = result.failure.unwrap();
        assert_eq!(failure.kind, WalkFailureKind::LimitViolation);
        assert_eq!(failure.level, 2);
        assert_eq!(result.tables.unwrap().descriptors.len(), 1);
    });

    assert!(
        cpu.pmmu_translate(5, query(5, false).address, false)
            .is_err()
    );
    assert!(cpu.regs.pmmu.psr.limit_violation());
}

#[test]
fn pmmu_inspection_matches_inherited_protection_and_limits() {
    protections::<M68020, FPU_M68881>();
    protections::<M68030, FPU_M68882>();
}

#[test]
fn pmmu_inspection_explains_roots_bypass_and_bounded_failures() {
    let mut cpu = machine::<M68030, FPU_M68882>();
    cpu.regs.pmmu.tc.set_sre(true);
    cpu.regs.pmmu.srp = RootPointerReg(0x7FFF_0002_0000_3000);
    put(&mut cpu.bus, 0x3040, 0x8001); // Early termination: 18-bit offset, base rounds down.

    assert_pure(&mut cpu, |cpu| {
        let user = cpu.inspect_translation(query(2, false)).unwrap();
        let supervisor = cpu.inspect_translation(query(6, false)).unwrap();
        assert_eq!(user.root, Some(RootBank::Cpu));
        assert_eq!(user.physical_address, Some(0x5006));
        assert_eq!(supervisor.root, Some(RootBank::Supervisor));
        assert_eq!(supervisor.physical_address, Some(0x3006));
        assert_eq!(supervisor.tables.unwrap().resolved.unwrap().offset_bits, 18);
    });

    cpu.regs.pmmu.tc.set_enable(false);
    cpu.regs.pmmu.tt[0] = TrTranslationReg(0x00FF_8177); // Ignore address, FC and direction.

    assert_pure(&mut cpu, |cpu| {
        assert_eq!(
            cpu.inspect_translation(query(5, false)).unwrap().route,
            TranslationRoute::Transparent
        );
    });

    cpu.regs.pmmu.tt[0] = TrTranslationReg(0);
    assert_eq!(
        cpu.inspect_translation(query(5, true)).unwrap().route,
        TranslationRoute::Disabled
    );
    cpu.regs.pmmu.tc.set_enable(true);
    cpu.regs.pmmu.tc.set_sre(false);
    cpu.regs.pmmu.crp = RootPointerReg(0x7FFF_0003_0000_6000);
    put(&mut cpu.bus, 0x6080, 0x5001); // Second half intentionally unreadable.

    assert_pure(&mut cpu, |cpu| {
        let result = cpu.inspect_translation(query(5, false)).unwrap();
        let failure = result.failure.unwrap();
        assert_eq!(failure.kind, WalkFailureKind::UnreadableDescriptor);
        assert_eq!(failure.address, Some(0x6084));
        assert_eq!(failure.partial_words, [0x5001]);
    });

    cpu.regs.pmmu.tc = TcReg(0x8000_0000);
    assert_pure(&mut cpu, |cpu| {
        assert_eq!(
            cpu.inspect_translation(query(5, false))
                .unwrap()
                .failure
                .unwrap()
                .kind,
            WalkFailureKind::InvalidConfiguration
        );
        assert!(cpu.inspect_translation(query(8, false)).is_err());
    });
}

#[test]
fn pmmu_inspection_cache_pages_bound_scanning_and_include_flushed_slots() {
    let mut cpu = machine::<M68030, FPU_M68882>();
    cpu.pmmu_translate(5, query(5, false).address, false)
        .unwrap();
    cpu.pmmu_cache_invalidate();

    assert_pure(&mut cpu, |cpu| {
        let empty = cpu.inspect_atc(0, 8, 1).unwrap();
        assert!(empty.entries.is_empty());
        assert_eq!(empty.scanned, 8);
        assert_eq!(empty.next, Some(8));

        let page = cpu.inspect_atc(0x403, 16, 1).unwrap();
        assert_eq!(page.entries.len(), 1);
        assert_eq!(page.scanned, 1);
        assert!(!page.entries[0].valid);
        assert_eq!(page.entries[0].logical_page, 0x403000);
        assert!(cpu.inspect_atc(0, 0, 1).is_err());
        assert!(cpu.inspect_atc(0, 65537, 1).is_err());
        assert!(cpu.inspect_atc(0, 1, 4097).is_err());
        assert!(cpu.inspect_atc(usize::MAX, 1, 1).is_err());
    });
}

#[test]
fn pmmu_inspection_fault_history_retains_original_instruction_and_context() {
    use crate::cpu_m68k::cpu::HistoryEntry;
    use crate::cpu_m68k::pmmu::inspect::FaultSignal;

    let mut cpu = machine::<M68030, FPU_M68882>();
    // TST.B (A0), NOP, BRA.S * at logical $403000 / physical $5000.
    put(&mut cpu.bus, 0x5000, 0x4A10_4E71);
    put(&mut cpu.bus, 0x5004, 0x60FE_4E71);
    put(&mut cpu.bus, 0x2010, 0); // Invalid descriptor for logical $404006.
    cpu.regs.sr.set_supervisor(true);
    cpu.regs.write_a(0, 0x404006u32);
    cpu.set_pc(0x403000).unwrap();
    cpu.prefetch_refill().unwrap();
    cpu.enable_history(true);

    let prediction = cpu
        .inspect_translation(TranslationQuery {
            address: 0x404006,
            ..query(5, false)
        })
        .unwrap();
    assert_eq!(
        prediction.failure.unwrap().kind,
        WalkFailureKind::InvalidDescriptor
    );
    let _ = cpu.step(); // Exception entry may itself fail with intentionally absent vector mappings.

    let detail = cpu
        .read_history()
        .unwrap()
        .iter()
        .find_map(|entry| match entry {
            HistoryEntry::Pagefault { detail, .. } if detail.address == 0x404006 => Some(detail),
            _ => None,
        })
        .unwrap();
    assert_eq!(detail.signal, FaultSignal::PageFault);
    assert_eq!(detail.instruction_pc, Some(0x403000));
    assert_eq!(detail.function_code, 5);
    assert_eq!(detail.failure.address, Some(0x2010));
    assert_eq!(detail.failure.level, 2);
    assert_eq!(detail.status & 0x8400, 0x8400);
    assert!(detail.tables.is_some());
    assert_eq!(cpu.debug_instruction_pc, None);
}
