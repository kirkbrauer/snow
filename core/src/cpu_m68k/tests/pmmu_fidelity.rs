//! Original tables specified from MC68030 UM 9.3/9.4/9.5.5.1 and MC68851 UM 5.1.

use crate::bus::testbus::Testbus;
use crate::cpu_m68k::cpu::CpuM68k;
use crate::cpu_m68k::pmmu::inspect::{
    TranslationInspection, TranslationMode, TranslationQuery, TranslationRoute,
};
use crate::cpu_m68k::pmmu::regs::{RootPointerReg, TcReg, TrTranslationReg};
use crate::cpu_m68k::pmmu::walk::{DescriptorIndex, WalkFailureKind};
use crate::cpu_m68k::{FPU_M68881, FPU_M68882, M68020, M68030};

type Cpu<const MODEL: usize, const FPU: usize> =
    CpuM68k<Testbus<u32, u8>, { u32::MAX }, MODEL, FPU, true>;

const LOGICAL: u32 = 0x0040_3006;

fn put<const MODEL: usize, const FPU: usize>(cpu: &mut Cpu<MODEL, FPU>, address: u32, value: u32) {
    for (offset, byte) in value.to_be_bytes().into_iter().enumerate() {
        cpu.bus.mem.insert(address + offset as u32, byte);
    }
}

fn query(function_code: u8, writing: bool) -> TranslationQuery {
    TranslationQuery {
        address: LOGICAL,
        function_code,
        writing,
        mode: TranslationMode::CacheAware,
    }
}

fn ordinary<const MODEL: usize, const FPU: usize>() -> Cpu<MODEL, FPU> {
    let mut cpu = Cpu::new(Testbus::new(u32::MAX));
    cpu.regs.pmmu.tc = TcReg(0x80C8_6600);
    cpu.regs.pmmu.crp = RootPointerReg(0x7FFF_0002_0000_1000);

    put(&mut cpu, 0x1040, 0x2002);
    put(&mut cpu, 0x200C, 0x5001);
    cpu.bus.mem.insert(0x5006, 0xA5);
    cpu.bus.mem.insert(0x9006, 0x5A);
    cpu.bus.mem.insert(LOGICAL, 0xCC);
    cpu.pmmu_cache_ensure();
    cpu.enable_history(true);

    cpu
}

fn inspect<const MODEL: usize, const FPU: usize>(
    cpu: &mut Cpu<MODEL, FPU>,
    request: TranslationQuery,
) -> TranslationInspection {
    let registers = serde_json::to_value(&cpu.regs).unwrap();
    let memory = cpu.bus.mem.clone();
    let cycles = cpu.cycles;
    let atc = cpu.pmmu_atc.clone();
    let generation = cpu.pmmu_atc_generation;
    let prefetch = cpu.prefetch.clone();
    let history = cpu.read_history().map(<[_]>::to_vec);
    let trace = cpu.bus.get_trace().len();
    let observed = cpu.inspect_translation(request).unwrap();

    assert_eq!(serde_json::to_value(&cpu.regs).unwrap(), registers);
    assert_eq!(cpu.bus.mem, memory);
    assert_eq!(cpu.cycles, cycles);
    assert_eq!(cpu.pmmu_atc, atc);
    assert_eq!(cpu.pmmu_atc_generation, generation);
    assert_eq!(cpu.prefetch, prefetch);
    assert!(cpu.read_history().map(<[_]>::to_vec) == history);
    assert_eq!(cpu.bus.get_trace().len(), trace);

    observed
}

fn function_lookup<const MODEL: usize, const FPU: usize>(supervisor_root: bool) {
    let mut cpu = ordinary::<MODEL, FPU>();
    cpu.regs.pmmu.tc.set_fcl(true);
    cpu.regs.pmmu.tc.set_sre(supervisor_root);
    // FCL ignores the root limit; the F table has eight entries.
    cpu.regs.pmmu.crp = RootPointerReg(0x0000_0002_0000_1000);
    cpu.regs.pmmu.srp = RootPointerReg(0xFFFF_0002_0000_8000);

    for fc in [1_u8, 2, 5, 6] {
        let a = 0x2000 + u32::from(fc) * 0x100;
        let b = 0x4000 + u32::from(fc) * 0x100;
        let page = 0x10000 + u32::from(fc) * 0x1000;
        let root = if supervisor_root && fc & 4 != 0 {
            0x8000
        } else {
            0x1000
        };

        put(&mut cpu, root + u32::from(fc) * 4, a | 2);
        put(&mut cpu, a + 0x40, b | 2);
        put(&mut cpu, b + 0x0C, page | 1);
        cpu.bus.mem.insert(page + 6, fc);
    }

    for fc in [1_u8, 2, 5, 6, 1, 6, 2, 5] {
        let observed = inspect(&mut cpu, query(fc, false));

        assert_eq!(
            observed.physical_address,
            Some(0x10006 + u32::from(fc) * 0x1000)
        );
        assert_eq!(observed.tables.as_ref().unwrap().level, 3);
        assert_eq!(
            observed.tables.as_ref().unwrap().descriptors[0].index,
            u32::from(fc)
        );
        assert_eq!(
            observed.tables.as_ref().unwrap().descriptors[0].parent_limit,
            None
        );
        assert_eq!(
            cpu.read_ticks_generic::<u8, false>(fc, LOGICAL).unwrap(),
            fc
        );
    }
}

#[test]
fn function_code_level_precedes_address_bits_and_ignores_root_limit() {
    for supervisor_root in [false, true] {
        function_lookup::<M68020, FPU_M68881>(supervisor_root);
        function_lookup::<M68030, FPU_M68882>(supervisor_root);
    }
}

fn cache_tags<const MODEL: usize, const FPU: usize>() {
    let mut cpu = ordinary::<MODEL, FPU>();

    assert_eq!(
        cpu.read_ticks_generic::<u8, false>(5, LOGICAL).unwrap(),
        0xA5
    );

    put(&mut cpu, 0x200C, 0x9001);

    let observed = cpu.inspect_translation(query(1, false)).unwrap();

    assert_eq!(observed.route, TranslationRoute::Table);
    assert_eq!(observed.physical_address, Some(0x9006));
    assert_eq!(observed.cache_matches_tables, None);
    assert_eq!(observed.cached.as_ref().unwrap().function_code, 5);
    assert!(observed.cached.as_ref().unwrap().valid); // Generation is distinct from tag match.
    assert_eq!(
        cpu.read_ticks_generic::<u8, false>(1, LOGICAL).unwrap(),
        0x5A
    );

    let replaced = inspect(&mut cpu, query(5, false));

    assert_eq!(replaced.cached.unwrap().function_code, 1);
    assert_eq!(replaced.route, TranslationRoute::Table);
    assert_eq!(replaced.physical_address, Some(0x9006));
}

#[test]
fn cache_entries_do_not_leak_between_function_codes_without_fcl() {
    cache_tags::<M68020, FPU_M68881>();
    cache_tags::<M68030, FPU_M68882>();
}

#[test]
fn transparent_direction_uses_bus_read_polarity_in_both_registers() {
    for register in [0, 1] {
        for transparent_read in [false, true] {
            for writing in [false, true] {
                let mut cpu = ordinary::<M68030, FPU_M68882>();
                cpu.regs.pmmu.tt[register] =
                    TrTranslationReg(0x0000_8050 | if transparent_read { 0x200 } else { 0 });
                let expected = if writing != transparent_read {
                    LOGICAL
                } else {
                    0x5006
                };
                let observed = inspect(&mut cpu, query(5, writing));

                assert_eq!(observed.physical_address, Some(expected));
                assert_eq!(cpu.pmmu_translate(5, LOGICAL, writing).unwrap(), expected);
            }
        }
    }
}

fn five_levels<const MODEL: usize, const FPU: usize>() {
    let mut cpu = ordinary::<MODEL, FPU>();
    cpu.regs.pmmu.tc = TcReg(0x81C8_4422);
    cpu.regs.pmmu.crp = RootPointerReg(0x0000_0002_0000_1000);

    for (address, descriptor) in [
        (0x1014, 0x2002),
        (0x2010, 0x3002),
        (0x3000, 0x4002),
        (0x4000, 0x5002),
        (0x500C, 0x9005),
    ] {
        put(&mut cpu, address, descriptor);
    }

    let observed = inspect(&mut cpu, query(5, false));
    let walk = observed.tables.unwrap();

    assert_eq!(walk.level, 5);
    assert_eq!(walk.resolved.unwrap().physical_address, 0x9006);
    assert_eq!(
        walk.descriptors
            .iter()
            .map(|step| step.index_source)
            .collect::<Vec<_>>(),
        [
            DescriptorIndex::FunctionCode,
            DescriptorIndex::TableA,
            DescriptorIndex::TableB,
            DescriptorIndex::TableC,
            DescriptorIndex::TableD
        ]
    );
    assert_eq!(
        walk.descriptors
            .iter()
            .map(|step| step.index)
            .collect::<Vec<_>>(),
        [5, 4, 0, 0, 3]
    );
    assert_eq!(
        cpu.read_ticks_generic::<u8, false>(5, LOGICAL).unwrap(),
        0x5A
    );

    let failure = inspect(&mut cpu, query(5, true)).failure.unwrap();

    assert_eq!(failure.kind, WalkFailureKind::WriteProtected);
    assert_eq!(failure.level, 5);
    assert!(cpu.pmmu_translate(5, LOGICAL, true).is_err());
    assert_eq!(cpu.regs.pmmu.psr.level_number(), 5);
    assert_eq!(cpu.bus.mem[&0x500F], 13); // U set, M remains clear after rejected write.

    // Early termination in the F table consumes no logical address bits.
    put(&mut cpu, 0x1014, 0x8000_0001);
    cpu.pmmu_cache_invalidate();

    let early = inspect(&mut cpu, query(5, true));

    assert_eq!(early.physical_address, Some(0x8040_3006));
    assert_eq!(early.tables.unwrap().resolved.unwrap().offset_bits, 24);
    assert_eq!(cpu.pmmu_translate(5, LOGICAL, true).unwrap(), 0x8040_3006);
    assert_eq!(cpu.bus.mem[&0x1017], 25); // F-level page gets both U and M.
}

#[test]
fn fcl_supports_all_five_levels_protection_and_early_termination() {
    five_levels::<M68020, FPU_M68881>();
    five_levels::<M68030, FPU_M68882>();
}

fn long_function_table<const MODEL: usize, const FPU: usize>() {
    let mut cpu = ordinary::<MODEL, FPU>();
    cpu.regs.pmmu.tc.set_fcl(true);
    cpu.regs.pmmu.crp = RootPointerReg(0xFFFF_0003_0000_1000);

    for (address, word) in [
        (0x1028, 0x0010_0003),
        (0x102C, 0x2000),
        (0x2080, 0x0003_0003),
        (0x2084, 0x3000),
        (0x3018, 0x0000_0001),
        (0x301C, 0x9000),
    ] {
        put(&mut cpu, address, word);
    }

    let observed = inspect(&mut cpu, query(5, false));

    assert_eq!(observed.physical_address, Some(0x9006));
    assert_eq!(
        observed
            .tables
            .unwrap()
            .descriptors
            .iter()
            .map(|step| step.width)
            .collect::<Vec<_>>(),
        [8, 8, 8]
    );
    assert_eq!(
        cpu.read_ticks_generic::<u8, false>(5, LOGICAL).unwrap(),
        0x5A
    );

    cpu.pmmu_cache_invalidate();
    put(&mut cpu, 0x1028, 0x000F_0003); // Child upper bound still applies to A=16.

    let failed = inspect(&mut cpu, query(5, false)).failure.unwrap();

    assert_eq!(failed.kind, WalkFailureKind::LimitViolation);
    assert_eq!(failed.level, 2);
    assert!(cpu.pmmu_translate(5, LOGICAL, false).is_err());
    assert_eq!(cpu.regs.pmmu.psr.level_number(), 2);
    assert!(cpu.regs.pmmu.psr.limit_violation());

    put(&mut cpu, 0x1028, 0x0010_0003);
    cpu.bus.mem.remove(&0x102F);

    let hole = inspect(&mut cpu, query(5, false)).failure.unwrap();

    assert_eq!(hole.kind, WalkFailureKind::UnreadableDescriptor);
    assert_eq!(hole.level, 1);
    assert_eq!(hole.partial_words, [0x0010_0003]);
    assert_eq!(hole.partial_bytes, [0, 0, 0x20]);

    put(&mut cpu, 0x102C, 0x2000);
    put(&mut cpu, 0x1028, 0);

    let invalid = inspect(&mut cpu, query(5, false)).failure.unwrap();

    assert_eq!(invalid.kind, WalkFailureKind::InvalidDescriptor);
    assert_eq!(invalid.level, 1);
    assert!(cpu.pmmu_translate(5, LOGICAL, false).is_err());
    assert!(cpu.regs.pmmu.psr.invalid());
    assert_eq!(cpu.regs.pmmu.psr.level_number(), 1);
}

#[test]
fn fcl_long_roots_preserve_child_limits_and_failure_boundaries() {
    long_function_table::<M68020, FPU_M68881>();
    long_function_table::<M68030, FPU_M68882>();
}

#[test]
fn transparent_masks_and_enable_are_independent_of_paging() {
    for paging in [false, true] {
        for fc in 0..=6 {
            for writing in [false, true] {
                let mut cpu = ordinary::<M68030, FPU_M68882>();
                cpu.regs.pmmu.tc.set_enable(paging);
                // Ignore all address and function-code bits, both directions.
                cpu.regs.pmmu.tt[1] = TrTranslationReg(0xFFFF_8177);
                let observation = inspect(&mut cpu, query(fc, writing));

                assert_eq!(observation.route, TranslationRoute::Transparent);
                assert_eq!(observation.transparent_register, Some(1));
                assert_eq!(cpu.pmmu_translate(fc, LOGICAL, writing).unwrap(), LOGICAL);
                assert!(cpu.inspect_atc(0, 65536, 1).unwrap().entries.is_empty());
            }
        }
    }

    let mut cpu = ordinary::<M68030, FPU_M68882>();
    cpu.regs.pmmu.tt[0] = TrTranslationReg(0x0100_8150); // Wrong top byte.

    assert_eq!(
        inspect(&mut cpu, query(5, false)).route,
        TranslationRoute::Table
    );

    cpu.regs.pmmu.tt[0] = TrTranslationReg(0x0000_8140); // Wrong FC.

    assert_eq!(
        inspect(&mut cpu, query(5, false)).route,
        TranslationRoute::Table
    );

    cpu.regs.pmmu.tt[0] = TrTranslationReg(0x0000_8150);
    cpu.regs.pmmu.tt[0].set_e(false);

    assert_eq!(
        inspect(&mut cpu, query(5, false)).route,
        TranslationRoute::Table
    );
}

fn cpu_space<const MODEL: usize, const FPU: usize>() {
    let mut cpu = ordinary::<MODEL, FPU>();
    cpu.regs.pmmu.crp = RootPointerReg(0);
    cpu.regs.pmmu.tt[0] = TrTranslationReg(0x00FF_8177);

    for writing in [false, true] {
        let observed = inspect(&mut cpu, query(7, writing));

        assert_eq!(observed.route, TranslationRoute::CpuSpace);
        assert_eq!(observed.physical_address, Some(LOGICAL));
        assert!(observed.tables.is_none());
        assert!(observed.cached.is_none());
        assert!(observed.transparent_register.is_none());
        assert_eq!(cpu.pmmu_translate(7, LOGICAL, writing).unwrap(), LOGICAL);
        assert!(cpu.inspect_atc(0, 65536, 1).unwrap().entries.is_empty());
    }
}

#[test]
fn cpu_space_never_consults_transparent_registers_tables_or_atc() {
    cpu_space::<M68020, FPU_M68881>();
    cpu_space::<M68030, FPU_M68882>();
}
