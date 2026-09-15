//! Native step-over targets must match the guest's actual call-return boundary.

use crate::bus::{Address, testbus::Testbus};
use crate::cpu_m68k::{
    FPU_M68881, FPU_M68882, FPU_NONE, M68000, M68000_ADDRESS_MASK, M68020, M68020_ADDRESS_MASK,
    M68030, M68030_ADDRESS_MASK,
    cpu::{Breakpoint, CpuM68k},
};

type CallCase = (&'static str, &'static [u16], u32, u32, u32);

fn calls<const MASK: Address, const CPU: usize, const FPU: usize, const PMMU: bool>() {
    // Instructions, A0, D0, destination. PC-relative addresses use the extension
    // word's address; return addresses follow the complete instruction.
    let mut cases: Vec<CallCase> = vec![
        ("JSR (A0)", &[0x4e90], 0x1100, 0, 0x1100),
        ("JSR d16(A0)", &[0x4ea8, 4], 0x10fc, 0, 0x1100),
        ("JSR d8(A0,D0.W)", &[0x4eb0, 4], 0x10fc, 0, 0x1100),
        ("JSR absolute.w", &[0x4eb8, 0x1100], 0, 0, 0x1100),
        ("JSR absolute.l", &[0x4eb9, 0, 0x1100], 0, 0, 0x1100),
        ("JSR d16(PC)", &[0x4eba, 0x00fe], 0, 0, 0x1100),
        ("JSR d8(PC,D0.W)", &[0x4ebb, 0x007e], 0, 0x80, 0x1100),
        ("BSR.b", &[0x617e], 0, 0, 0x1080),
        ("BSR.w", &[0x6100, 0x00fe], 0, 0, 0x1100),
    ];

    if CPU >= M68020 {
        cases.push(("BSR.l", &[0x61ff, 0, 0x00fe], 0, 0, 0x1100));
    }

    for (name, words, a0, d0, destination) in cases {
        let mut cpu: CpuM68k<Testbus<Address, u8>, MASK, CPU, FPU, PMMU> =
            CpuM68k::new(Testbus::new(MASK));
        let next = 0x1000 + words.len() as u32 * 2;

        for (index, byte) in words.iter().flat_map(|word| word.to_be_bytes()).enumerate() {
            cpu.bus.mem.insert(0x1000 + index as u32, byte);
        }

        cpu.bus.mem.insert(destination, 0x4e);
        cpu.bus.mem.insert(destination + 1, 0x75); // RTS
        cpu.regs.isp = 0x2000;
        cpu.regs.a[0] = a0;
        cpu.regs.d[0] = d0;
        cpu.regs.sr.set_supervisor(true);
        cpu.regs.sr.set_int_prio_mask(7);

        cpu.set_pc(0x1000).unwrap();
        cpu.prefetch_refill().unwrap();
        cpu.step().unwrap();

        let stacked = u32::from_be_bytes(std::array::from_fn(|offset| {
            cpu.bus.mem[&(0x1ffc + offset as u32)]
        }));

        assert_eq!(cpu.regs.pc, destination, "{CPU}: {name} target");
        assert_eq!(stacked, next, "{CPU}: {name} stacked return");
        assert_eq!(
            cpu.get_step_over(),
            Some(next),
            "{CPU}: {name} debugger target"
        );

        cpu.set_breakpoint(Breakpoint::Execution(next));
        cpu.set_breakpoint(Breakpoint::StepOver(next));
        cpu.step().unwrap();

        let (hits, dropped) = cpu.take_breakpoint_hits();

        assert_eq!(cpu.regs.pc, next, "{CPU}: {name} RTS");
        assert_eq!(cpu.regs.isp, 0x2000);
        assert_eq!(dropped, 0);
        assert!(
            hits.iter()
                .any(|hit| hit.trigger == Breakpoint::StepOver(next))
        );
        assert!(
            hits.iter()
                .any(|hit| hit.trigger == Breakpoint::Execution(next))
        );
        assert!(!cpu.breakpoints().contains(&Breakpoint::StepOver(next)));
        assert!(cpu.breakpoints().contains(&Breakpoint::Execution(next)));
    }
}

#[test]
fn step_over_68000_uses_the_stacked_return_for_every_call_form() {
    calls::<M68000_ADDRESS_MASK, M68000, FPU_NONE, false>();
}

#[test]
fn step_over_68020_uses_the_stacked_return_for_every_call_form() {
    calls::<M68020_ADDRESS_MASK, M68020, FPU_M68881, false>();
}

#[test]
fn step_over_68020_pmmu_uses_the_stacked_return_for_every_call_form() {
    calls::<M68020_ADDRESS_MASK, M68020, FPU_M68881, true>();
}

#[test]
fn step_over_68030_uses_the_stacked_return_for_every_call_form() {
    calls::<M68030_ADDRESS_MASK, M68030, FPU_M68882, true>();
}
