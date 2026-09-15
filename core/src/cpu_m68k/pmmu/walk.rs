//! Shared bounded descriptor walk. The reader chooses execution or inspection effects.

use num_traits::FromPrimitive;
use serde::{Deserialize, Serialize};

use super::regs::{PmmuPageDescriptorType as Dt, RootPointerReg, TcReg};

/// Why a descriptor search cannot produce a translation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WalkFailureKind {
    InvalidDescriptor,
    LimitViolation,
    InvalidConfiguration,
    UnsupportedRoot,
    DepthExceeded,
    UnreadableDescriptor,
    DescriptorUpdate,
    WriteProtected,
    SupervisorOnly,
}

/// A failure belongs to this attempted search, not to the live status register.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WalkFailure {
    pub kind: WalkFailureKind,
    pub level: u8,
    pub address: Option<u32>,
    pub detail: String,
    /// Long words read before an unreadable boundary in the failing descriptor.
    pub partial_words: Vec<u32>,
    /// Bytes read within the incomplete long word at the failing boundary.
    pub partial_bytes: Vec<u8>,
}

impl WalkFailure {
    pub(super) fn new(
        kind: WalkFailureKind,
        address: Option<u32>,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            level: 0,
            address,
            detail: detail.into(),
            partial_words: Vec::new(),
            partial_bytes: Vec::new(),
        }
    }
}

/// Which address-space or logical-address field selects a descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DescriptorIndex {
    FunctionCode,
    TableA,
    TableB,
    TableC,
    TableD,
}

/// Evidence for one descriptor, before any execution-side U/M updates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DescriptorStep {
    pub level: u8,
    pub index_source: DescriptorIndex,
    pub index: u32,
    pub address: u32,
    pub width: u8,
    pub words: Vec<u32>,
    pub descriptor_type: u8,
    pub target_address: Option<u32>,
    pub write_protected: bool,
    pub supervisor_only: bool,
    pub used: bool,
    pub modified: bool,
    pub cache_inhibit: bool,
    pub parent_limit: Option<IndexLimit>,
    pub child_limit: Option<IndexLimit>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexLimit {
    pub value: u16,
    pub lower: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedPage {
    pub physical_address: u32,
    pub offset_bits: u8,
    pub write_protected: bool,
    pub supervisor_only: bool,
    pub leaf_address: u32,
    pub modified: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TableWalk {
    pub descriptors: Vec<DescriptorStep>,
    pub level: u8,
    pub resolved: Option<ResolvedPage>,
    pub failure: Option<WalkFailure>,
}

/// Implementations must not choose a different descriptor interpretation.
pub(super) trait WalkMemory {
    fn select_descriptor(&mut self, address: u32);
    fn read_long(&mut self, address: u32) -> Result<u32, WalkFailure>;
    fn mark_used(&mut self, address: u32, original: u32) -> Result<(), WalkFailure>;
}

impl TableWalk {
    fn fail(&mut self, kind: WalkFailureKind, address: Option<u32>, detail: &str) {
        self.failure = Some(WalkFailure {
            kind,
            level: self.level,
            address,
            detail: detail.into(),
            partial_words: Vec::new(),
            partial_bytes: Vec::new(),
        });
    }
}

/// Walk the optional F level followed by A/B/C/D, using the same effects policy.
pub(super) fn walk_tables(
    memory: &mut impl WalkMemory,
    tc: TcReg,
    root: RootPointerReg,
    logical: u32,
    function_code: u8,
) -> TableWalk {
    let mut evidence = TableWalk {
        descriptors: Vec::with_capacity(5),
        level: 0,
        resolved: None,
        failure: None,
    };

    let indices = [tc.tia(), tc.tib(), tc.tic(), tc.tid()];
    let geometry =
        tc.is() + u32::from(tc.ps()) + indices.iter().map(|&bits| u32::from(bits)).sum::<u32>();

    if geometry != 32 {
        evidence.fail(
            WalkFailureKind::InvalidConfiguration,
            None,
            "TC initial shift, table indices and page offset must total 32 bits",
        );
        return evidence;
    }

    let mut descriptor_type = Dt::from_u8(root.dt()).expect("two-bit descriptor type");
    let mut table_address = root.table_addr() << 4;
    // MC68030 UM 9.7.2 and MC68851 UM 6.1.3.3: FCL suppresses the root limit.
    let mut limit = (!tc.fcl()).then_some(IndexLimit {
        value: root.limit(),
        lower: root.lu(),
    });
    let mut consumed = tc.is();
    let mut shifted = logical << tc.is();
    let mut write_protected = false;
    let mut supervisor_only = false;

    if !matches!(descriptor_type, Dt::Valid4b | Dt::Valid8b) {
        evidence.fail(
            WalkFailureKind::UnsupportedRoot,
            None,
            "Snow's walker requires a short or long table root (DT=2 or DT=3)",
        );
        return evidence;
    }

    let fields = [
        (DescriptorIndex::FunctionCode, 0),
        (DescriptorIndex::TableA, tc.tia()),
        (DescriptorIndex::TableB, tc.tib()),
        (DescriptorIndex::TableC, tc.tic()),
        (DescriptorIndex::TableD, tc.tid()),
    ];

    for (level, (index_source, bits)) in fields
        .into_iter()
        .filter(|(source, _)| tc.fcl() || *source != DescriptorIndex::FunctionCode)
        .enumerate()
    {
        evidence.level = (level + 1) as u8;

        if bits == 0 && index_source != DescriptorIndex::FunctionCode {
            evidence.fail(WalkFailureKind::DepthExceeded, Some(table_address),
                "A table descriptor requires another nonzero index; indirect descriptors are unsupported");
            return evidence;
        }

        consumed += u32::from(bits);
        let index = if index_source == DescriptorIndex::FunctionCode {
            u32::from(function_code & 7)
        } else {
            shifted >> (32 - bits)
        };

        if let Some(bound) = limit {
            let invalid = if bound.lower {
                index < u32::from(bound.value)
            } else {
                index > u32::from(bound.value)
            };

            if invalid {
                evidence.fail(
                    WalkFailureKind::LimitViolation,
                    Some(table_address),
                    "Descriptor index violates its parent limit",
                );
                return evidence;
            }
        }

        let width = if descriptor_type == Dt::Valid4b { 4 } else { 8 };
        let address = table_address.wrapping_add(index * u32::from(width));
        memory.select_descriptor(address);
        let mut words = Vec::with_capacity(2);

        for offset in (0..u32::from(width)).step_by(4) {
            match memory.read_long(address.wrapping_add(offset)) {
                Ok(word) => words.push(word),
                Err(mut error) => {
                    error.level = evidence.level;
                    error.partial_words = words;
                    evidence.failure = Some(error);
                    return evidence;
                }
            }
        }

        let flags = words[0];
        let target = words[words.len() - 1];
        descriptor_type = Dt::from_u32(flags & 3).expect("two-bit descriptor type");
        let page = descriptor_type == Dt::PageDescriptor;
        let step_wp = flags & 4 != 0;
        let step_s = width == 8 && flags & 0x100 != 0;
        let child_limit = if width == 8 && !page && descriptor_type != Dt::Invalid {
            Some(IndexLimit {
                value: ((flags >> 16) & 0x7FFF) as u16,
                lower: flags >> 31 != 0,
            })
        } else {
            None
        };
        let target_address =
            (descriptor_type != Dt::Invalid).then_some(target & if page { !255 } else { !15 });

        evidence.descriptors.push(DescriptorStep {
            level: evidence.level,
            index_source,
            index,
            address,
            width,
            words,
            descriptor_type: descriptor_type as u8,
            target_address,
            write_protected: step_wp,
            supervisor_only: step_s,
            used: flags & 8 != 0,
            modified: page && flags & 16 != 0,
            cache_inhibit: page && flags & 64 != 0,
            parent_limit: limit,
            child_limit,
        });

        if descriptor_type == Dt::Invalid {
            evidence.fail(
                WalkFailureKind::InvalidDescriptor,
                Some(address),
                "Descriptor DT is invalid",
            );
            return evidence;
        }

        if flags & 8 == 0
            && let Err(mut error) = memory.mark_used(address, flags)
        {
            error.level = evidence.level;
            evidence.failure = Some(error);
            return evidence;
        }

        write_protected |= step_wp;
        supervisor_only |= step_s;

        if page {
            let mask = u32::MAX.unbounded_shr(consumed);
            evidence.resolved = Some(ResolvedPage {
                physical_address: (target & !255 & !mask) | (logical & mask),
                offset_bits: (32 - consumed) as u8,
                write_protected,
                supervisor_only,
                leaf_address: address,
                modified: flags & 16 != 0,
            });
            return evidence;
        }

        table_address = target & !15;
        limit = child_limit;
        shifted <<= bits;
    }

    evidence.fail(
        WalkFailureKind::DepthExceeded,
        Some(table_address),
        "Table search exceeds F/A/B/C/D descriptor levels",
    );
    evidence
}

pub(super) fn protection_failure(
    page: ResolvedPage,
    fc: u8,
    writing: bool,
) -> Option<WalkFailureKind> {
    if fc & 4 == 0 && page.supervisor_only {
        Some(WalkFailureKind::SupervisorOnly)
    } else if writing && page.write_protected {
        Some(WalkFailureKind::WriteProtected)
    } else {
        None
    }
}
