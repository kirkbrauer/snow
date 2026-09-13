/// Represents the data type of a watchpoint
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatchpointType {
    U8,
    U16,
    U32,
    String(usize), // String with length
}

impl WatchpointType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::String(_) => "string",
        }
    }

    pub fn size_bytes(&self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 => 2,
            Self::U32 => 4,
            Self::String(len) => *len,
        }
    }

    /// Try to parse a watchpoint type from a string
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "u8" => Some(Self::U8),
            "u16" => Some(Self::U16),
            "u32" => Some(Self::U32),
            s if s.starts_with("string(") && s.ends_with(')') => {
                let len_str = &s[7..s.len() - 1];
                if let Ok(len) = len_str.parse::<usize>() {
                    if len > 0 && len <= 1024 {
                        // Reasonable bounds
                        Some(Self::String(len))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

impl std::fmt::Display for WatchpointType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::String(len) => write!(f, "string({})", len),
            _ => write!(f, "{}", self.as_str()),
        }
    }
}
