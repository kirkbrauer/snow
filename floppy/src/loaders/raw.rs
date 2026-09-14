//! Raw, sector-based image format

use super::FloppyImageLoader;
use crate::FloppyType;
#[cfg(feature = "fluxfox")]
use crate::loaders::fluxfox::Fluxfox;
use crate::{FloppyImage, macformat::MacFormatEncoder};

use anyhow::{Result, bail};
use strum::IntoEnumIterator;

/// Raw image loader
pub struct RawImage {}

impl FloppyImageLoader for RawImage {
    fn load_with_noise(
        data: &[u8],
        filename: Option<&str>,
        noise: crate::noise::Noise,
    ) -> Result<FloppyImage> {
        let Some(floppytype) = FloppyType::iter().find(|t| t.get_logical_size() == data.len())
        else {
            bail!("Invalid raw image length: {}", data.len())
        };

        if floppytype == FloppyType::Mfm144M {
            #[cfg(feature = "fluxfox")]
            {
                // Hand-off to Fluxfox
                Fluxfox::load_with_noise(data, filename, noise)
            }
            #[cfg(not(feature = "fluxfox"))]
            {
                bail!("Requires fluxfox feature");
            }
        } else {
            MacFormatEncoder::encode_with_noise(
                floppytype,
                data,
                None,
                filename.unwrap_or_default(),
                noise,
            )
        }
    }
}
