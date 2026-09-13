use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

use anyhow::Result;

use super::{DisplayBuffer, Renderer};

/// A renderer that feeds it display buffer back over a channel.
pub struct ChannelRenderer {
    displaybuffer: DisplayBuffer,
    sequence: Arc<AtomicU64>,
    channel: Arc<Mutex<Option<DisplayBuffer>>>,
}

impl ChannelRenderer {
    pub fn frame_sequence(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.sequence)
    }

    pub fn get_receiver(&self) -> Arc<Mutex<Option<DisplayBuffer>>> {
        self.channel.clone()
    }
}

impl Renderer for ChannelRenderer {
    /// Creates a new renderer with a screen of the given size
    fn new(width: u16, height: u16) -> Result<Self> {
        Ok(Self {
            displaybuffer: DisplayBuffer::new(width, height),
            sequence: Arc::new(AtomicU64::new(0)),
            channel: Default::default(),
        })
    }

    fn buffer_mut(&mut self) -> &mut DisplayBuffer {
        &mut self.displaybuffer
    }

    /// Renders changes to screen
    fn update(&mut self) -> Result<()> {
        let new_buffer = self.displaybuffer.new_from_this();
        let buffer = std::mem::replace(&mut self.displaybuffer, new_buffer);
        *self.channel.lock().unwrap() = Some(buffer);
        self.sequence.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}
