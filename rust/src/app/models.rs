//! Networks loaded once and shared: by every video of every batch, by every camera of the live
//! preview, and between jobs while the profile stays the same. The networks keep no state between
//! frames or videos (trackers and temporal windows are per recording), so sharing them changes
//! nothing in the results; it saves the load time and the memory of duplicate copies.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Result;

use super::library::PipelineFiles;
use crate::cli::predict::{checkpoint_path, Pipeline};
use crate::data::config::{Config, Value};

/// The most recently used pipeline, keyed by everything that shapes it.
#[derive(Default)]
pub(crate) struct ModelCache {
    slot: Mutex<Option<(String, Arc<Pipeline>)>>,
    loads: AtomicUsize,
}

impl ModelCache {
    /// The pipeline for `files`, loading it only when the resolved config or a checkpoint file
    /// changed since the last call. Concurrent callers wait for one load.
    pub(crate) fn get(&self, files: &PipelineFiles) -> Result<Arc<Pipeline>> {
        let config = Config::load(&files.config, &files.overrides)?;
        let key = key(&config);
        let mut slot = self.slot.lock().unwrap();
        if let Some((cached, pipeline)) = slot.as_ref() {
            if *cached == key {
                // Loading sets the device; a reused pipeline must set it again.
                crate::tensor::set_gpu(config.str_at("device", "auto") != "cpu");
                return Ok(pipeline.clone());
            }
        }
        // Free the old networks before loading new ones.
        *slot = None;
        let pipeline = Arc::new(Pipeline::from_config(&config)?);
        self.loads.fetch_add(1, Ordering::Relaxed);
        *slot = Some((key, pipeline.clone()));
        Ok(pipeline)
    }

    /// Loads `files` in the background, so the next preview or batch starts at once.
    pub(crate) fn prewarm(self: &Arc<Self>, files: PipelineFiles) {
        let cache = self.clone();
        std::thread::spawn(move || {
            let _ = cache.get(&files);
        });
    }

    /// How many times networks were loaded.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn loads(&self) -> usize {
        self.loads.load(Ordering::Relaxed)
    }
}

/// The resolved config, plus the size and modification time of each checkpoint it names.
fn key(config: &Config) -> String {
    let mut key = config.json().to_string();
    if let Some(Value::Map(checkpoints)) = config.get("checkpoints") {
        for reference in checkpoints.values().filter_map(Value::as_str) {
            if let Ok(meta) = std::fs::metadata(checkpoint_path(config, reference)) {
                key += &format!("|{reference}:{}:{:?}", meta.len(), meta.modified().ok());
            }
        }
    }
    key
}
