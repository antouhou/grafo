use crate::commands::EffectParameters;
use ahash::RandomState;
use std::mem;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct EffectParameterRange {
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct EffectParametersHandle {
    pub range: EffectParameterRange,
    pub hash: u64,
}

#[derive(Default)]
pub(crate) struct EffectParameterStorage {
    pub(crate) bytes: Vec<u8>,
    compacted: Vec<u8>,
    hasher: RandomState,
    needs_compaction: bool,
}

impl EffectParameterStorage {
    pub(crate) fn store(&mut self, parameters: &[u8]) -> EffectParametersHandle {
        let start = self.bytes.len();
        self.bytes.extend_from_slice(parameters);
        EffectParametersHandle {
            range: EffectParameterRange {
                start,
                end: self.bytes.len(),
            },
            hash: self.hasher.hash_one(parameters),
        }
    }

    pub(crate) fn get(&self, parameters: EffectParametersHandle) -> EffectParameters<'_> {
        EffectParameters {
            bytes: &self.bytes[parameters.range.start..parameters.range.end],
            hash: parameters.hash,
        }
    }

    pub(crate) fn request_compaction(&mut self) {
        self.needs_compaction = true;
    }

    pub(crate) fn compact<'attachments>(
        &mut self,
        parameters: impl Iterator<Item = &'attachments mut EffectParametersHandle>,
    ) {
        if !self.needs_compaction {
            return;
        }
        self.compacted.clear();
        for parameters in parameters {
            let start = self.compacted.len();
            self.compacted
                .extend_from_slice(&self.bytes[parameters.range.start..parameters.range.end]);
            parameters.range.start = start;
            parameters.range.end = self.compacted.len();
        }
        mem::swap(&mut self.bytes, &mut self.compacted);
        self.compacted.clear();
        self.needs_compaction = false;
    }

    pub(crate) fn clear(&mut self) {
        self.bytes.clear();
        self.compacted.clear();
        self.needs_compaction = false;
    }
}
