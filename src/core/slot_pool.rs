/// Keeps each use in a batch in its own reusable slot.
pub(crate) struct SlotPool<T> {
    slots: Vec<T>,
    used: usize,
}

impl<T> Default for SlotPool<T> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            used: 0,
        }
    }
}

impl<T> SlotPool<T> {
    pub(crate) fn acquire(&mut self, create: impl FnOnce() -> T) -> &mut T {
        if self.used == self.slots.len() {
            self.slots.push(create());
        }
        let slot = &mut self.slots[self.used];
        self.used += 1;
        slot
    }

    pub(crate) fn acquire_indexed(&mut self, create: impl FnOnce() -> T) -> (usize, &mut T) {
        let index = self.used;
        (index, self.acquire(create))
    }

    pub(crate) fn slot(&self, index: usize) -> &T {
        &self.slots[index]
    }

    pub(crate) fn reset(&mut self) {
        self.used = 0;
    }

    pub(crate) fn truncate_unused(&mut self) {
        self.slots.truncate(self.used);
    }

    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.slots.iter_mut()
    }
}
