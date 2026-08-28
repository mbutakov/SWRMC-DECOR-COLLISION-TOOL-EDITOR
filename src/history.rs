//! Small, bounded undo stack for collision edits.

pub struct UndoHistory<T> {
    entries: Vec<T>,
    capacity: usize,
}

impl<T> UndoHistory<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::new(),
            capacity,
        }
    }
    pub fn push(&mut self, state: T) {
        if self.entries.len() == self.capacity {
            self.entries.remove(0);
        }
        self.entries.push(state);
    }
    pub fn pop(&mut self) -> Option<T> {
        self.entries.pop()
    }
}
