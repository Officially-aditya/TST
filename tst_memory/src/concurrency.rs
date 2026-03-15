use std::sync::{Arc, RwLock};
use crate::kernel::Kernel;

/// MemoryGuard encapsulates the Kernel inside an RwLock to satisfy Phase 9 Concurrency requirements.
/// It provides concurrent reads and serialized writes for the edge environment (2-4 cores).
#[derive(Clone)]
pub struct MemoryGuard {
    inner: Arc<RwLock<Kernel>>,
}

impl MemoryGuard {
    pub fn new(kernel: Kernel) -> Self {
        Self {
            inner: Arc::new(RwLock::new(kernel)),
        }
    }

    pub fn read<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&Kernel) -> R,
    {
        let lock = self.inner.read().expect("RwLock poisoned");
        f(&*lock)
    }

    pub fn write<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut Kernel) -> R,
    {
        let mut lock = self.inner.write().expect("RwLock poisoned");
        f(&mut *lock)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_concurrency_guard() {
        let kernel = Kernel::new();
        let guard = MemoryGuard::new(kernel);
        
        let g1 = guard.clone();
        let t1 = thread::spawn(move || {
            g1.read(|k| {
                let _ = k.route_read(b"test");
            });
        });

        let g2 = guard.clone();
        let t2 = thread::spawn(move || {
            g2.read(|k| {
                let _ = k.route_read(b"test2");
            });
        });

        t1.join().unwrap();
        t2.join().unwrap();
    }
}
