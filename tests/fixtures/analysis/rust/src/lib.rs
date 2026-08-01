mod worker;

use crate::worker::{run as execute, Worker};

trait Runnable {
    fn run(&self) -> usize;
}

struct Service {
    worker: Worker,
}

impl Runnable for Service {
    fn run(&self) -> usize {
        execute() + self.worker.value()
    }
}
