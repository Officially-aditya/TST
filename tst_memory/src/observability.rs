use crate::types::Timestamp;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct MutationLog {
    pub timestamp: Timestamp,
    pub worker_id: String,
    pub payload_id: u32,
    pub old_value: String,
    pub new_value: String,
}

pub struct Metrics {
    pub total_latencies_ms: AtomicU64,
    pub request_count: AtomicU64,
    pub schema_failures: AtomicU64,
    pub hallucinations_detected: AtomicU64,
    pub escalations: AtomicU64,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            total_latencies_ms: AtomicU64::new(0),
            request_count: AtomicU64::new(0),
            schema_failures: AtomicU64::new(0),
            hallucinations_detected: AtomicU64::new(0),
            escalations: AtomicU64::new(0),
        }
    }

    pub fn record_latency(&self, ms: u64) {
        self.total_latencies_ms.fetch_add(ms, Ordering::Relaxed);
        self.request_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn avg_latency_ms(&self) -> f64 {
        let count = self.request_count.load(Ordering::Relaxed);
        if count == 0 {
            0.0
        } else {
            self.total_latencies_ms.load(Ordering::Relaxed) as f64 / count as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics() {
        let metrics = Metrics::new();
        metrics.record_latency(10);
        metrics.record_latency(20);
        
        assert_eq!(metrics.avg_latency_ms(), 15.0);
        assert_eq!(metrics.request_count.load(Ordering::Relaxed), 2);
    }
}
