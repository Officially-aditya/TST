use tst_memory::kernel::Kernel;
use tst_memory::api::{ApiServer, WriteRequest};
use tst_memory::payload::{Payload, PayloadHeader, PayloadData};
use tst_memory::types::Timestamp;
use std::mem::size_of;
use tst_memory::tst::Node;

/// Dummy worker SLM mock implementation that just reads/writes via API.
struct MockWorkerSLM;

impl MockWorkerSLM {
    fn process_request(&self, input: &str, api: &mut ApiServer) -> String {
        // Read memory
        let read_req = format!(r#"{{"keys":["{}"], "max_results":1}}"#, input);
        let _read_res = api.handle_read(&read_req).unwrap();
        
        // Write memory based on observation
        let payload = Payload {
            header: PayloadHeader {
                payload_type: 1,
                version: 1,
                created_ts: Timestamp(0),
                last_access_ts: Timestamp(0),
                access_count: 1,
            },
            data: PayloadData::TokenStats {
                canonical_form: input.to_string(),
                frequency: 1,
                decay_score: 1.0,
                preferred_tokenizer_origin: None,
            }
        };
        let write_req = WriteRequest {
            op: "insert".to_string(),
            key: input.to_string(),
            layer: None,
            payload,
        };
        let write_json = serde_json::to_string(&write_req).unwrap();
        api.handle_write(&write_json).unwrap();
        
        format!("Processed {}", input)
    }
}

#[test]
fn test_end_to_end_flow() {
    let mut kernel = Kernel::new();
    let mut api = ApiServer::new(&mut kernel);
    let worker = MockWorkerSLM;
    
    let res = worker.process_request("hello", &mut api);
    assert_eq!(res, "Processed hello");

    // Verify it was stored
    let read_req = r#"{"keys":["hello"], "max_results":1}"#;
    let read_res = api.handle_read(read_req).unwrap();
    assert!(read_res.contains("hello"));
}

#[test]
fn test_memory_budget() {
    // Simulating 200k symbols
    let num_symbols = 200_000;
    let node_size = size_of::<Node>();
    assert_eq!(node_size, 24);
    
    // Each symbol corresponds to a node in TST at worst case without prefix compression 
    let node_arena_mb = (num_symbols * node_size) as f64 / 1_048_576.0;
    
    // Specification states 15-23 MB total size.
    assert!(node_arena_mb < 23.0);
    println!("Computed Node Arena Size for 200k nodes: {:.2} MB", node_arena_mb);
}
