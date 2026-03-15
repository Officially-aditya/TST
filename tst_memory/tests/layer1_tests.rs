use tst_memory::stm::{ShortTermMemory, STMEntry};
use tst_memory::tree::{TreeMemory, TreeEvent, NodeType};
use tst_memory::types::Timestamp;

#[test]
fn test_layer1_stm_isolation() {
    let mut stm = ShortTermMemory::new(256, 5.0);
    let start_time = std::time::Instant::now();
    for i in 0..10 {
        stm.insert(STMEntry {
            entry_id: i,
            created_ts: Timestamp(0),
            last_access_ts: Timestamp(0),
            key_ref: i,
            payload_ref: i * 2,
            reinforcement_score: 1.0,
            flags: 0,
        });
    }
    
    // Check sub-1ms lookup
    let _ = stm.lookup_mut(5);
    let lookup_time = start_time.elapsed();
    assert!(lookup_time.as_millis() < 1, "Lookup exceeded 1ms");

    // Decay test
    let initial_score = stm.lookup_mut(5).unwrap().reinforcement_score;
    stm.decay_tick();
    let decayed_score = stm.lookup_mut(5).unwrap().reinforcement_score;
    assert_eq!(decayed_score, initial_score * 0.98);

    // Promotion test
    for _ in 0..10 {
        let entry = stm.lookup_mut(5).unwrap();
        entry.reinforcement_score += 1.0;
    }
    
    let promoted = stm.check_promotion();
    let mut found = false;
    for p in promoted {
         if p.key_ref == 5 { found = true; break; }
    }
    assert!(found, "Entry not flagged for promotion");

    // Overflow test (Ring Buffer eviction over capacity)
    let mut small_stm = ShortTermMemory::new(4, 5.0);
    for i in 1..=5 {
        small_stm.insert(STMEntry {
            entry_id: i,
            created_ts: Timestamp(0),
            last_access_ts: Timestamp(0),
            key_ref: i,
            payload_ref: i * 2,
            reinforcement_score: 1.0,
            flags: 0,
        });
    }
    
    // 5 elements inserted into a 4-slot buffer. 1st should be overwritten.
    assert!(small_stm.lookup_mut(1).is_none(), "Oldest entry was not evicted");
    assert!(small_stm.lookup_mut(5).is_some(), "Newest entry not found");
}

#[test]
fn test_layer1_tree_isolation() {
    let mut tree = TreeMemory::new();
    
    let pid = tree.insert_node(NodeType::Project, "src".to_string(), None);

    // file1.py with function_c
    let f1 = tree.process_event(TreeEvent::FileAdded { parent_id: pid, name: "file1.py".to_string() }).unwrap();
    let c_id = tree.insert_node(NodeType::Function, "function_c".to_string(), Some(f1));

    // file2.py with function_e
    let f2 = tree.process_event(TreeEvent::FileAdded { parent_id: pid, name: "file2.py".to_string() }).unwrap();
    let e_id = tree.insert_node(NodeType::Function, "function_e".to_string(), Some(f2));

    // Add dependency edge function_c -> function_e
    tree.process_event(TreeEvent::DependencyChanged { source_id: c_id, target_id: e_id, added: true });

    // Subgraph query on function_c should now retrieve function_e via dependency link
    let subgraph = tree.query_subgraph(c_id, 1);
    let names: Vec<String> = subgraph.iter().map(|n| n.name.clone()).collect();
    
    assert!(names.contains(&"function_c".to_string()));
    assert!(names.contains(&"function_e".to_string()));
    assert!(!names.contains(&"file2.py".to_string()));
}
