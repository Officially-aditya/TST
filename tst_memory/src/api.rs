use serde::{Deserialize, Serialize};
use crate::kernel::{Kernel, WriteProposal, MemoryLayer};
use crate::payload::Payload;
use crate::tree::{NodeType, TreeEvent};

#[derive(Serialize, Deserialize, Debug)]
pub struct ReadRequest {
    pub keys: Vec<String>,
    pub max_results: u32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ReadResponse {
    pub slices: Vec<Option<Payload>>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct WriteRequest {
    pub op: String,
    pub key: String,
    pub layer: Option<String>,
    pub payload: Payload,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TreeInsertRequest {
    pub node_type: String,
    pub name: String,
    pub parent_id: Option<u64>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TreeQueryRequest {
    pub node_id: u64,
    pub depth: u32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TreeLinkRequest {
    pub source_id: u64,
    pub target_id: u64,
    pub add: bool,
}

fn parse_node_type(s: &str) -> Result<NodeType, String> {
    match s {
        "Project"   => Ok(NodeType::Project),
        "Directory" => Ok(NodeType::Directory),
        "File"      => Ok(NodeType::File),
        "Class"     => Ok(NodeType::Class),
        "Function"  => Ok(NodeType::Function),
        "Symbol"    => Ok(NodeType::Symbol),
        "Module"    => Ok(NodeType::Module),
        other => Err(format!("Unknown node type: {}", other)),
    }
}

pub struct ApiServer<'a> {
    pub kernel: &'a mut Kernel,
}

impl<'a> ApiServer<'a> {
    pub fn new(kernel: &'a mut Kernel) -> Self {
        Self { kernel }
    }

    pub fn handle_read(&mut self, req_json: &str) -> Result<String, String> {
        let req: ReadRequest = serde_json::from_str(req_json)
            .map_err(|e| format!("Invalid ReadRequest JSON: {}", e))?;

        let mut slices = Vec::new();
        for key in &req.keys {
            // First check if STM has the payload natively using Hash cache
            let key_hash = Kernel::hash_key(key.as_bytes());
            if let Some(entry) = self.kernel.stm.lookup_mut(key_hash) {
                // Return payload from the Arena natively via entry.payload_ref
                if let Some(p) = self.kernel.ltm.payloads.get(entry.payload_ref) {
                    slices.push(Some(p.clone()));
                } else {
                    slices.push(None); // Should never happen unless arena is corrupted
                }
            } else {
                let res = self.kernel.route_read(key.as_bytes());
                slices.push(res);
            }
        }

        let resp = ReadResponse { slices };
        serde_json::to_string(&resp).map_err(|e| format!("Failed to serialize ReadResponse: {}", e))
    }

    pub fn handle_write(&mut self, req_json: &str) -> Result<String, String> {
        let req: WriteRequest = serde_json::from_str(req_json)
            .map_err(|e| format!("Invalid WriteRequest JSON: {}", e))?;

        let layer = match req.layer.as_deref() {
            Some("STM") => MemoryLayer::STM,
            Some("Tree") => MemoryLayer::Tree,
            _ => MemoryLayer::LTM, // Default
        };

        let proposal = WriteProposal {
            layer,
            key: req.key.into_bytes(),
            payload: Some(req.payload),
            tree_event: None,
        };

        match self.kernel.validate_and_commit(proposal) {
            Ok(_) => Ok(r#"{"status":"ok"}"#.to_string()),
            Err(e) => Err(format!("Kernel validation failed: {}", e)),
        }
    }

    pub fn handle_tree_insert(&mut self, req_json: &str) -> Result<String, String> {
        let req: TreeInsertRequest = serde_json::from_str(req_json)
            .map_err(|e| format!("Invalid TreeInsertRequest JSON: {}", e))?;
        let nt = parse_node_type(&req.node_type)?;
        let node_id = self.kernel.tree.insert_node(nt, req.name, req.parent_id);
        serde_json::to_string(&serde_json::json!({"node_id": node_id}))
            .map_err(|e| format!("Serialize error: {}", e))
    }

    pub fn handle_tree_query(&self, req_json: &str) -> Result<String, String> {
        let req: TreeQueryRequest = serde_json::from_str(req_json)
            .map_err(|e| format!("Invalid TreeQueryRequest JSON: {}", e))?;
        let nodes = self.kernel.tree.query_subgraph(req.node_id, req.depth);
        serde_json::to_string(&serde_json::json!({"nodes": nodes}))
            .map_err(|e| format!("Serialize error: {}", e))
    }

    pub fn handle_tree_link(&mut self, req_json: &str) -> Result<String, String> {
        let req: TreeLinkRequest = serde_json::from_str(req_json)
            .map_err(|e| format!("Invalid TreeLinkRequest JSON: {}", e))?;
        self.kernel.tree.process_event(TreeEvent::DependencyChanged {
            source_id: req.source_id,
            target_id: req.target_id,
            added: req.add,
        });
        Ok(r#"{"status":"ok"}"#.to_string())
    }

    pub fn handle_tree_clear(&mut self) -> Result<String, String> {
        self.kernel.tree.clear();
        Ok(r#"{"status":"ok"}"#.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_read_write() {
        let mut kernel = Kernel::new();
        let mut api = ApiServer::new(&mut kernel);

        let write_json = r#"{
            "op": "update",
            "key": "user:test",
            "payload": {
                "header": {
                    "payload_type": 1,
                    "version": 1,
                    "created_ts": 100,
                    "last_access_ts": 100,
                    "access_count": 0
                },
                "data": {
                    "Preference": {
                        "key": "color",
                        "value": "blue",
                        "weight": 1.0
                    }
                }
            }
        }"#;

        let write_res = api.handle_write(write_json).unwrap();
        assert_eq!(write_res, r#"{"status":"ok"}"#);

        let read_json = r#"{
            "keys": ["user:test", "not:found"],
            "max_results": 10
        }"#;

        let read_res = api.handle_read(read_json).unwrap();
        assert!(read_res.contains("blue"));
        assert!(read_res.contains("null"));
    }
}
