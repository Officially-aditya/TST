use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeType {
    Project,
    Directory,
    File,
    Class,
    Function,
    Symbol,
    Module,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeNode {
    pub node_id: u64,
    pub node_type: NodeType,
    pub name: String,
    pub parent: Option<u64>,
    pub children: Vec<u64>,
    pub dependencies: Vec<u64>,
}

pub struct TreeMemory {
    pub nodes: HashMap<u64, TreeNode>,
    next_id: u64,
}

pub enum TreeEvent {
    FileAdded { parent_id: u64, name: String },
    FileRemoved { node_id: u64 },
    FunctionRenamed { node_id: u64, new_name: String },
    DependencyChanged { source_id: u64, target_id: u64, added: bool },
}

impl TreeMemory {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            next_id: 1,
        }
    }

    pub fn insert_node(&mut self, node_type: NodeType, name: String, parent: Option<u64>) -> u64 {
        let node_id = self.next_id;
        self.next_id += 1;
        
        let node = TreeNode {
            node_id,
            node_type,
            name,
            parent,
            children: Vec::new(),
            dependencies: Vec::new(),
        };

        self.nodes.insert(node_id, node);

        if let Some(pid) = parent {
            if let Some(pnode) = self.nodes.get_mut(&pid) {
                pnode.children.push(node_id);
            }
        }

        node_id
    }

    pub fn process_event(&mut self, event: TreeEvent) -> Option<u64> {
        match event {
            TreeEvent::FileAdded { parent_id, name } => {
                Some(self.insert_node(NodeType::File, name, Some(parent_id)))
            }
            TreeEvent::FileRemoved { node_id } => {
                self.remove_node(node_id);
                None
            }
            TreeEvent::FunctionRenamed { node_id, new_name } => {
                if let Some(node) = self.nodes.get_mut(&node_id) {
                    node.name = new_name;
                }
                None
            }
            TreeEvent::DependencyChanged { source_id, target_id, added } => {
                if let Some(node) = self.nodes.get_mut(&source_id) {
                    if added {
                        if !node.dependencies.contains(&target_id) {
                            node.dependencies.push(target_id);
                        }
                    } else {
                        node.dependencies.retain(|&id| id != target_id);
                    }
                }
                None
            }
        }
    }

    pub fn remove_node(&mut self, node_id: u64) {
        if let Some(node) = self.nodes.remove(&node_id) {
            if let Some(pid) = node.parent {
                if let Some(pnode) = self.nodes.get_mut(&pid) {
                    pnode.children.retain(|&id| id != node_id);
                }
            }
            // Map the child IDs to a local vector so we don't have borrow overlap issues
            let children_to_remove = node.children.clone();
            for child_id in children_to_remove {
                self.remove_node(child_id);
            }
        }
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.next_id = 1;
    }

    pub fn query_subgraph(&self, start_id: u64, depth: u32) -> Vec<TreeNode> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = vec![(start_id, 0)];

        while let Some((curr_id, curr_depth)) = queue.pop() {
            if visited.insert(curr_id) {
                if let Some(node) = self.nodes.get(&curr_id) {
                    result.push(node.clone());
                    
                    if curr_depth < depth {
                        for &child_id in &node.children {
                            queue.push((child_id, curr_depth + 1));
                        }
                        for &dep_id in &node.dependencies {
                            queue.push((dep_id, curr_depth + 1));
                        }
                    }
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_crud() {
        let mut mem = TreeMemory::new();
        let p_id = mem.insert_node(NodeType::Project, "MyProj".to_string(), None);
        let f_id = mem.insert_node(NodeType::File, "main.rs".to_string(), Some(p_id));

        assert_eq!(mem.nodes.len(), 2);
        assert_eq!(mem.nodes.get(&p_id).unwrap().children.len(), 1);

        mem.process_event(TreeEvent::FunctionRenamed {
            node_id: f_id,
            new_name: "lib.rs".to_string(),
        });
        assert_eq!(mem.nodes.get(&f_id).unwrap().name, "lib.rs");

        mem.remove_node(p_id);
        assert_eq!(mem.nodes.len(), 0); // Cascading delete
    }

    #[test]
    fn test_subgraph_query() {
        let mut mem = TreeMemory::new();
        let p = mem.insert_node(NodeType::Project, "P".to_string(), None);
        let f1 = mem.insert_node(NodeType::File, "F1".to_string(), Some(p));
        let f2 = mem.insert_node(NodeType::File, "F2".to_string(), Some(p));
        mem.insert_node(NodeType::Function, "Fn1".to_string(), Some(f1));

        let subgraph = mem.query_subgraph(p, 1);
        // p, f1, f2 (depth 1 stops before Fn1)
        assert_eq!(subgraph.len(), 3);
        
        let full_graph = mem.query_subgraph(p, 2);
        assert_eq!(full_graph.len(), 4);
    }
}
