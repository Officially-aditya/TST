use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{BuildHasherDefault, Hasher};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeType {
    Project,
    Directory,
    File,
    Class,
    Interface,
    Struct,
    Enum,
    Trait,
    Function,
    Symbol,
    Module,
    External,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeType {
    Contains,
    Imports,
    Calls,
    References,
    Defines,
    Inherits,
    Implements,
    Tests,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Edge {
    pub target_id: u64,
    pub edge_type: EdgeType,
    #[serde(default = "default_confidence")]
    pub confidence: f32,
}

fn default_confidence() -> f32 {
    1.0
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TreeNodeMetadata {
    pub qualified_name: Option<String>,
    pub file_path: Option<String>,
    pub start_line: Option<u32>,
    pub end_line: Option<u32>,
    pub signature: Option<String>,
    pub content_hash: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TreeNode {
    pub node_id: u64,
    pub node_type: NodeType,
    pub name: String,
    pub qualified_name: String,
    pub file_path: Option<String>,
    pub start_line: Option<u32>,
    pub end_line: Option<u32>,
    pub signature: Option<String>,
    pub content_hash: Option<String>,
    pub parent: Option<u64>,
    pub children: Vec<u64>,
    pub dependencies: Vec<Edge>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TreeQueryResult {
    pub nodes: Vec<TreeNode>,
    pub truncated: bool,
    pub estimated_tokens: usize,
}

pub struct TreeMemory {
    pub nodes: NodeIdMap<TreeNode>,
    incoming: NodeIdMap<Vec<(u64, EdgeType)>>,
    next_id: u64,
}

pub type NodeIdMap<T> = HashMap<u64, T, BuildHasherDefault<NodeIdHasher>>;

#[derive(Default)]
pub struct NodeIdHasher(u64);

impl Hasher for NodeIdHasher {
    fn finish(&self) -> u64 {
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut value = 0xcbf2_9ce4_8422_2325;
        for byte in bytes {
            value ^= u64::from(*byte);
            value = value.wrapping_mul(0x0000_0100_0000_01b3);
        }
        self.0 = value;
    }

    fn write_u64(&mut self, value: u64) {
        self.0 = value;
    }
}

enum VisitSet {
    Dense(Vec<u8>),
    Sparse(HashSet<u64>),
}

impl VisitSet {
    fn new(next_id: u64, node_count: usize, expected_nodes: usize) -> Self {
        let dense_limit = node_count.saturating_mul(4).saturating_add(1_024);
        if let Ok(length) = usize::try_from(next_id)
            && length <= dense_limit
            && length <= 10_000_000
        {
            return Self::Dense(vec![0; length]);
        }
        Self::Sparse(HashSet::with_capacity(expected_nodes))
    }

    #[inline]
    fn insert(&mut self, node_id: u64) -> bool {
        match self {
            Self::Dense(visited) => {
                let index = node_id as usize;
                if visited[index] != 0 {
                    return false;
                }
                visited[index] = 1;
                true
            }
            Self::Sparse(visited) => visited.insert(node_id),
        }
    }

    #[inline]
    fn contains(&self, node_id: &u64) -> bool {
        match self {
            Self::Dense(visited) => visited[*node_id as usize] != 0,
            Self::Sparse(visited) => visited.contains(node_id),
        }
    }
}

pub enum TreeEvent {
    FileAdded {
        parent_id: u64,
        name: String,
    },
    FileRemoved {
        node_id: u64,
    },
    FunctionRenamed {
        node_id: u64,
        new_name: String,
    },
    DependencyChanged {
        source_id: u64,
        target_id: u64,
        added: bool,
    },
}

impl Default for TreeMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl TreeMemory {
    pub fn new() -> Self {
        Self {
            nodes: NodeIdMap::default(),
            incoming: NodeIdMap::default(),
            next_id: 1,
        }
    }

    pub fn insert_node(&mut self, node_type: NodeType, name: String, parent: Option<u64>) -> u64 {
        self.insert_node_with_metadata(node_type, name, parent, TreeNodeMetadata::default())
    }

    pub fn insert_node_with_metadata(
        &mut self,
        node_type: NodeType,
        name: String,
        parent: Option<u64>,
        metadata: TreeNodeMetadata,
    ) -> u64 {
        let node_id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);

        let parent = parent.filter(|parent_id| self.nodes.contains_key(parent_id));
        let parent_node = parent.and_then(|parent_id| self.nodes.get(&parent_id));
        let qualified_name = metadata.qualified_name.unwrap_or_else(|| {
            parent_node
                .map(|parent| format!("{}::{}", parent.qualified_name, name))
                .unwrap_or_else(|| name.clone())
        });
        let file_path = metadata.file_path.or_else(|| match node_type {
            NodeType::File => Some(name.clone()),
            _ => parent_node.and_then(|parent| parent.file_path.clone()),
        });

        let node = TreeNode {
            node_id,
            node_type,
            name,
            qualified_name,
            file_path,
            start_line: metadata.start_line,
            end_line: metadata.end_line,
            signature: metadata.signature,
            content_hash: metadata.content_hash,
            parent,
            children: Vec::new(),
            dependencies: Vec::new(),
        };

        self.nodes.insert(node_id, node);
        if let Some(parent_id) = parent
            && let Some(parent_node) = self.nodes.get_mut(&parent_id)
        {
            parent_node.children.push(node_id);
        }
        node_id
    }

    pub fn contains_node(&self, node_id: u64) -> bool {
        self.nodes.contains_key(&node_id)
    }

    pub fn link(
        &mut self,
        source_id: u64,
        target_id: u64,
        edge_type: EdgeType,
        confidence: f32,
    ) -> bool {
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return false;
        }
        if !self.nodes.contains_key(&target_id) {
            return false;
        }
        let Some(source) = self.nodes.get_mut(&source_id) else {
            return false;
        };
        let inserted = if let Some(edge) = source
            .dependencies
            .iter_mut()
            .find(|edge| edge.target_id == target_id && edge.edge_type == edge_type)
        {
            edge.confidence = confidence;
            false
        } else {
            source.dependencies.push(Edge {
                target_id,
                edge_type,
                confidence,
            });
            true
        };
        if inserted {
            self.incoming
                .entry(target_id)
                .or_default()
                .push((source_id, edge_type));
        }
        true
    }

    pub fn unlink(&mut self, source_id: u64, target_id: u64, edge_type: Option<EdgeType>) -> bool {
        let Some(source) = self.nodes.get_mut(&source_id) else {
            return false;
        };
        let old_len = source.dependencies.len();
        source.dependencies.retain(|edge| {
            !(edge.target_id == target_id
                && edge_type.is_none_or(|edge_type| edge.edge_type == edge_type))
        });
        let removed = old_len != source.dependencies.len();
        if removed && let Some(incoming) = self.incoming.get_mut(&target_id) {
            incoming.retain(|(candidate_source, candidate_type)| {
                !(*candidate_source == source_id
                    && edge_type.is_none_or(|edge_type| *candidate_type == edge_type))
            });
            if incoming.is_empty() {
                self.incoming.remove(&target_id);
            }
        }
        removed
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
                    node.name = new_name.clone();
                    if let Some((prefix, _)) = node.qualified_name.rsplit_once("::") {
                        node.qualified_name = format!("{}::{}", prefix, new_name);
                    } else {
                        node.qualified_name = new_name;
                    }
                }
                None
            }
            TreeEvent::DependencyChanged {
                source_id,
                target_id,
                added,
            } => {
                if added {
                    self.link(source_id, target_id, EdgeType::References, 1.0);
                } else {
                    self.unlink(source_id, target_id, None);
                }
                None
            }
        }
    }

    /// Removes a node, its descendants, and every incoming edge/reference.
    pub fn remove_node(&mut self, node_id: u64) -> bool {
        if !self.nodes.contains_key(&node_id) {
            return false;
        }
        let mut to_remove = HashSet::new();
        let mut stack = vec![node_id];
        while let Some(current) = stack.pop() {
            if to_remove.insert(current)
                && let Some(node) = self.nodes.get(&current)
            {
                stack.extend(node.children.iter().copied());
            }
        }

        for id in &to_remove {
            self.nodes.remove(id);
        }
        for node in self.nodes.values_mut() {
            node.children.retain(|id| !to_remove.contains(id));
            node.dependencies
                .retain(|edge| !to_remove.contains(&edge.target_id));
        }
        self.incoming.retain(|target_id, sources| {
            if to_remove.contains(target_id) {
                return false;
            }
            sources.retain(|(source_id, _)| !to_remove.contains(source_id));
            !sources.is_empty()
        });
        true
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.incoming.clear();
        self.next_id = 1;
    }

    pub fn edge_count(&self) -> usize {
        self.nodes
            .values()
            .map(|node| node.children.len() + node.dependencies.len())
            .sum()
    }

    pub fn find(&self, name: &str, limit: usize) -> Vec<TreeNode> {
        let needle = name.to_lowercase();
        let mut matches: Vec<TreeNode> = self
            .nodes
            .values()
            .filter(|node| {
                node.name.to_lowercase().contains(&needle)
                    || node.qualified_name.to_lowercase().contains(&needle)
                    || node
                        .file_path
                        .as_ref()
                        .is_some_and(|path| path.to_lowercase().contains(&needle))
            })
            .cloned()
            .collect();
        matches.sort_by(|left, right| {
            let left_exact = left.name.eq_ignore_ascii_case(name);
            let right_exact = right.name.eq_ignore_ascii_case(name);
            right_exact
                .cmp(&left_exact)
                .then_with(|| left.qualified_name.cmp(&right.qualified_name))
                .then_with(|| left.node_id.cmp(&right.node_id))
        });
        matches.truncate(limit.clamp(1, 1_000));
        matches
    }

    pub fn query_subgraph(&self, start_id: u64, depth: u32) -> Vec<TreeNode> {
        if self.incoming.is_empty() {
            return self.query_hierarchy(start_id, depth);
        }
        self.query_subgraph_bounded(start_id, depth, usize::MAX, usize::MAX)
            .nodes
    }

    fn query_hierarchy(&self, start_id: u64, depth: u32) -> Vec<TreeNode> {
        let mut result = Vec::with_capacity(self.nodes.len());
        let mut stack = vec![(start_id, 0, None)];
        while let Some((current_id, current_depth, previous_id)) = stack.pop() {
            let Some(node) = self.nodes.get(&current_id) else {
                continue;
            };
            result.push(node.clone());
            if current_depth >= depth {
                continue;
            }
            for &child_id in node.children.iter().rev() {
                if Some(child_id) != previous_id {
                    stack.push((child_id, current_depth + 1, Some(current_id)));
                }
            }
            if let Some(parent) = node.parent
                && Some(parent) != previous_id
            {
                stack.push((parent, current_depth + 1, Some(current_id)));
            }
        }
        result
    }

    pub fn query_subgraph_bounded(
        &self,
        start_id: u64,
        depth: u32,
        max_nodes: usize,
        token_budget: usize,
    ) -> TreeQueryResult {
        let expected_nodes = max_nodes.min(self.nodes.len()).max(1);
        let mut result = Vec::with_capacity(expected_nodes);
        let has_dependency_edges = !self.incoming.is_empty();
        let mut visited = has_dependency_edges
            .then(|| VisitSet::new(self.next_id, self.nodes.len(), expected_nodes));
        let mut queue = VecDeque::with_capacity(expected_nodes);
        queue.push_back((start_id, 0, None));
        let mut estimated_tokens: usize = 0;
        let mut truncated = false;
        let enforce_node_budget = max_nodes != usize::MAX;
        let enforce_token_budget = token_budget != usize::MAX;

        while let Some((current_id, current_depth, previous_id)) = queue.pop_front() {
            if visited
                .as_mut()
                .is_none_or(|visited| visited.insert(current_id))
                && let Some(node) = self.nodes.get(&current_id)
            {
                let cost = if enforce_token_budget {
                    node_cost(node)
                } else {
                    0
                };
                if !result.is_empty()
                    && ((enforce_node_budget && result.len() >= max_nodes)
                        || (enforce_token_budget
                            && estimated_tokens.saturating_add(cost) > token_budget))
                {
                    truncated = true;
                    continue;
                }
                result.push(node.clone());
                estimated_tokens = estimated_tokens.saturating_add(cost);
                if current_depth < depth {
                    if !has_dependency_edges {
                        if let Some(parent) = node.parent
                            && Some(parent) != previous_id
                        {
                            queue.push_back((parent, current_depth + 1, Some(current_id)));
                        }
                        for &child_id in &node.children {
                            if Some(child_id) != previous_id {
                                queue.push_back((child_id, current_depth + 1, Some(current_id)));
                            }
                        }
                        continue;
                    }
                    let mut neighbors = Vec::new();
                    if let Some(parent) = node.parent {
                        neighbors.push((0_u8, parent));
                    }
                    if let Some(incoming) = self.incoming.get(&current_id) {
                        for &(source_id, edge_type) in incoming {
                            neighbors.push((edge_priority(edge_type), source_id));
                        }
                    }
                    for edge in &node.dependencies {
                        neighbors.push((edge_priority(edge.edge_type), edge.target_id));
                    }
                    for &child_id in &node.children {
                        neighbors.push((9, child_id));
                    }
                    neighbors.sort_unstable_by_key(|(priority, node_id)| (*priority, *node_id));
                    for (_, neighbor) in neighbors {
                        if !visited
                            .as_ref()
                            .is_some_and(|visited| visited.contains(&neighbor))
                        {
                            queue.push_back((neighbor, current_depth + 1, Some(current_id)));
                        }
                    }
                }
            }
        }
        TreeQueryResult {
            nodes: result,
            truncated,
            estimated_tokens,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        for node in self.nodes.values() {
            if let Some(parent) = node.parent {
                let parent_node = self.nodes.get(&parent).ok_or_else(|| {
                    format!("tree node {} has missing parent {parent}", node.node_id)
                })?;
                if !parent_node.children.contains(&node.node_id) {
                    return Err(format!(
                        "tree parent {parent} does not contain child {}",
                        node.node_id
                    ));
                }
            }
            let unique_children: HashSet<u64> = node.children.iter().copied().collect();
            if unique_children.len() != node.children.len() {
                return Err(format!("tree node {} has duplicate children", node.node_id));
            }
            for child in &node.children {
                if self.nodes.get(child).and_then(|child| child.parent) != Some(node.node_id) {
                    return Err(format!(
                        "tree child {child} does not point back to parent {}",
                        node.node_id
                    ));
                }
            }
            for edge in &node.dependencies {
                if !self.nodes.contains_key(&edge.target_id) {
                    return Err(format!(
                        "tree edge from {} has missing target {}",
                        node.node_id, edge.target_id
                    ));
                }
                if !self
                    .incoming
                    .get(&edge.target_id)
                    .is_some_and(|incoming| incoming.contains(&(node.node_id, edge.edge_type)))
                {
                    return Err(format!(
                        "tree edge from {} is missing its reverse index",
                        node.node_id
                    ));
                }
                if !edge.confidence.is_finite() || !(0.0..=1.0).contains(&edge.confidence) {
                    return Err(format!(
                        "tree edge from {} has invalid confidence",
                        node.node_id
                    ));
                }
            }
        }
        for (target_id, incoming) in &self.incoming {
            if !self.nodes.contains_key(target_id) {
                return Err(format!("tree reverse index has missing target {target_id}"));
            }
            for (source_id, edge_type) in incoming {
                if !self.nodes.get(source_id).is_some_and(|source| {
                    source
                        .dependencies
                        .iter()
                        .any(|edge| edge.target_id == *target_id && edge.edge_type == *edge_type)
                }) {
                    return Err(format!(
                        "tree reverse index from {source_id} has no matching edge"
                    ));
                }
            }
        }
        Ok(())
    }
}

fn edge_priority(edge_type: EdgeType) -> u8 {
    match edge_type {
        EdgeType::Calls => 1,
        EdgeType::Tests => 2,
        EdgeType::Imports => 3,
        EdgeType::Inherits | EdgeType::Implements => 4,
        EdgeType::References => 5,
        EdgeType::Defines | EdgeType::Contains => 8,
    }
}

fn node_cost(node: &TreeNode) -> usize {
    (node.qualified_name.len() + node.signature.as_deref().map_or(0, str::len))
        .div_ceil(4)
        .max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_metadata_and_find() {
        let mut memory = TreeMemory::new();
        let project = memory.insert_node(NodeType::Project, "tst".to_string(), None);
        let file = memory.insert_node_with_metadata(
            NodeType::File,
            "server.rs".to_string(),
            Some(project),
            TreeNodeMetadata {
                file_path: Some("src/bin/server.rs".to_string()),
                content_hash: Some("abc".to_string()),
                ..TreeNodeMetadata::default()
            },
        );
        let function = memory.insert_node_with_metadata(
            NodeType::Function,
            "main".to_string(),
            Some(file),
            TreeNodeMetadata {
                start_line: Some(10),
                end_line: Some(20),
                signature: Some("fn main()".to_string()),
                ..TreeNodeMetadata::default()
            },
        );
        let found = memory.find("main", 10);
        assert_eq!(found[0].node_id, function);
        assert_eq!(found[0].file_path.as_deref(), Some("src/bin/server.rs"));
        assert_eq!(found[0].start_line, Some(10));
    }

    #[test]
    fn removal_cleans_incoming_edges() {
        let mut memory = TreeMemory::new();
        let project = memory.insert_node(NodeType::Project, "project".to_string(), None);
        let first = memory.insert_node(NodeType::Function, "first".to_string(), Some(project));
        let second = memory.insert_node(NodeType::Function, "second".to_string(), Some(project));
        assert!(memory.link(first, second, EdgeType::Calls, 1.0));
        assert!(memory.remove_node(second));
        assert!(memory.nodes.get(&first).unwrap().dependencies.is_empty());
        assert!(
            !memory
                .nodes
                .get(&project)
                .unwrap()
                .children
                .contains(&second)
        );
    }

    #[test]
    fn subgraph_follows_typed_edges() {
        let mut memory = TreeMemory::new();
        let project = memory.insert_node(NodeType::Project, "p".to_string(), None);
        let first = memory.insert_node(NodeType::Function, "first".to_string(), Some(project));
        let second = memory.insert_node(NodeType::Function, "second".to_string(), Some(project));
        memory.link(first, second, EdgeType::Calls, 1.0);
        let nodes = memory.query_subgraph(first, 1);
        assert_eq!(nodes.len(), 3);
        assert!(nodes.iter().any(|node| node.node_id == project));
        assert!(nodes.iter().any(|node| node.node_id == second));
    }

    #[test]
    fn bounded_subgraph_includes_start_and_incoming_callers() {
        let mut memory = TreeMemory::new();
        let project = memory.insert_node(NodeType::Project, "p".to_string(), None);
        let target = memory.insert_node(
            NodeType::Function,
            "a_very_long_target_name".to_string(),
            Some(project),
        );
        let caller = memory.insert_node(NodeType::Function, "caller".to_string(), Some(project));
        assert!(memory.link(caller, target, EdgeType::Calls, 1.0));

        let tiny = memory.query_subgraph_bounded(target, 1, 1, 1);
        assert_eq!(tiny.nodes[0].node_id, target);
        assert!(tiny.truncated);

        let context = memory.query_subgraph_bounded(target, 1, 3, 100);
        assert!(context.nodes.iter().any(|node| node.node_id == caller));
        memory.validate().unwrap();
    }
}
