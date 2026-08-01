use crate::arena::Arena;
use std::collections::HashSet;

pub const NO_PAYLOAD: u32 = u32::MAX;
pub const NULL_NODE: u32 = u32::MAX;

#[repr(C)]
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Node {
    pub ch: u8,
    pub flags: u8,
    pub lo: u32,
    pub eq: u32,
    pub hi: u32,
    pub payload_idx: u32,
    pub next_leaf: u32,
}

impl Node {
    pub fn new(ch: u8) -> Self {
        Self {
            ch,
            flags: 0,
            lo: NULL_NODE,
            eq: NULL_NODE,
            hi: NULL_NODE,
            payload_idx: NO_PAYLOAD,
            next_leaf: NULL_NODE,
        }
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct TernarySearchTrie {
    pub arena: Arena<Node>,
    pub root: u32,
    pub head_leaf: u32,
    pub tail_leaf: u32,
}

impl Default for TernarySearchTrie {
    fn default() -> Self {
        Self::new()
    }
}

impl TernarySearchTrie {
    pub fn new() -> Self {
        Self {
            arena: Arena::new(),
            root: NULL_NODE,
            head_leaf: NULL_NODE,
            tail_leaf: NULL_NODE,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            arena: Arena::with_capacity(capacity),
            root: NULL_NODE,
            head_leaf: NULL_NODE,
            tail_leaf: NULL_NODE,
        }
    }

    pub fn insert(&mut self, key: &[u8], payload_idx: u32) {
        if key.is_empty() {
            return;
        }

        if self.root == NULL_NODE {
            self.root = self.arena.alloc(Node::new(key[0]));
        }

        let mut curr = self.root;
        let mut i = 0;

        loop {
            let node = *self.arena.get(curr).unwrap();
            let ch = key[i];

            if ch < node.ch {
                if node.lo == NULL_NODE {
                    let new_idx = self.arena.alloc(Node::new(ch));
                    self.arena.get_mut(curr).unwrap().lo = new_idx;
                }
                curr = self.arena.get(curr).unwrap().lo;
            } else if ch > node.ch {
                if node.hi == NULL_NODE {
                    let new_idx = self.arena.alloc(Node::new(ch));
                    self.arena.get_mut(curr).unwrap().hi = new_idx;
                }
                curr = self.arena.get(curr).unwrap().hi;
            } else {
                if i + 1 == key.len() {
                    let node_mut = self.arena.get_mut(curr).unwrap();
                    let is_new_leaf = node_mut.payload_idx == NO_PAYLOAD;
                    node_mut.payload_idx = payload_idx;

                    if is_new_leaf {
                        let node_idx = curr;
                        if self.tail_leaf != NULL_NODE {
                            if let Some(tail) = self.arena.get_mut(self.tail_leaf) {
                                tail.next_leaf = node_idx;
                            }
                        } else {
                            self.head_leaf = node_idx;
                        }
                        self.tail_leaf = node_idx;
                    }
                    return;
                }
                if node.eq == NULL_NODE {
                    let new_idx = self.arena.alloc(Node::new(key[i + 1]));
                    self.arena.get_mut(curr).unwrap().eq = new_idx;
                }
                curr = self.arena.get(curr).unwrap().eq;
                i += 1;
            }
        }
    }

    pub fn lookup(&self, key: &[u8]) -> Option<u32> {
        if key.is_empty() || self.root == NULL_NODE {
            return None;
        }

        let mut curr = self.root;
        let mut i = 0;

        loop {
            let node = self.arena.get(curr)?;
            let ch = key[i];

            if ch < node.ch {
                if node.lo == NULL_NODE {
                    return None;
                }
                curr = node.lo;
            } else if ch > node.ch {
                if node.hi == NULL_NODE {
                    return None;
                }
                curr = node.hi;
            } else {
                if i + 1 == key.len() {
                    if node.payload_idx != NO_PAYLOAD {
                        return Some(node.payload_idx);
                    } else {
                        return None;
                    }
                }
                if node.eq == NULL_NODE {
                    return None;
                }
                curr = node.eq;
                i += 1;
            }
        }
    }

    pub fn delete(&mut self, key: &[u8]) -> bool {
        if key.is_empty() || self.root == NULL_NODE {
            return false;
        }

        let mut curr = self.root;
        let mut i = 0;

        loop {
            // Need a copy so we don't hold the immutable borrow while trying to borrow mutably
            let node = *self.arena.get(curr).unwrap();
            let ch = key[i];

            if ch < node.ch {
                if node.lo == NULL_NODE {
                    return false;
                }
                curr = node.lo;
            } else if ch > node.ch {
                if node.hi == NULL_NODE {
                    return false;
                }
                curr = node.hi;
            } else {
                if i + 1 == key.len() {
                    let node_mut = self.arena.get_mut(curr).unwrap();
                    if node_mut.payload_idx != NO_PAYLOAD {
                        node_mut.payload_idx = NO_PAYLOAD;
                        self.unlink_leaf(curr);
                        return true;
                    }
                    return false;
                }
                if node.eq == NULL_NODE {
                    return false;
                }
                curr = node.eq;
                i += 1;
            }
        }
    }

    fn unlink_leaf(&mut self, node_idx: u32) {
        let mut previous = NULL_NODE;
        let mut current = self.head_leaf;
        let mut visited = HashSet::new();
        while current != NULL_NODE && visited.insert(current) {
            let next = self
                .arena
                .get(current)
                .map(|node| node.next_leaf)
                .unwrap_or(NULL_NODE);
            if current == node_idx {
                if previous == NULL_NODE {
                    self.head_leaf = next;
                } else if let Some(previous_node) = self.arena.get_mut(previous) {
                    previous_node.next_leaf = next;
                }
                if self.tail_leaf == current {
                    self.tail_leaf = previous;
                }
                if let Some(node) = self.arena.get_mut(current) {
                    node.next_leaf = NULL_NODE;
                }
                return;
            }
            previous = current;
            current = next;
        }
    }

    pub fn leaf_payload_indices(&self) -> Result<Vec<u32>, String> {
        let mut payloads = Vec::new();
        let mut current = self.head_leaf;
        let mut visited = HashSet::new();
        while current != NULL_NODE {
            if !visited.insert(current) {
                return Err("TST leaf list contains a cycle".to_string());
            }
            let node = self
                .arena
                .get(current)
                .ok_or_else(|| format!("TST leaf list references missing node {current}"))?;
            if node.payload_idx == NO_PAYLOAD {
                return Err(format!("TST leaf node {current} has no payload"));
            }
            payloads.push(node.payload_idx);
            current = node.next_leaf;
        }
        if visited.is_empty() {
            if self.tail_leaf != NULL_NODE {
                return Err("TST empty leaf list has a non-null tail".to_string());
            }
        } else if !visited.contains(&self.tail_leaf) {
            return Err("TST leaf tail is not reachable from the head".to_string());
        }
        Ok(payloads)
    }

    pub fn validate(&self) -> Result<(), String> {
        self.arena.validate()?;
        if self.root == NULL_NODE {
            if !self.arena.is_empty() {
                return Err("TST has allocated nodes but no root".to_string());
            }
            return self.leaf_payload_indices().map(|_| ());
        }
        if self.arena.get(self.root).is_none() {
            return Err("TST root references a missing node".to_string());
        }

        let mut reachable = HashSet::new();
        let mut terminals = HashSet::new();
        let mut stack = vec![self.root];
        while let Some(index) = stack.pop() {
            if !reachable.insert(index) {
                continue;
            }
            let node = self
                .arena
                .get(index)
                .ok_or_else(|| format!("TST references missing node {index}"))?;
            for child in [node.lo, node.eq, node.hi] {
                if child != NULL_NODE {
                    if self.arena.get(child).is_none() {
                        return Err(format!("TST node {index} references missing child {child}"));
                    }
                    stack.push(child);
                }
            }
            if node.payload_idx != NO_PAYLOAD {
                terminals.insert(node.payload_idx);
            }
        }
        if reachable.len() != self.arena.len() {
            return Err("TST contains unreachable arena nodes".to_string());
        }
        let leaves: HashSet<u32> = self.leaf_payload_indices()?.into_iter().collect();
        if leaves != terminals {
            return Err("TST leaf list does not match terminal nodes".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    #[test]
    fn test_node_size() {
        assert_eq!(size_of::<Node>(), 24);
    }

    #[test]
    fn test_insert_and_lookup() {
        let mut tst = TernarySearchTrie::new();
        tst.insert(b"hello", 100);
        tst.insert(b"world", 200);
        tst.insert(b"hell", 300); // prefix

        assert_eq!(tst.lookup(b"hello"), Some(100));
        assert_eq!(tst.lookup(b"world"), Some(200));
        assert_eq!(tst.lookup(b"hell"), Some(300));
        assert_eq!(tst.lookup(b"he"), None);
        assert_eq!(tst.lookup(b"world2"), None);
    }

    #[test]
    fn test_delete() {
        let mut tst = TernarySearchTrie::new();
        tst.insert(b"test", 42);
        assert_eq!(tst.lookup(b"test"), Some(42));

        let deleted = tst.delete(b"test");
        assert!(deleted);
        assert_eq!(tst.lookup(b"test"), None);

        let deleted_again = tst.delete(b"test");
        assert!(!deleted_again);
    }

    #[test]
    fn test_leaf_traversal() {
        let mut tst = TernarySearchTrie::new();
        tst.insert(b"a", 1);
        tst.insert(b"b", 2);
        tst.insert(b"c", 3);

        let mut curr = tst.head_leaf;
        let mut count = 0;
        while curr != NULL_NODE {
            count += 1;
            curr = tst.arena.get(curr).unwrap().next_leaf;
        }
        assert_eq!(count, 3);
    }

    #[test]
    fn delete_and_reinsert_does_not_create_a_leaf_cycle() {
        let mut tst = TernarySearchTrie::new();
        tst.insert(b"same", 1);
        assert!(tst.delete(b"same"));
        tst.insert(b"same", 2);
        assert_eq!(tst.lookup(b"same"), Some(2));
        assert_eq!(tst.leaf_payload_indices().unwrap(), vec![2]);
        tst.validate().unwrap();
    }
}
