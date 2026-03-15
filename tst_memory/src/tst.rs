use std::mem::size_of;
use crate::arena::Arena;

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
                if node.lo == NULL_NODE { return None; }
                curr = node.lo;
            } else if ch > node.ch {
                if node.hi == NULL_NODE { return None; }
                curr = node.hi;
            } else {
                if i + 1 == key.len() {
                    if node.payload_idx != NO_PAYLOAD {
                        return Some(node.payload_idx);
                    } else {
                        return None;
                    }
                }
                if node.eq == NULL_NODE { return None; }
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
                if node.lo == NULL_NODE { return false; }
                curr = node.lo;
            } else if ch > node.ch {
                if node.hi == NULL_NODE { return false; }
                curr = node.hi;
            } else {
                if i + 1 == key.len() {
                    let node_mut = self.arena.get_mut(curr).unwrap();
                    if node_mut.payload_idx != NO_PAYLOAD {
                        node_mut.payload_idx = NO_PAYLOAD;
                        return true;
                    }
                    return false;
                }
                if node.eq == NULL_NODE { return false; }
                curr = node.eq;
                i += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
