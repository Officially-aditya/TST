use std::collections::HashMap;

pub type ModelFamily = String;

pub struct TokenizerCache {
    cache: HashMap<(ModelFamily, String), Vec<u32>>,
    capacity: usize,
}

impl TokenizerCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(capacity),
            capacity,
        }
    }

    pub fn default() -> Self {
        Self::new(50_000)
    }

    pub fn resolve_tokens(&mut self, model: &str, canonical_string: &str) -> Vec<u32> {
        let key = (model.to_string(), canonical_string.to_string());
        if let Some(tokens) = self.cache.get(&key) {
            return tokens.clone();
        }

        // Mock tokenizer for MVP
        let mut tokens = Vec::new();
        for ch in canonical_string.bytes() {
            tokens.push(ch as u32);
        }

        if self.cache.len() >= self.capacity {
            // Evict an arbitrary entry
            let key_to_remove = self.cache.keys().next().cloned().unwrap();
            self.cache.remove(&key_to_remove);
        }

        self.cache.insert(key, tokens.clone());
        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_cache() {
        let mut cache = TokenizerCache::new(2);
        let _ = cache.resolve_tokens("model1", "hello");
        let _ = cache.resolve_tokens("model1", "world");
        
        assert_eq!(cache.cache.len(), 2);
        
        let _ = cache.resolve_tokens("model1", "test");
        assert_eq!(cache.cache.len(), 2);
    }
}
