use std::path::Path;
use std::time::{Duration, Instant};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Value, json};

use crate::errors::{KernelError, KernelResult};
use crate::kernel::{Kernel, MemoryLayer, STMConfig};
use crate::payload::{Payload, PayloadData, PayloadHeader};
use crate::persistence::{PersistenceHandler, RecoveryReport};
use crate::tree::{EdgeType, NodeType, TreeNodeMetadata};
use crate::types::Timestamp;

pub const PROTOCOL_VERSION: u32 = 1;
const MAX_REQUEST_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    KernelPing,
    KernelStatus,
    KernelShutdown,
    MemoryStore,
    MemoryGet,
    MemoryUpdate,
    MemoryDelete,
    MemorySearch,
    TreeClear,
    TreeInsert,
    TreeRemove,
    TreeLink,
    TreeUnlink,
    TreeQuery,
    TreeFind,
    PersistenceSave,
    PersistenceStatus,
}

impl Operation {
    pub fn parse(value: &str) -> KernelResult<Self> {
        match value {
            "kernel.ping" => Ok(Self::KernelPing),
            "kernel.status" => Ok(Self::KernelStatus),
            "kernel.shutdown" => Ok(Self::KernelShutdown),
            "memory.store" => Ok(Self::MemoryStore),
            "memory.get" => Ok(Self::MemoryGet),
            "memory.update" => Ok(Self::MemoryUpdate),
            "memory.delete" => Ok(Self::MemoryDelete),
            "memory.search" => Ok(Self::MemorySearch),
            "tree.clear" => Ok(Self::TreeClear),
            "tree.insert" => Ok(Self::TreeInsert),
            "tree.remove" => Ok(Self::TreeRemove),
            "tree.link" => Ok(Self::TreeLink),
            "tree.unlink" => Ok(Self::TreeUnlink),
            "tree.query" => Ok(Self::TreeQuery),
            "tree.find" => Ok(Self::TreeFind),
            "persistence.save" => Ok(Self::PersistenceSave),
            "persistence.status" => Ok(Self::PersistenceStatus),
            unknown => Err(KernelError::new(
                "unknown_operation",
                format!("unknown operation: {unknown}"),
            )),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireRequest {
    protocol_version: u32,
    request_id: String,
    operation: String,
    params: Value,
}

#[derive(Debug)]
pub struct ProtocolRequest {
    pub request_id: String,
    pub operation: Operation,
    pub params: Value,
}

#[derive(Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResponseMetrics {
    pub kernel_ms: f64,
}

#[derive(Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProtocolResponse {
    pub protocol_version: u32,
    pub request_id: String,
    pub ok: bool,
    pub result: Option<Value>,
    pub error: Option<KernelError>,
    pub metrics: ResponseMetrics,
}

impl ProtocolResponse {
    fn success(request_id: String, result: Value, started: Instant) -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            request_id,
            ok: true,
            result: Some(result),
            error: None,
            metrics: ResponseMetrics {
                kernel_ms: started.elapsed().as_secs_f64() * 1_000.0,
            },
        }
    }

    fn failure(request_id: String, error: KernelError, started: Instant) -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            request_id,
            ok: false,
            result: None,
            error: Some(error),
            metrics: ResponseMetrics {
                kernel_ms: started.elapsed().as_secs_f64() * 1_000.0,
            },
        }
    }
}

#[derive(Debug)]
pub struct ProtocolOutcome {
    pub response: ProtocolResponse,
    pub shutdown: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProtocolMemoryLayer {
    Stm,
    Ltm,
}

impl ProtocolMemoryLayer {
    fn kernel_layer(self) -> MemoryLayer {
        match self {
            Self::Stm => MemoryLayer::STM,
            Self::Ltm => MemoryLayer::LTM,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Stm => "stm",
            Self::Ltm => "ltm",
        }
    }
}

impl<'de> Deserialize<'de> for ProtocolMemoryLayer {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        match value.as_str() {
            "stm" => Ok(Self::Stm),
            "ltm" => Ok(Self::Ltm),
            _ => Err(serde::de::Error::custom(format!(
                "invalid memory layer: {value}"
            ))),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum IncomingPayload {
    Full(Payload),
    Simple(SimplePayload),
}

impl IncomingPayload {
    fn into_payload(self) -> KernelResult<Payload> {
        match self {
            Self::Full(payload) => Ok(payload),
            Self::Simple(payload) => payload.into_payload(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SimplePayload {
    #[serde(rename = "type")]
    payload_type: String,
    data: Value,
}

impl SimplePayload {
    fn into_payload(self) -> KernelResult<Payload> {
        if self.data.get("memory_type").is_some() {
            if !matches!(self.payload_type.as_str(), "preference" | "token_stats") {
                return Err(KernelError::invalid_params(format!(
                    "unknown payload type: {}",
                    self.payload_type
                )));
            }
            let payload_type_code = if self.payload_type == "preference" {
                5
            } else {
                1
            };
            let data: MemoryRecordData = deserialize_payload_data(self.data)?;
            return Ok(Payload {
                header: PayloadHeader {
                    payload_type: payload_type_code,
                    version: 1,
                    created_ts: Timestamp(data.created_at),
                    last_access_ts: Timestamp(data.updated_at),
                    access_count: 0,
                },
                data: PayloadData::MemoryRecord {
                    payload_type: self.payload_type,
                    key: data.key,
                    value: data.value,
                    memory_type: data.memory_type,
                    source_text: data.source_text,
                    created_at: data.created_at,
                    updated_at: data.updated_at,
                    confidence: data.confidence,
                    tags: data.tags,
                    source: data.source,
                    layer: data.layer,
                    reinforcement_score: data.reinforcement_score,
                    deleted: data.deleted,
                },
            });
        }
        let (payload_type, data) = match self.payload_type.as_str() {
            "token_stats" => {
                let data: TokenStatsData = deserialize_payload_data(self.data)?;
                (
                    1,
                    PayloadData::TokenStats {
                        canonical_form: data.canonical_form,
                        frequency: data.frequency,
                        decay_score: data.decay_score,
                        preferred_tokenizer_origin: data.preferred_tokenizer_origin,
                    },
                )
            }
            "phrase_meta" => {
                let data: PhraseMetaData = deserialize_payload_data(self.data)?;
                (
                    2,
                    PayloadData::PhraseMeta {
                        canonical_phrase: data.canonical_phrase,
                        usage_count: data.usage_count,
                        domain_mask: data.domain_mask,
                    },
                )
            }
            "concept_anchor" => {
                let data: ConceptAnchorData = deserialize_payload_data(self.data)?;
                (
                    3,
                    PayloadData::ConceptAnchor {
                        concept_id: data.concept_id,
                        related_tokens: data.related_tokens,
                        strength: data.strength,
                    },
                )
            }
            "structure_pattern" => {
                let data: StructurePatternData = deserialize_payload_data(self.data)?;
                (
                    4,
                    PayloadData::StructurePattern {
                        pattern_id: data.pattern_id,
                        steps: data.steps,
                        success_score: data.success_score,
                    },
                )
            }
            "preference" => {
                let data: PreferenceData = deserialize_payload_data(self.data)?;
                (
                    5,
                    PayloadData::Preference {
                        key: data.key,
                        value: data.value,
                        weight: data.weight,
                    },
                )
            }
            unknown => {
                return Err(KernelError::invalid_params(format!(
                    "unknown payload type: {unknown}"
                )));
            }
        };
        Ok(Payload {
            header: PayloadHeader {
                payload_type,
                version: 1,
                created_ts: Timestamp(0),
                last_access_ts: Timestamp(0),
                access_count: 0,
            },
            data,
        })
    }
}

fn deserialize_payload_data<T: DeserializeOwned>(value: Value) -> KernelResult<T> {
    serde_json::from_value(value)
        .map_err(|error| KernelError::invalid_params(format!("invalid payload data: {error}")))
}

fn default_one_u32() -> u32 {
    1
}
fn default_one_f32() -> f32 {
    1.0
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenStatsData {
    canonical_form: String,
    #[serde(default = "default_one_u32")]
    frequency: u32,
    #[serde(default = "default_one_f32")]
    decay_score: f32,
    #[serde(default)]
    preferred_tokenizer_origin: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PhraseMetaData {
    canonical_phrase: String,
    #[serde(default = "default_one_u32")]
    usage_count: u32,
    #[serde(default)]
    domain_mask: u16,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConceptAnchorData {
    concept_id: u32,
    #[serde(default)]
    related_tokens: Vec<String>,
    #[serde(default = "default_one_f32")]
    strength: f32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StructurePatternData {
    pattern_id: u32,
    #[serde(default)]
    steps: Vec<u8>,
    #[serde(default = "default_one_f32")]
    success_score: f32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PreferenceData {
    key: String,
    value: String,
    #[serde(default = "default_one_f32")]
    weight: f32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MemoryRecordData {
    key: String,
    value: String,
    memory_type: String,
    source_text: String,
    created_at: u64,
    updated_at: u64,
    confidence: f32,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default = "default_user_source")]
    source: String,
    layer: String,
    #[serde(default)]
    reinforcement_score: f32,
    #[serde(default)]
    deleted: bool,
}

fn default_user_source() -> String {
    "user".to_string()
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EmptyParams {}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StoreParams {
    layer: ProtocolMemoryLayer,
    key: String,
    payload: IncomingPayload,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct KeyParams {
    layer: ProtocolMemoryLayer,
    key: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchParams {
    #[serde(default)]
    layer: Option<ProtocolMemoryLayer>,
    query: String,
    #[serde(default)]
    prefix: Option<String>,
    #[serde(default = "default_search_limit")]
    limit: usize,
}

fn default_search_limit() -> usize {
    10
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TreeInsertParams {
    node_type: String,
    name: String,
    #[serde(default)]
    parent_id: Option<u64>,
    #[serde(default)]
    qualified_name: Option<String>,
    #[serde(default)]
    file_path: Option<String>,
    #[serde(default)]
    start_line: Option<u32>,
    #[serde(default)]
    end_line: Option<u32>,
    #[serde(default)]
    signature: Option<String>,
    #[serde(default)]
    content_hash: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TreeNodeParams {
    node_id: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TreeLinkParams {
    source_id: u64,
    target_id: u64,
    #[serde(default)]
    edge_type: Option<EdgeType>,
    #[serde(default = "default_one_f32")]
    confidence: f32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TreeQueryParams {
    node_id: u64,
    depth: u32,
    #[serde(default = "default_tree_max_nodes")]
    max_nodes: usize,
    #[serde(default = "default_tree_token_budget")]
    token_budget: usize,
}

fn default_tree_max_nodes() -> usize {
    100
}

fn default_tree_token_budget() -> usize {
    2_000
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TreeFindParams {
    name: String,
    #[serde(default = "default_search_limit")]
    limit: usize,
}

fn parse_node_type(value: &str) -> KernelResult<NodeType> {
    match value.to_ascii_lowercase().as_str() {
        "project" => Ok(NodeType::Project),
        "directory" => Ok(NodeType::Directory),
        "file" => Ok(NodeType::File),
        "class" => Ok(NodeType::Class),
        "interface" => Ok(NodeType::Interface),
        "struct" => Ok(NodeType::Struct),
        "enum" => Ok(NodeType::Enum),
        "trait" => Ok(NodeType::Trait),
        "function" => Ok(NodeType::Function),
        "symbol" => Ok(NodeType::Symbol),
        "module" => Ok(NodeType::Module),
        "external" => Ok(NodeType::External),
        _ => Err(KernelError::invalid_params(format!(
            "invalid tree node type: {value}"
        ))),
    }
}

pub struct ProtocolService {
    pub kernel: Kernel,
    persistence: PersistenceHandler,
    recovery: RecoveryReport,
    dirty: bool,
    dirty_since: Option<Instant>,
    save_debounce: Duration,
    last_save: Instant,
}

impl ProtocolService {
    pub fn recover(
        snapshot_path: impl AsRef<Path>,
        save_debounce: Duration,
    ) -> Result<Self, String> {
        Self::recover_with_config(snapshot_path, save_debounce, STMConfig::default())
    }

    pub fn recover_with_config(
        snapshot_path: impl AsRef<Path>,
        save_debounce: Duration,
        stm_config: STMConfig,
    ) -> Result<Self, String> {
        stm_config.validate().map_err(|error| error.to_string())?;
        let persistence = PersistenceHandler::new(snapshot_path);
        let mut kernel = Kernel::with_stm_config(stm_config);
        let recovery = persistence.recover(&mut kernel)?;
        let now = Instant::now();
        Ok(Self {
            kernel,
            persistence,
            recovery,
            dirty: false,
            dirty_since: None,
            save_debounce,
            last_save: now,
        })
    }

    pub fn from_parts(
        kernel: Kernel,
        persistence: PersistenceHandler,
        recovery: RecoveryReport,
        save_debounce: Duration,
    ) -> Self {
        let now = Instant::now();
        Self {
            kernel,
            persistence,
            recovery,
            dirty: false,
            dirty_since: None,
            save_debounce,
            last_save: now,
        }
    }

    pub fn handle_line(&mut self, line: &str) -> ProtocolOutcome {
        let started = Instant::now();
        let request_id = serde_json::from_str::<Value>(line)
            .ok()
            .and_then(|value| value.get("request_id")?.as_str().map(str::to_string))
            .unwrap_or_else(|| "unknown".to_string());
        if line.len() > MAX_REQUEST_BYTES {
            self.kernel.metrics.protocol_errors =
                self.kernel.metrics.protocol_errors.saturating_add(1);
            return ProtocolOutcome {
                response: ProtocolResponse::failure(
                    request_id,
                    KernelError::invalid_request("request exceeds the 4 MiB protocol limit"),
                    started,
                ),
                shutdown: false,
            };
        }
        if let Err(error) = self.maintenance_tick() {
            self.kernel.metrics.protocol_errors =
                self.kernel.metrics.protocol_errors.saturating_add(1);
            return ProtocolOutcome {
                response: ProtocolResponse::failure(request_id, error, started),
                shutdown: false,
            };
        }
        let response = match self.parse_request(line) {
            Ok(request) => {
                let request_id = request.request_id.clone();
                match self.dispatch(request) {
                    Ok((result, shutdown)) => {
                        return ProtocolOutcome {
                            response: ProtocolResponse::success(request_id, result, started),
                            shutdown,
                        };
                    }
                    Err(error) => ProtocolResponse::failure(request_id, error, started),
                }
            }
            Err(error) => ProtocolResponse::failure(request_id, error, started),
        };
        self.kernel.metrics.protocol_errors = self.kernel.metrics.protocol_errors.saturating_add(1);
        ProtocolOutcome {
            response,
            shutdown: false,
        }
    }

    fn parse_request(&self, line: &str) -> KernelResult<ProtocolRequest> {
        let wire: WireRequest = serde_json::from_str(line).map_err(|error| {
            KernelError::invalid_request(format!("invalid request envelope: {error}"))
        })?;
        if wire.protocol_version != PROTOCOL_VERSION {
            return Err(KernelError::new(
                "unsupported_protocol_version",
                format!(
                    "unsupported protocol version {}; expected {}",
                    wire.protocol_version, PROTOCOL_VERSION
                ),
            ));
        }
        if wire.request_id.is_empty() || wire.request_id.len() > 128 {
            return Err(KernelError::invalid_request(
                "request_id must contain between 1 and 128 bytes",
            ));
        }
        Ok(ProtocolRequest {
            request_id: wire.request_id,
            operation: Operation::parse(&wire.operation)?,
            params: wire.params,
        })
    }

    fn params<T: DeserializeOwned>(&self, operation: &str, value: Value) -> KernelResult<T> {
        serde_json::from_value(value).map_err(|error| {
            KernelError::invalid_params(format!("invalid parameters for {operation}: {error}"))
        })
    }

    fn dispatch(&mut self, request: ProtocolRequest) -> KernelResult<(Value, bool)> {
        match request.operation {
            Operation::KernelPing => {
                let _: EmptyParams = self.params("kernel.ping", request.params)?;
                Ok((json!({"pong": true}), false))
            }
            Operation::KernelStatus => {
                let _: EmptyParams = self.params("kernel.status", request.params)?;
                Ok((self.status_value(), false))
            }
            Operation::KernelShutdown => {
                let _: EmptyParams = self.params("kernel.shutdown", request.params)?;
                let saved = self.flush()?;
                Ok((json!({"shutdown": true, "saved": saved}), true))
            }
            Operation::MemoryStore => {
                let params: StoreParams = self.params("memory.store", request.params)?;
                let layer = params.layer;
                let promotions_before = self.kernel.metrics.stm_promotions;
                let promoted = self.kernel.store_memory(
                    layer.kernel_layer(),
                    params.key.as_bytes(),
                    params.payload.into_payload()?,
                )?;
                if layer == ProtocolMemoryLayer::Ltm
                    || self.kernel.metrics.stm_promotions > promotions_before
                {
                    self.mark_dirty_and_maybe_save()?;
                }
                Ok((
                    json!({
                        "stored": true,
                        "layer": layer.as_str(),
                        "key": params.key,
                        "promoted": promoted
                    }),
                    false,
                ))
            }
            Operation::MemoryGet => {
                let params: KeyParams = self.params("memory.get", request.params)?;
                let layer = params.layer;
                let promotions_before = self.kernel.metrics.stm_promotions;
                let payload = self
                    .kernel
                    .read_memory(layer.kernel_layer(), params.key.as_bytes())?;
                if (layer == ProtocolMemoryLayer::Ltm && payload.is_some())
                    || self.kernel.metrics.stm_promotions > promotions_before
                {
                    self.mark_dirty_and_maybe_save()?;
                }
                Ok((
                    json!({
                        "found": payload.is_some(),
                        "layer": layer.as_str(),
                        "key": params.key,
                        "payload": payload
                    }),
                    false,
                ))
            }
            Operation::MemoryUpdate => {
                let params: StoreParams = self.params("memory.update", request.params)?;
                let layer = params.layer;
                let promotions_before = self.kernel.metrics.stm_promotions;
                let promoted = self.kernel.update_memory(
                    layer.kernel_layer(),
                    params.key.as_bytes(),
                    params.payload.into_payload()?,
                )?;
                let updated = promoted.is_some();
                if updated
                    && (layer == ProtocolMemoryLayer::Ltm
                        || self.kernel.metrics.stm_promotions > promotions_before)
                {
                    self.mark_dirty_and_maybe_save()?;
                }
                Ok((
                    json!({
                        "updated": updated,
                        "layer": layer.as_str(),
                        "key": params.key,
                        "promoted": promoted.unwrap_or(false)
                    }),
                    false,
                ))
            }
            Operation::MemoryDelete => {
                let params: KeyParams = self.params("memory.delete", request.params)?;
                let layer = params.layer;
                let deleted = self
                    .kernel
                    .delete_memory(layer.kernel_layer(), params.key.as_bytes())?;
                if deleted && layer == ProtocolMemoryLayer::Ltm {
                    self.mark_dirty_and_maybe_save()?;
                }
                Ok((
                    json!({"deleted": deleted, "layer": layer.as_str(), "key": params.key}),
                    false,
                ))
            }
            Operation::MemorySearch => {
                let params: SearchParams = self.params("memory.search", request.params)?;
                if !(1..=1_000).contains(&params.limit) {
                    return Err(KernelError::invalid_params(
                        "memory.search limit must be between 1 and 1000",
                    ));
                }
                let matches = self.kernel.search_memory(
                    params.layer.map(ProtocolMemoryLayer::kernel_layer),
                    &params.query,
                    params.prefix.as_deref(),
                    params.limit,
                )?;
                let count = matches.len();
                Ok((json!({"matches": matches, "count": count}), false))
            }
            Operation::TreeClear => {
                let _: EmptyParams = self.params("tree.clear", request.params)?;
                self.kernel.tree.clear();
                Ok((json!({"cleared": true}), false))
            }
            Operation::TreeInsert => {
                let params: TreeInsertParams = self.params("tree.insert", request.params)?;
                if params.name.trim().is_empty() || params.name.len() > 1_024 {
                    return Err(KernelError::invalid_params(
                        "tree node name must contain between 1 and 1024 bytes",
                    ));
                }
                if let Some(parent_id) = params.parent_id
                    && !self.kernel.tree.contains_node(parent_id)
                {
                    return Err(KernelError::not_found(format!(
                        "tree parent node {parent_id} does not exist"
                    )));
                }
                if matches!((params.start_line, params.end_line), (Some(start), Some(end)) if end < start)
                {
                    return Err(KernelError::invalid_params(
                        "tree node end_line must be greater than or equal to start_line",
                    ));
                }
                if params.start_line.is_some() != params.end_line.is_some() {
                    return Err(KernelError::invalid_params(
                        "tree source spans require both start_line and end_line",
                    ));
                }
                if params
                    .qualified_name
                    .as_ref()
                    .is_some_and(|value| value.is_empty() || value.len() > 4_096)
                {
                    return Err(KernelError::invalid_params(
                        "tree qualified_name must contain between 1 and 4096 bytes",
                    ));
                }
                if params
                    .signature
                    .as_ref()
                    .is_some_and(|value| value.len() > 16_384)
                {
                    return Err(KernelError::invalid_params(
                        "tree signature exceeds the 16384-byte limit",
                    ));
                }
                if let Some(file_path) = &params.file_path {
                    validate_tree_file_path(file_path)?;
                }
                if params.content_hash.as_ref().is_some_and(|value| {
                    value.len() != 64
                        || !value
                            .bytes()
                            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                }) {
                    return Err(KernelError::invalid_params(
                        "tree content_hash must be a lowercase SHA-256 digest",
                    ));
                }
                let node_type = parse_node_type(&params.node_type)?;
                let node_id = self.kernel.tree.insert_node_with_metadata(
                    node_type,
                    params.name,
                    params.parent_id,
                    TreeNodeMetadata {
                        qualified_name: params.qualified_name,
                        file_path: params.file_path,
                        start_line: params.start_line,
                        end_line: params.end_line,
                        signature: params.signature,
                        content_hash: params.content_hash,
                    },
                );
                Ok((json!({"node_id": node_id}), false))
            }
            Operation::TreeRemove => {
                let params: TreeNodeParams = self.params("tree.remove", request.params)?;
                let removed = self.kernel.tree.remove_node(params.node_id);
                Ok((
                    json!({"removed": removed, "node_id": params.node_id}),
                    false,
                ))
            }
            Operation::TreeLink => {
                let params: TreeLinkParams = self.params("tree.link", request.params)?;
                if !params.confidence.is_finite() || !(0.0..=1.0).contains(&params.confidence) {
                    return Err(KernelError::invalid_params(
                        "tree edge confidence must be between zero and one",
                    ));
                }
                let edge_type = params.edge_type.unwrap_or(EdgeType::References);
                let linked = self.kernel.tree.link(
                    params.source_id,
                    params.target_id,
                    edge_type,
                    params.confidence,
                );
                if !linked {
                    return Err(KernelError::not_found(
                        "tree link source or target node does not exist",
                    ));
                }
                Ok((json!({"linked": true}), false))
            }
            Operation::TreeUnlink => {
                let params: TreeLinkParams = self.params("tree.unlink", request.params)?;
                let unlinked =
                    self.kernel
                        .tree
                        .unlink(params.source_id, params.target_id, params.edge_type);
                Ok((json!({"unlinked": unlinked}), false))
            }
            Operation::TreeQuery => {
                let params: TreeQueryParams = self.params("tree.query", request.params)?;
                if !self.kernel.tree.contains_node(params.node_id) {
                    return Err(KernelError::not_found(format!(
                        "tree node {} does not exist",
                        params.node_id
                    )));
                }
                if params.depth > 32 {
                    return Err(KernelError::invalid_params(
                        "tree query depth exceeds the maximum of 32",
                    ));
                }
                if !(1..=10_000).contains(&params.max_nodes)
                    || !(1..=1_000_000).contains(&params.token_budget)
                {
                    return Err(KernelError::invalid_params(
                        "tree query budgets are outside the supported range",
                    ));
                }
                let result = self.kernel.tree.query_subgraph_bounded(
                    params.node_id,
                    params.depth,
                    params.max_nodes,
                    params.token_budget,
                );
                Ok((json!(result), false))
            }
            Operation::TreeFind => {
                let params: TreeFindParams = self.params("tree.find", request.params)?;
                if !(1..=1_000).contains(&params.limit) {
                    return Err(KernelError::invalid_params(
                        "tree.find limit must be between 1 and 1000",
                    ));
                }
                let nodes = self.kernel.tree.find(&params.name, params.limit);
                Ok((json!({"nodes": nodes}), false))
            }
            Operation::PersistenceSave => {
                let _: EmptyParams = self.params("persistence.save", request.params)?;
                self.persistence
                    .save_snapshot(&self.kernel)
                    .map_err(KernelError::persistence)?;
                self.dirty = false;
                self.dirty_since = None;
                self.last_save = Instant::now();
                Ok((
                    json!({
                        "saved": true,
                        "size_bytes": self.persistence.snapshot_size_bytes()
                    }),
                    false,
                ))
            }
            Operation::PersistenceStatus => {
                let _: EmptyParams = self.params("persistence.status", request.params)?;
                Ok((self.persistence_value(), false))
            }
        }
    }

    fn mark_dirty_and_maybe_save(&mut self) -> KernelResult<()> {
        if !self.dirty {
            self.dirty = true;
            self.dirty_since = Some(Instant::now());
        }
        self.maybe_save_due()
    }

    fn maybe_save_due(&mut self) -> KernelResult<()> {
        let due = self.dirty
            && self
                .dirty_since
                .is_some_and(|started| started.elapsed() >= self.save_debounce);
        if due {
            self.persistence
                .save_snapshot(&self.kernel)
                .map_err(KernelError::persistence)?;
            self.dirty = false;
            self.dirty_since = None;
            self.last_save = Instant::now();
        }
        Ok(())
    }

    pub fn maintenance_tick(&mut self) -> KernelResult<()> {
        if self.kernel.maintenance() > 0 {
            self.mark_dirty_and_maybe_save()?;
        }
        self.maybe_save_due()
    }

    pub fn note_external_ltm_mutation(&mut self) -> KernelResult<()> {
        self.mark_dirty_and_maybe_save()
    }

    /// Flushes dirty persistent state. The return value says whether a write
    /// occurred; callers use it in the shutdown acknowledgement.
    pub fn flush(&mut self) -> KernelResult<bool> {
        if !self.dirty {
            return Ok(false);
        }
        self.persistence
            .save_snapshot(&self.kernel)
            .map_err(KernelError::persistence)?;
        self.dirty = false;
        self.dirty_since = None;
        self.last_save = Instant::now();
        Ok(true)
    }

    fn persistence_value(&self) -> Value {
        json!({
            "dirty": self.dirty,
            "snapshot_path": self.persistence.snapshot_path().display().to_string(),
            "snapshot_size_bytes": self.persistence.snapshot_size_bytes(),
            "snapshot_age_seconds": self.persistence.snapshot_age_seconds(),
            "recovery": self.recovery.source,
            "recovery_warning": self.recovery.warning,
            "preserved_corrupt_path": self.recovery.preserved_corrupt_path,
        })
    }

    fn status_value(&self) -> Value {
        json!({
            "protocol_version": PROTOCOL_VERSION,
            "stm": {
                "entries": self.kernel.stm.len(),
                "capacity": self.kernel.stm.capacity,
                "evictions": self.kernel.metrics.stm_evictions,
                "expirations": self.kernel.metrics.stm_expirations,
                "promotions": self.kernel.metrics.stm_promotions,
            },
            "ltm": {
                "entries": self.kernel.ltm.len(),
            },
            "tree": {
                "nodes": self.kernel.tree.nodes.len(),
                "edges": self.kernel.tree.edge_count(),
            },
            "retrieval": {
                "hits": self.kernel.metrics.retrieval_hits,
                "misses": self.kernel.metrics.retrieval_misses,
                "searches": self.kernel.metrics.retrieval_searches,
                "average_result_count": if self.kernel.metrics.retrieval_searches == 0 {
                    0.0
                } else {
                    self.kernel.metrics.retrieval_result_count as f64
                        / self.kernel.metrics.retrieval_searches as f64
                },
            },
            "stm_config": {
                "half_life_seconds": self.kernel.stm_config.half_life_seconds,
                "promotion_threshold": self.kernel.stm_config.promotion_threshold,
                "read_reinforcement": self.kernel.stm_config.read_reinforcement,
                "write_reinforcement": self.kernel.stm_config.write_reinforcement,
                "expiry_score": self.kernel.stm_config.expiry_score,
            },
            "protocol_errors": self.kernel.metrics.protocol_errors,
            "persistence": self.persistence_value(),
        })
    }
}

fn validate_tree_file_path(value: &str) -> KernelResult<()> {
    if value.is_empty()
        || value.len() > 4_096
        || value.starts_with('/')
        || value.contains('\\')
        || value
            .split('/')
            .any(|component| component.is_empty() || component == "." || component == "..")
        || value.as_bytes().get(1) == Some(&b':')
    {
        return Err(KernelError::invalid_params(
            "tree file_path must be a normalized project-relative POSIX path",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::RecoverySource;

    fn service() -> ProtocolService {
        let path = std::env::temp_dir().join(format!(
            "tst-protocol-{}-{}.snapshot",
            std::process::id(),
            Kernel::now().0
        ));
        ProtocolService::from_parts(
            Kernel::new(),
            PersistenceHandler::new(path),
            RecoveryReport {
                source: RecoverySource::Empty,
                warning: None,
                preserved_corrupt_path: None,
            },
            Duration::ZERO,
        )
    }

    fn request(request_id: &str, operation: &str, params: Value) -> String {
        json!({
            "protocol_version": 1,
            "request_id": request_id,
            "operation": operation,
            "params": params,
        })
        .to_string()
    }

    #[test]
    fn ping_echoes_request_id() {
        let mut service = service();
        let outcome = service.handle_line(&request("ping-1", "kernel.ping", json!({})));
        assert!(outcome.response.ok);
        assert_eq!(outcome.response.request_id, "ping-1");
        assert_eq!(outcome.response.result, Some(json!({"pong": true})));
    }

    #[test]
    fn unknown_operation_and_unknown_fields_are_rejected() {
        let mut service = service();
        let wrong_version = service.handle_line(
            &json!({
                "protocol_version": 99,
                "request_id": "version-1",
                "operation": "kernel.ping",
                "params": {}
            })
            .to_string(),
        );
        assert_eq!(
            wrong_version.response.error.unwrap().code,
            "unsupported_protocol_version"
        );
        let unknown = service.handle_line(&request("bad-1", "memory.teleport", json!({})));
        assert_eq!(unknown.response.error.unwrap().code, "unknown_operation");

        let bad_params = service.handle_line(&request(
            "bad-2",
            "memory.get",
            json!({"layer": "ltm", "key": "missing", "unexpected": true}),
        ));
        assert_eq!(bad_params.response.error.unwrap().code, "invalid_params");
    }

    #[test]
    fn simple_payload_round_trip() {
        let mut service = service();
        let store = service.handle_line(&request(
            "store-1",
            "memory.store",
            json!({
                "layer": "ltm",
                "key": "user:default:preference:language",
                "payload": {
                    "type": "preference",
                    "data": {"key": "language", "value": "TypeScript", "weight": 1.0}
                }
            }),
        ));
        assert!(store.response.ok, "{:?}", store.response.error);
        let get = service.handle_line(&request(
            "get-1",
            "memory.get",
            json!({"layer": "ltm", "key": "user:default:preference:language"}),
        ));
        assert!(get.response.ok);
        assert_eq!(
            get.response.result.unwrap()["payload"]["data"]["Preference"]["value"],
            "TypeScript"
        );
    }

    #[test]
    fn tree_metadata_and_integrity_operations() {
        let mut service = service();
        let root = service.handle_line(&request(
            "tree-1",
            "tree.insert",
            json!({"node_type": "Project", "name": "demo"}),
        ));
        let root_id = root.response.result.unwrap()["node_id"].as_u64().unwrap();
        let function = service.handle_line(&request(
            "tree-2",
            "tree.insert",
            json!({
                "node_type": "Function",
                "name": "run_route",
                "parent_id": root_id,
                "file_path": "router/server.py",
                "start_line": 10,
                "end_line": 20,
                "signature": "def run_route(query: str)"
            }),
        ));
        let function_id = function.response.result.unwrap()["node_id"]
            .as_u64()
            .unwrap();
        let find = service.handle_line(&request(
            "tree-3",
            "tree.find",
            json!({"name": "run_route", "limit": 10}),
        ));
        assert_eq!(
            find.response.result.unwrap()["nodes"][0]["node_id"],
            function_id
        );
        let remove = service.handle_line(&request(
            "tree-4",
            "tree.remove",
            json!({"node_id": function_id}),
        ));
        assert_eq!(remove.response.result.unwrap()["removed"], true);
    }

    #[test]
    fn structured_payload_integrity_and_search_limits_are_strict() {
        let mut service = service();
        let key = "user:default:preference:language";
        let invalid = service.handle_line(&request(
            "record-1",
            "memory.store",
            json!({
                "layer": "ltm",
                "key": key,
                "payload": {
                    "type": "preference",
                    "data": {
                        "key": "user:default:preference:different",
                        "value": "Rust",
                        "memory_type": "preference",
                        "source_text": "I prefer Rust",
                        "created_at": 1,
                        "updated_at": 1,
                        "confidence": 1.0,
                        "tags": ["rust"],
                        "source": "user",
                        "layer": "ltm",
                        "reinforcement_score": 0.0,
                        "deleted": false
                    }
                }
            }),
        ));
        assert!(!invalid.response.ok);
        assert_eq!(invalid.response.error.unwrap().code, "invalid_params");
        assert!(service.kernel.ltm.is_empty());

        let invalid_limit = service.handle_line(&request(
            "search-1",
            "memory.search",
            json!({"layer": "ltm", "query": "language", "limit": 0}),
        ));
        assert_eq!(invalid_limit.response.error.unwrap().code, "invalid_params");
    }

    #[test]
    fn dirty_ltm_is_saved_only_after_the_debounce_interval() {
        let path = std::env::temp_dir().join(format!(
            "tst-debounce-{}-{}.snapshot",
            std::process::id(),
            Kernel::now().0
        ));
        let mut service = ProtocolService::from_parts(
            Kernel::new(),
            PersistenceHandler::new(&path),
            RecoveryReport {
                source: RecoverySource::Empty,
                warning: None,
                preserved_corrupt_path: None,
            },
            Duration::from_millis(20),
        );
        let stored = service.handle_line(&request(
            "store-debounce",
            "memory.store",
            json!({
                "layer": "ltm",
                "key": "user:default:preference:editor",
                "payload": {
                    "type": "preference",
                    "data": {"key": "editor", "value": "Vim", "weight": 1.0}
                }
            }),
        ));
        assert!(stored.response.ok);
        assert!(service.dirty);
        assert!(!path.exists());

        std::thread::sleep(Duration::from_millis(30));
        service.maintenance_tick().unwrap();
        assert!(!service.dirty);
        assert!(path.exists());

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(format!("{}.previous", path.display()));
    }

    #[test]
    fn tree_query_reports_budget_truncation_and_rejects_bad_paths() {
        let mut service = service();
        let bad_path = service.handle_line(&request(
            "bad-path",
            "tree.insert",
            json!({"node_type": "File", "name": "bad.py", "file_path": "../bad.py"}),
        ));
        assert_eq!(bad_path.response.error.unwrap().code, "invalid_params");

        let root = service.handle_line(&request(
            "budget-root",
            "tree.insert",
            json!({"node_type": "Project", "name": "demo"}),
        ));
        let root_id = root.response.result.unwrap()["node_id"].as_u64().unwrap();
        let child = service.handle_line(&request(
            "budget-child",
            "tree.insert",
            json!({"node_type": "Function", "name": "child", "parent_id": root_id}),
        ));
        assert!(child.response.ok);
        let query = service.handle_line(&request(
            "budget-query",
            "tree.query",
            json!({"node_id": root_id, "depth": 1, "max_nodes": 1, "token_budget": 1}),
        ));
        let result = query.response.result.unwrap();
        assert_eq!(result["nodes"].as_array().unwrap().len(), 1);
        assert_eq!(result["truncated"], true);
    }
}
