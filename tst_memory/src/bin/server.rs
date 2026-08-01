use std::env;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread;
use std::time::Duration;

use tst_memory::api::ApiServer;
use tst_memory::kernel::STMConfig;
use tst_memory::protocol::ProtocolService;

struct ServerConfig {
    snapshot_path: PathBuf,
    save_debounce: Duration,
    legacy_protocol: bool,
    stm_config: STMConfig,
}

impl ServerConfig {
    fn from_env_and_args() -> Result<Self, String> {
        let mut snapshot_path = env::var_os("TST_SNAPSHOT_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(".tst/ltm.snapshot"));
        let mut debounce_ms = env::var("TST_SNAPSHOT_DEBOUNCE_MS")
            .ok()
            .map(|value| {
                value.parse::<u64>().map_err(|_| {
                    "TST_SNAPSHOT_DEBOUNCE_MS must be a non-negative integer".to_string()
                })
            })
            .transpose()?
            .unwrap_or(1_000);
        let mut legacy_protocol = env::var("TST_ENABLE_LEGACY_PROTOCOL")
            .ok()
            .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "yes"));
        let mut stm_config = STMConfig::default();
        stm_config.capacity = env_value("TST_STM_CAPACITY", stm_config.capacity)?;
        stm_config.half_life_seconds =
            env_value("TST_STM_HALF_LIFE_SECONDS", stm_config.half_life_seconds)?;
        stm_config.promotion_threshold = env_value(
            "TST_STM_PROMOTION_THRESHOLD",
            stm_config.promotion_threshold,
        )?;
        stm_config.read_reinforcement =
            env_value("TST_STM_READ_REINFORCEMENT", stm_config.read_reinforcement)?;
        stm_config.write_reinforcement = env_value(
            "TST_STM_WRITE_REINFORCEMENT",
            stm_config.write_reinforcement,
        )?;
        stm_config.expiry_score = env_value("TST_STM_EXPIRY_SCORE", stm_config.expiry_score)?;

        let mut arguments = env::args().skip(1);
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--snapshot" => {
                    snapshot_path = arguments
                        .next()
                        .map(PathBuf::from)
                        .ok_or_else(|| "--snapshot requires a path".to_string())?;
                }
                "--save-debounce-ms" => {
                    debounce_ms = arguments
                        .next()
                        .ok_or_else(|| "--save-debounce-ms requires an integer".to_string())?
                        .parse::<u64>()
                        .map_err(|_| {
                            "--save-debounce-ms requires a non-negative integer".to_string()
                        })?;
                }
                "--legacy-protocol" => legacy_protocol = true,
                "--help" | "-h" => {
                    println!(
                        "tst-memory server\n\n  --snapshot PATH\n  --save-debounce-ms N\n  --legacy-protocol"
                    );
                    std::process::exit(0);
                }
                unknown => return Err(format!("unknown server argument: {unknown}")),
            }
        }

        stm_config.validate().map_err(|error| error.to_string())?;
        Ok(Self {
            snapshot_path,
            save_debounce: Duration::from_millis(debounce_ms),
            legacy_protocol,
            stm_config,
        })
    }
}

fn env_value<T>(name: &str, default: T) -> Result<T, String>
where
    T: std::str::FromStr,
{
    env::var(name)
        .ok()
        .map(|value| {
            value
                .parse::<T>()
                .map_err(|_| format!("{name} has an invalid value"))
        })
        .transpose()
        .map(|value| value.unwrap_or(default))
}

enum InputMessage {
    Line(String),
    Error(String),
}

fn main() -> ExitCode {
    let config = match ServerConfig::from_env_and_args() {
        Ok(config) => config,
        Err(error) => {
            eprintln!("configuration error: {error}");
            return ExitCode::from(2);
        }
    };
    let mut service = match ProtocolService::recover_with_config(
        &config.snapshot_path,
        config.save_debounce,
        config.stm_config,
    ) {
        Ok(service) => service,
        Err(error) => {
            eprintln!("snapshot recovery failed: {error}");
            return ExitCode::FAILURE;
        }
    };

    let mut stdout = io::stdout();
    println!("READY");
    if let Err(error) = stdout.flush() {
        eprintln!("failed to flush READY: {error}");
        return ExitCode::FAILURE;
    }

    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            match line {
                Ok(line) => {
                    if sender.send(InputMessage::Line(line)).is_err() {
                        return;
                    }
                }
                Err(error) => {
                    let _ = sender.send(InputMessage::Error(error.to_string()));
                    return;
                }
            }
        }
    });

    loop {
        let line = match receiver.recv_timeout(Duration::from_millis(50)) {
            Ok(InputMessage::Line(line)) => line,
            Ok(InputMessage::Error(error)) => {
                eprintln!("stdin read failed: {error}");
                break;
            }
            Err(RecvTimeoutError::Timeout) => {
                if let Err(error) = service.maintenance_tick() {
                    eprintln!("background maintenance failed: {error}");
                    return ExitCode::FAILURE;
                }
                continue;
            }
            Err(RecvTimeoutError::Disconnected) => break,
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if trimmed.starts_with('{') {
            let outcome = service.handle_line(trimmed);
            if let Some(error) = &outcome.response.error {
                let request_id = serde_json::to_string(&outcome.response.request_id)
                    .unwrap_or_else(|_| "\"invalid\"".to_string());
                eprintln!("protocol error request_id={request_id} code={}", error.code);
            }
            match serde_json::to_string(&outcome.response) {
                Ok(response) => println!("{response}"),
                Err(error) => {
                    eprintln!("response serialization failed: {error}");
                    return ExitCode::FAILURE;
                }
            }
            if let Err(error) = stdout.flush() {
                eprintln!("stdout flush failed: {error}");
                return ExitCode::FAILURE;
            }
            if outcome.shutdown {
                return ExitCode::SUCCESS;
            }
            continue;
        }

        if config.legacy_protocol {
            let response = handle_legacy_line(&mut service, trimmed);
            println!("{response}");
        } else {
            let escaped = serde_json::json!({
                "protocol_version": 1,
                "request_id": "unknown",
                "ok": false,
                "result": null,
                "error": {
                    "code": "invalid_request",
                    "message": "expected a versioned JSON request envelope",
                    "details": null
                },
                "metrics": {"kernel_ms": 0.0}
            });
            println!("{escaped}");
        }
        if let Err(error) = stdout.flush() {
            eprintln!("stdout flush failed: {error}");
            return ExitCode::FAILURE;
        }
    }

    if let Err(error) = service.flush() {
        eprintln!("snapshot flush on EOF failed: {error}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn handle_legacy_line(service: &mut ProtocolService, line: &str) -> String {
    let mut persistent_mutation = false;
    let result = {
        let mut api = ApiServer::new(&mut service.kernel);
        if let Some(json) = line.strip_prefix("READ ") {
            api.handle_read(json)
        } else if let Some(json) = line.strip_prefix("WRITE ") {
            persistent_mutation = true;
            api.handle_write(json)
        } else if let Some(json) = line.strip_prefix("TREE_INSERT ") {
            api.handle_tree_insert(json)
        } else if let Some(json) = line.strip_prefix("TREE_QUERY ") {
            api.handle_tree_query(json)
        } else if let Some(json) = line.strip_prefix("TREE_LINK ") {
            api.handle_tree_link(json)
        } else if line == "TREE_CLEAR" {
            api.handle_tree_clear()
        } else {
            Err("Unknown legacy command".to_string())
        }
    };

    if persistent_mutation
        && result.is_ok()
        && let Err(error) = service.note_external_ltm_mutation()
    {
        return serde_json::json!({"error": error.to_string()}).to_string();
    }
    result.unwrap_or_else(|error| serde_json::json!({"error": error}).to_string())
}
