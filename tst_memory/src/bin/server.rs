use std::io::{self, BufRead, Write};
use tst_memory::kernel::Kernel;
use tst_memory::api::ApiServer;

fn main() {
    let mut kernel = Kernel::new();
    let mut api = ApiServer::new(&mut kernel);
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    println!("READY");
    stdout.flush().unwrap();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if trimmed.starts_with("READ ") {
            let json = &trimmed[5..];
            match api.handle_read(json) {
                Ok(resp) => { println!("{}", resp); }
                Err(e) => { println!(r#"{{"error": "{}"}}"#, e); }
            }
        } else if trimmed.starts_with("WRITE ") {
            let json = &trimmed[6..];
            match api.handle_write(json) {
                Ok(resp) => { println!("{}", resp); }
                Err(e) => { println!(r#"{{"error": "{}"}}"#, e); }
            }
        } else if trimmed.starts_with("TREE_INSERT ") {
            let json = &trimmed[12..];
            match api.handle_tree_insert(json) {
                Ok(resp) => { println!("{}", resp); }
                Err(e) => { println!(r#"{{"error": "{}"}}"#, e); }
            }
        } else if trimmed.starts_with("TREE_QUERY ") {
            let json = &trimmed[11..];
            match api.handle_tree_query(json) {
                Ok(resp) => { println!("{}", resp); }
                Err(e) => { println!(r#"{{"error": "{}"}}"#, e); }
            }
        } else if trimmed.starts_with("TREE_LINK ") {
            let json = &trimmed[10..];
            match api.handle_tree_link(json) {
                Ok(resp) => { println!("{}", resp); }
                Err(e) => { println!(r#"{{"error": "{}"}}"#, e); }
            }
        } else if trimmed == "TREE_CLEAR" {
            match api.handle_tree_clear() {
                Ok(resp) => { println!("{}", resp); }
                Err(e) => { println!(r#"{{"error": "{}"}}"#, e); }
            }
        } else {
            println!(r#"{{"error": "Unknown command"}}"#);
        }
        stdout.flush().unwrap();
    }
}
