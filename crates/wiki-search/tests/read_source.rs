use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;

use serde_json::{json, Value};

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_wiki-search"))
}

#[test]
fn read_returns_bounded_source_text() {
    let tmp = ingest("read");
    let output = Command::new(bin())
        .arg("read")
        .arg("--index")
        .arg(&tmp)
        .arg("--chunk-id")
        .arg("hypothermia:treatment")
        .output()
        .expect("read source");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(json["article_id"], "hypothermia");
    assert_eq!(json["chunk_id"], "hypothermia:treatment");
    assert_eq!(json["title"], "Hypothermia");
    assert!(json["heading_path"].is_array());
    assert!(json["url"].as_str().unwrap().starts_with("https://"));
    assert!(json["text"].as_str().unwrap().contains("warm gradually"));
    assert_eq!(json["truncated"], false);
}

#[tokio::test]
async fn read_truncated_chunk_stays_within_max_chars() {
    let tmp = ingest("read-bounded-chunk");
    let service_addr = unused_addr();
    let mut service = Command::new(bin())
        .arg("serve")
        .arg("--index")
        .arg(&tmp)
        .arg("--listen")
        .arg(service_addr.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn wiki service");

    wait_for_health(service_addr).await;

    let json: Value = reqwest::Client::new()
        .post(format!("http://{service_addr}/read"))
        .json(&json!({
            "chunkId": "water-purification:lead",
            "maxChars": 40
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    service.kill().ok();

    let text = json["text"].as_str().unwrap();
    assert!(text.chars().count() <= 40, "{text}");
    assert!(text.ends_with("..."), "{text}");
    assert_eq!(json["truncated"], true);
}

#[test]
fn read_returns_bounded_article_text() {
    let tmp = ingest("read-article");
    let output = Command::new(bin())
        .arg("read")
        .arg("--index")
        .arg(&tmp)
        .arg("--article-id")
        .arg("water-purification")
        .output()
        .expect("read article source");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(json["article_id"], "water-purification");
    assert_eq!(json["title"], "Water purification");
    assert!(json["chunk_id"].is_null());
    assert!(json["text"].as_str().unwrap().contains("Boiling"));
    assert!(json["text"].as_str().unwrap().contains("Filtration"));
    assert_eq!(json["truncated"], false);
}

#[tokio::test]
async fn read_truncated_article_stays_within_max_chars() {
    let tmp = ingest("read-bounded-article");
    let service_addr = unused_addr();
    let mut service = Command::new(bin())
        .arg("serve")
        .arg("--index")
        .arg(&tmp)
        .arg("--listen")
        .arg(service_addr.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn wiki service");

    wait_for_health(service_addr).await;

    let json: Value = reqwest::Client::new()
        .post(format!("http://{service_addr}/read"))
        .json(&json!({
            "articleId": "water-purification",
            "maxChars": 60
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    service.kill().ok();

    let text = json["text"].as_str().unwrap();
    assert!(text.chars().count() <= 60, "{text}");
    assert!(text.ends_with("..."), "{text}");
    assert_eq!(json["truncated"], true);
}

#[tokio::test]
async fn read_unknown_chunk_returns_structured_not_found_error() {
    let tmp = ingest("read-missing");
    let service_addr = unused_addr();
    let mut service = Command::new(bin())
        .arg("serve")
        .arg("--index")
        .arg(&tmp)
        .arg("--listen")
        .arg(service_addr.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn wiki service");

    wait_for_health(service_addr).await;

    let response = reqwest::Client::new()
        .post(format!("http://{service_addr}/read"))
        .json(&json!({"chunkId": "missing"}))
        .send()
        .await
        .unwrap();
    let status = response.status();
    let body: Value = response.json().await.unwrap();

    service.kill().ok();

    assert_eq!(status, reqwest::StatusCode::NOT_FOUND);
    assert_eq!(body["error"], "unknown chunk id: missing");
}

#[tokio::test]
async fn scoped_literal_search_finds_exact_phrase_without_global_scan() {
    let tmp = ingest("literal-scoped");
    let service_addr = unused_addr();
    let mut service = Command::new(bin())
        .arg("serve")
        .arg("--index")
        .arg(&tmp)
        .arg("--listen")
        .arg(service_addr.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn wiki service");

    wait_for_health(service_addr).await;

    let client = reqwest::Client::new();
    let chunk_response: Value = client
        .post(format!("http://{service_addr}/literal_search"))
        .json(&json!({
            "query": "warm gradually",
            "chunkId": "hypothermia:treatment",
            "limit": 3
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let article_response: Value = client
        .post(format!("http://{service_addr}/literal_search"))
        .json(&json!({
            "query": "Boiling",
            "articleId": "water-purification",
            "limit": 3
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    service.kill().ok();

    assert_eq!(chunk_response["mode"], "literal");
    assert_eq!(chunk_response["hits"].as_array().unwrap().len(), 1);
    assert_eq!(
        chunk_response["hits"][0]["chunk_id"],
        "hypothermia:treatment"
    );
    assert!(chunk_response["hits"][0]["snippet"]
        .as_str()
        .unwrap()
        .contains("warm gradually"));

    assert_eq!(article_response["hits"].as_array().unwrap().len(), 1);
    assert_eq!(
        article_response["hits"][0]["article_id"],
        "water-purification"
    );
    assert!(article_response["hits"][0]["snippet"]
        .as_str()
        .unwrap()
        .contains("Boiling"));
}

fn ingest(name: &str) -> PathBuf {
    let mut tmp = std::env::temp_dir();
    tmp.push(format!("wiki-search-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let output = Command::new(bin())
        .arg("ingest")
        .arg("--input")
        .arg("tests/fixtures/wiki-mini.jsonl")
        .arg("--out")
        .arg(&tmp)
        .output()
        .expect("ingest");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    tmp
}

fn unused_addr() -> std::net::SocketAddr {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    addr
}

async fn wait_for_health(addr: std::net::SocketAddr) {
    let client = reqwest::Client::new();
    for _ in 0..60 {
        if client
            .get(format!("http://{addr}/health"))
            .send()
            .await
            .is_ok_and(|response| response.status().is_success())
        {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("wiki service did not become healthy at {addr}");
}
