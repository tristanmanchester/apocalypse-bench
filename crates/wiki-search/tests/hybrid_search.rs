use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use axum::routing::post;
use axum::{Json, Router};
use serde_json::{json, Value};

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_wiki-search"))
}

#[tokio::test]
async fn hybrid_search_fuses_bm25_and_dense_hits() {
    let tmp = ingest("hybrid");
    let corpus_id = manifest_id(&tmp);
    let dense_manifest = tmp.join("dense_manifest.json");
    std::fs::write(
        &dense_manifest,
        serde_json::to_string_pretty(&json!({
            "corpus_id": corpus_id,
            "model": "Snowflake/snowflake-arctic-embed-s",
            "dimension": 384,
            "collection": "wiki_dense",
            "query_prefix": "Represent this sentence for searching relevant passages: ",
            "normalized": true,
            "point_count": 2,
            "signpost_rules": ["all_article_leads", "fixture_practical_sections"]
        }))
        .unwrap(),
    )
    .unwrap();

    let embed_addr = spawn_embed_server().await;
    let qdrant_addr = spawn_qdrant_server().await;
    let service_addr = unused_addr();
    let mut service = spawn_wiki_service(
        &tmp,
        service_addr,
        format!("http://{embed_addr}/embed"),
        format!("http://{qdrant_addr}"),
        dense_manifest,
    );

    wait_for_health(service_addr).await;

    let client = reqwest::Client::new();
    let response: Value = client
        .post(format!("http://{service_addr}/hybrid_search"))
        .json(&json!({"query": "water purification boiling pathogens", "limit": 3}))
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap()
        .json()
        .await
        .unwrap();

    service.kill().ok();

    assert_eq!(response["mode"], "hybrid");
    let hits = response["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 3);
    assert!(hits.iter().any(|hit| {
        hit["chunk_id"] == "water-purification:lead"
            && hit["sources"].as_array().unwrap().contains(&json!("bm25"))
            && hit["sources"].as_array().unwrap().contains(&json!("dense"))
    }));
    assert!(hits.iter().any(|hit| hit["article_id"] == "hypothermia"
        && hit["sources"].as_array().unwrap().contains(&json!("dense"))));
}

#[tokio::test]
async fn hybrid_search_reports_dense_unavailable_without_degrading_to_bm25() {
    let tmp = ingest("hybrid-unavailable");
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
    let response = client
        .post(format!("http://{service_addr}/hybrid_search"))
        .json(&json!({"query": "water purification", "limit": 3}))
        .send()
        .await
        .unwrap();
    let status = response.status();
    let body: Value = response.json().await.unwrap();

    service.kill().ok();

    assert_eq!(status, reqwest::StatusCode::SERVICE_UNAVAILABLE);
    assert!(body["error"]
        .as_str()
        .unwrap()
        .contains("dense search is unavailable"));
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

fn manifest_id(index_root: &Path) -> String {
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(index_root.join("manifest.json")).unwrap())
            .unwrap();
    manifest["corpus_id"].as_str().unwrap().to_string()
}

fn spawn_wiki_service(
    index_root: &Path,
    service_addr: SocketAddr,
    embed_url: String,
    qdrant_url: String,
    dense_manifest: PathBuf,
) -> Child {
    Command::new(bin())
        .arg("serve")
        .arg("--index")
        .arg(index_root)
        .arg("--listen")
        .arg(service_addr.to_string())
        .env("WIKI_EMBED_URL", embed_url)
        .env("WIKI_QDRANT_URL", qdrant_url)
        .env("WIKI_QDRANT_COLLECTION", "wiki_dense")
        .env("WIKI_DENSE_MANIFEST", dense_manifest)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn wiki service")
}

async fn spawn_embed_server() -> SocketAddr {
    let app = Router::new().route(
        "/embed",
        post(|| async { Json(json!({ "embedding": vec![0.0_f32; 384] })) }),
    );
    spawn_axum(app).await
}

async fn spawn_qdrant_server() -> SocketAddr {
    let app = Router::new().route(
        "/collections/wiki_dense/points/search",
        post(|| async {
            Json(json!({
                "result": [
                    {
                        "score": 0.99,
                        "payload": {
                            "article_id": "water-purification",
                            "chunk_id": "water-purification:lead"
                        }
                    },
                    {
                        "score": 0.88,
                        "payload": {
                            "article_id": "hypothermia",
                            "chunk_id": "hypothermia:treatment"
                        }
                    }
                ]
            }))
        }),
    );
    spawn_axum(app).await
}

async fn spawn_axum(app: Router) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    addr
}

fn unused_addr() -> SocketAddr {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    addr
}

async fn wait_for_health(addr: SocketAddr) {
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
