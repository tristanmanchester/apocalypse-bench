use std::path::PathBuf;
use std::process::Command;

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_wiki-search"))
}

#[test]
fn bm25_search_returns_stable_chunk_pointer() {
    let tmp = ingest("bm25");
    let output = Command::new(bin())
        .arg("search")
        .arg("--index")
        .arg(&tmp)
        .arg("--query")
        .arg("boiling disinfect water")
        .arg("--limit")
        .arg("3")
        .output()
        .expect("run search");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let json: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let hits = json["hits"].as_array().unwrap();
    assert!(!hits.is_empty());
    assert_eq!(hits[0]["title"], "Water purification");
    assert!(hits[0]["chunk_id"]
        .as_str()
        .unwrap()
        .starts_with("water-purification:"));
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
