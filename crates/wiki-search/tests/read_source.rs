use std::path::PathBuf;
use std::process::Command;

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
    assert_eq!(json["title"], "Hypothermia");
    assert!(json["text"].as_str().unwrap().contains("warm gradually"));
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
