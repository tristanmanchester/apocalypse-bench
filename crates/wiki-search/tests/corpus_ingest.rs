use std::path::PathBuf;
use std::process::Command;

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_wiki-search"))
}

#[test]
fn ingest_fixture_is_stable() {
    let tmp = tempfile_dir("ingest-stable");
    let input = PathBuf::from("tests/fixtures/wiki-mini.jsonl");

    let output = Command::new(bin())
        .arg("ingest")
        .arg("--input")
        .arg(&input)
        .arg("--out")
        .arg(&tmp)
        .output()
        .expect("run ingest");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let first_manifest = std::fs::read_to_string(tmp.join("manifest.json")).unwrap();
    let first_chunks = std::fs::read_to_string(tmp.join("chunks.jsonl")).unwrap();

    let output = Command::new(bin())
        .arg("ingest")
        .arg("--input")
        .arg(&input)
        .arg("--out")
        .arg(&tmp)
        .output()
        .expect("rerun ingest");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    assert_eq!(
        first_manifest,
        std::fs::read_to_string(tmp.join("manifest.json")).unwrap()
    );
    assert_eq!(
        first_chunks,
        std::fs::read_to_string(tmp.join("chunks.jsonl")).unwrap()
    );
    assert!(first_chunks.contains("water-purification:boiling"));
}

fn tempfile_dir(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("wiki-search-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).unwrap();
    path
}
