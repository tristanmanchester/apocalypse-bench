use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context};
use regex::Regex;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceArticle {
    pub id: String,
    pub url: String,
    pub title: String,
    pub abstract_text: Option<String>,
    pub date_created: Option<String>,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkRecord {
    pub article_id: String,
    pub chunk_id: String,
    pub url: String,
    pub title: String,
    pub abstract_text: Option<String>,
    pub date_created: Option<String>,
    pub heading_path: Vec<String>,
    pub chunk_kind: ChunkKind,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChunkKind {
    Lead,
    Section,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusManifest {
    pub manifest_version: u32,
    pub corpus_id: String,
    pub source_path: String,
    pub article_count: usize,
    pub chunk_count: usize,
    pub chunker: String,
    pub bm25_index_dir: String,
}

#[derive(Debug, Deserialize)]
struct RawArticle {
    id: serde_json::Value,
    url: Option<String>,
    title: Option<String>,
    #[serde(rename = "abstract")]
    abstract_text: Option<String>,
    date_created: Option<String>,
    text: Option<String>,
}

pub fn ingest_jsonl(input: &Path, out_dir: &Path) -> anyhow::Result<CorpusManifest> {
    fs::create_dir_all(out_dir)?;
    let chunks_path = chunks_path(out_dir);
    let source = File::open(input)?;
    let reader = BufReader::new(source);
    let mut writer = BufWriter::new(File::create(&chunks_path)?);
    let mut article_count = 0usize;
    let mut chunk_count = 0usize;
    let mut hash = Sha256::new();

    for (line_no, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        hash.update(line.as_bytes());
        let raw: RawArticle = serde_json::from_str(&line)
            .with_context(|| format!("invalid JSONL at line {}", line_no + 1))?;
        let article = normalize_article(raw)
            .with_context(|| format!("invalid article at line {}", line_no + 1))?;
        article_count += 1;
        for chunk in chunk_article(&article)? {
            serde_json::to_writer(&mut writer, &chunk)?;
            writer.write_all(b"\n")?;
            chunk_count += 1;
        }
    }
    writer.flush()?;

    if article_count == 0 {
        bail!("no articles found in {}", input.display());
    }

    let corpus_id = format!("{:x}", hash.finalize());
    let manifest = CorpusManifest {
        manifest_version: 1,
        corpus_id,
        source_path: input.display().to_string(),
        article_count,
        chunk_count,
        chunker: "markdown-heading-v1".to_string(),
        bm25_index_dir: "tantivy".to_string(),
    };
    fs::write(
        manifest_path(out_dir),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    Ok(manifest)
}

pub fn load_manifest(index_root: &Path) -> anyhow::Result<CorpusManifest> {
    let raw = fs::read_to_string(manifest_path(index_root))?;
    Ok(serde_json::from_str(&raw)?)
}

pub fn load_chunks(index_root: &Path) -> anyhow::Result<Vec<ChunkRecord>> {
    let file = File::open(chunks_path(index_root))?;
    let reader = BufReader::new(file);
    let mut chunks = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        chunks.push(serde_json::from_str(&line)?);
    }
    Ok(chunks)
}

pub struct ChunkLine {
    pub offset: u64,
    pub chunk: ChunkRecord,
}

pub fn for_each_chunk(
    index_root: &Path,
    mut f: impl FnMut(ChunkLine) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    let file = File::open(chunks_path(index_root))?;
    let mut reader = BufReader::new(file);
    let mut offset = 0u64;
    let mut line = String::new();
    loop {
        line.clear();
        let bytes_read = reader.read_line(&mut line)?;
        if bytes_read == 0 {
            break;
        }
        if !line.trim().is_empty() {
            let chunk = serde_json::from_str(&line)?;
            f(ChunkLine { offset, chunk })?;
        }
        offset += bytes_read as u64;
    }
    Ok(())
}

pub fn read_chunk_at(index_root: &Path, offset: u64) -> anyhow::Result<ChunkRecord> {
    let mut file = File::open(chunks_path(index_root))?;
    file.seek(SeekFrom::Start(offset))?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    Ok(serde_json::from_str(&line)?)
}

pub fn manifest_path(index_root: &Path) -> PathBuf {
    index_root.join("manifest.json")
}

pub fn chunks_path(index_root: &Path) -> PathBuf {
    index_root.join("chunks.jsonl")
}

pub fn chunk_offsets_path(index_root: &Path) -> PathBuf {
    index_root.join("chunk_offsets.tsv")
}

pub fn tantivy_path(index_root: &Path) -> PathBuf {
    index_root.join("tantivy")
}

fn normalize_article(raw: RawArticle) -> anyhow::Result<SourceArticle> {
    let id = match raw.id {
        serde_json::Value::String(s) => s,
        serde_json::Value::Number(n) => n.to_string(),
        _ => bail!("article id must be string or number"),
    };
    let title = raw.title.context("missing title")?;
    let text = raw.text.context("missing text")?;
    Ok(SourceArticle {
        id: stable_slug(&id),
        url: raw.url.unwrap_or_default(),
        title,
        abstract_text: raw.abstract_text,
        date_created: raw.date_created,
        text,
    })
}

fn chunk_article(article: &SourceArticle) -> anyhow::Result<Vec<ChunkRecord>> {
    let heading_re = Regex::new(r"(?m)^(#{2,6})\s+(.+?)\s*$")?;
    let mut headings: Vec<(usize, usize, String)> = heading_re
        .captures_iter(&article.text)
        .filter_map(|caps| {
            let mat = caps.get(0)?;
            let level = caps.get(1)?.as_str().len();
            let heading = caps.get(2)?.as_str().trim().to_string();
            Some((mat.start(), level, heading))
        })
        .collect();

    let mut chunks = Vec::new();
    let lead_end = headings
        .first()
        .map(|(start, _, _)| *start)
        .unwrap_or(article.text.len());
    let lead = article.text[..lead_end].trim();
    if !lead.is_empty() {
        chunks.push(make_chunk(
            article,
            "lead",
            Vec::new(),
            ChunkKind::Lead,
            lead,
        ));
    }

    headings.push((article.text.len(), 0, String::new()));
    let mut stack: Vec<(usize, String)> = Vec::new();
    for pair in headings.windows(2) {
        let (start, level, heading) = &pair[0];
        let next_start = pair[1].0;
        while stack
            .last()
            .is_some_and(|(existing_level, _)| existing_level >= level)
        {
            stack.pop();
        }
        stack.push((*level, heading.clone()));
        let section = article.text[*start..next_start].trim();
        if section.is_empty() {
            continue;
        }
        let heading_path: Vec<String> = stack.iter().map(|(_, h)| h.clone()).collect();
        let slug = stable_slug(&heading_path.join("-"));
        chunks.push(make_chunk(
            article,
            &slug,
            heading_path,
            ChunkKind::Section,
            section,
        ));
    }

    Ok(chunks)
}

fn make_chunk(
    article: &SourceArticle,
    suffix: &str,
    heading_path: Vec<String>,
    chunk_kind: ChunkKind,
    text: &str,
) -> ChunkRecord {
    ChunkRecord {
        article_id: article.id.clone(),
        chunk_id: format!("{}:{}", article.id, suffix),
        url: article.url.clone(),
        title: article.title.clone(),
        abstract_text: article.abstract_text.clone(),
        date_created: article.date_created.clone(),
        heading_path,
        chunk_kind,
        text: text.to_string(),
    }
}

fn stable_slug(input: &str) -> String {
    let mut out = String::new();
    for ch in input.to_ascii_lowercase().chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    out.trim_matches('-').to_string()
}
