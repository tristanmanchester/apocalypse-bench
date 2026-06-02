use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::Context;
use serde::Serialize;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, Schema, Value, STORED, TEXT};
use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument};

use crate::corpus::{self, ChunkRecord, CorpusManifest};
use crate::service::SearchHit;

pub struct WikiIndex {
    reader: IndexReader,
    query_parser: QueryParser,
    schema: WikiSchema,
}

#[derive(Clone)]
struct WikiSchema {
    chunk_id: Field,
    article_id: Field,
    title: Field,
    heading_path: Field,
    body: Field,
    snippet: Field,
    url: Field,
}

#[derive(Debug, Serialize)]
pub struct OptimizeReport {
    pub segments_before: usize,
    pub segments_after: usize,
}

impl WikiIndex {
    pub fn open(index_root: &Path) -> anyhow::Result<Self> {
        let index = Index::open_in_dir(corpus::tantivy_path(index_root))?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;
        let schema = fields(&index.schema())?;
        let query_parser = query_parser(&index, &schema);
        Ok(Self {
            reader,
            query_parser,
            schema,
        })
    }

    pub fn search(&self, query: &str, limit: usize) -> anyhow::Result<Vec<SearchHit>> {
        let searcher = self.reader.searcher();
        let parsed = self.query_parser.parse_query(query)?;
        let top_docs = searcher.search(&parsed, &TopDocs::with_limit(limit.max(1)))?;
        self.hits_from_docs(top_docs)
    }

    pub fn literal_candidates(&self, query: &str, limit: usize) -> anyhow::Result<Vec<SearchHit>> {
        let searcher = self.reader.searcher();
        let parsed = match self.query_parser.parse_query(&quoted_phrase(query)) {
            Ok(parsed) => parsed,
            Err(_) => return Ok(Vec::new()),
        };
        let top_docs = searcher.search(&parsed, &TopDocs::with_limit(limit.max(1)))?;
        self.hits_from_docs(top_docs)
    }

    fn hits_from_docs(
        &self,
        top_docs: Vec<(f32, tantivy::DocAddress)>,
    ) -> anyhow::Result<Vec<SearchHit>> {
        let searcher = self.reader.searcher();
        let mut hits = Vec::with_capacity(top_docs.len());
        for (score, address) in top_docs {
            let doc: TantivyDocument = searcher.doc(address)?;
            hits.push(SearchHit {
                mode: "bm25".to_string(),
                score: Some(score),
                article_id: text(&doc, self.schema.article_id),
                chunk_id: text(&doc, self.schema.chunk_id),
                title: text(&doc, self.schema.title),
                heading_path: parse_heading_path(&text(&doc, self.schema.heading_path)),
                url: text(&doc, self.schema.url),
                snippet: text(&doc, self.schema.snippet),
                sources: vec!["bm25".to_string()],
            });
        }
        Ok(hits)
    }
}

pub fn optimize_index(index_root: &Path) -> anyhow::Result<OptimizeReport> {
    let index = Index::open_in_dir(corpus::tantivy_path(index_root))?;
    let segment_ids = index.searchable_segment_ids()?;
    let segments_before = segment_ids.len();
    if segments_before <= 1 {
        return Ok(OptimizeReport {
            segments_before,
            segments_after: segments_before,
        });
    }

    let mut writer: IndexWriter<TantivyDocument> = index.writer(500_000_000)?;
    writer.merge(&segment_ids).wait()?;
    Ok(OptimizeReport {
        segments_before,
        segments_after: index.searchable_segment_ids()?.len(),
    })
}

fn query_parser(index: &Index, schema: &WikiSchema) -> QueryParser {
    let mut parser =
        QueryParser::for_index(index, vec![schema.title, schema.heading_path, schema.body]);
    parser.set_field_boost(schema.title, 3.0);
    parser.set_field_boost(schema.heading_path, 2.0);
    parser
}

fn quoted_phrase(query: &str) -> String {
    let escaped = query.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{escaped}\"")
}

pub fn build_index(index_root: &Path, _manifest: &CorpusManifest) -> anyhow::Result<()> {
    let index_dir = corpus::tantivy_path(index_root);
    if index_dir.exists() {
        std::fs::remove_dir_all(&index_dir)?;
    }
    std::fs::create_dir_all(&index_dir)?;

    let mut schema_builder = Schema::builder();
    let chunk_id = schema_builder.add_text_field("chunk_id", STORED);
    let article_id = schema_builder.add_text_field("article_id", STORED);
    let title = schema_builder.add_text_field("title", TEXT | STORED);
    let heading_path = schema_builder.add_text_field("heading_path", TEXT | STORED);
    let body = schema_builder.add_text_field("body", TEXT);
    let snippet = schema_builder.add_text_field("snippet", STORED);
    let url = schema_builder.add_text_field("url", STORED);
    let schema = schema_builder.build();

    let index = Index::create_in_dir(&index_dir, schema)?;
    let mut writer = index.writer(100_000_000)?;
    let offsets_file = File::create(corpus::chunk_offsets_path(index_root))?;
    let mut offsets = BufWriter::new(offsets_file);
    corpus::for_each_chunk(index_root, |line| {
        writeln!(offsets, "{}\t{}", line.chunk.chunk_id, line.offset)?;
        add_chunk(
            &mut writer,
            &line.chunk,
            Fields {
                chunk_id,
                article_id,
                title,
                heading_path,
                body,
                snippet,
                url,
            },
        )
    })?;
    offsets.flush()?;
    writer.commit()?;
    Ok(())
}

struct Fields {
    chunk_id: Field,
    article_id: Field,
    title: Field,
    heading_path: Field,
    body: Field,
    snippet: Field,
    url: Field,
}

fn add_chunk(
    writer: &mut tantivy::IndexWriter,
    chunk: &ChunkRecord,
    fields: Fields,
) -> anyhow::Result<()> {
    writer.add_document(doc!(
        fields.chunk_id => chunk.chunk_id.clone(),
        fields.article_id => chunk.article_id.clone(),
        fields.title => chunk.title.clone(),
        fields.heading_path => chunk.heading_path.join(" > "),
        fields.body => chunk.text.clone(),
        fields.snippet => snippet(&chunk.text),
        fields.url => chunk.url.clone(),
    ))?;
    Ok(())
}

fn fields(schema: &Schema) -> anyhow::Result<WikiSchema> {
    Ok(WikiSchema {
        chunk_id: schema
            .get_field("chunk_id")
            .context("missing chunk_id field")?,
        article_id: schema
            .get_field("article_id")
            .context("missing article_id field")?,
        title: schema.get_field("title").context("missing title field")?,
        heading_path: schema
            .get_field("heading_path")
            .context("missing heading_path field")?,
        body: schema.get_field("body").context("missing body field")?,
        snippet: schema
            .get_field("snippet")
            .context("missing snippet field")?,
        url: schema.get_field("url").context("missing url field")?,
    })
}

fn text(doc: &TantivyDocument, field: Field) -> String {
    doc.get_first(field)
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string()
}

fn parse_heading_path(raw: &str) -> Vec<String> {
    raw.split(" > ")
        .filter(|part| !part.trim().is_empty())
        .map(|part| part.to_string())
        .collect()
}

fn snippet(text: &str) -> String {
    const LIMIT: usize = 240;
    if text.chars().count() <= LIMIT {
        return text.to_string();
    }
    let mut s: String = text.chars().take(LIMIT).collect();
    s.push_str("...");
    s
}
