use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context;
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tower_http::trace::TraceLayer;

use crate::bm25::WikiIndex;
use crate::corpus::{self, ChunkRecord, CorpusManifest};

#[derive(Clone)]
pub struct WikiState {
    manifest: CorpusManifest,
    bm25: Arc<WikiIndex>,
    chunks: Arc<ChunkStore>,
    dense: Option<DenseConfig>,
    http: reqwest::Client,
}

struct ChunkStore {
    index_root: PathBuf,
    offsets: HashMap<String, u64>,
    article_offsets: HashMap<String, Vec<u64>>,
}

#[derive(Clone)]
struct DenseConfig {
    qdrant_url: String,
    qdrant_collection: String,
    embed_url: String,
    manifest: DenseManifest,
}

#[derive(Clone, Debug, Deserialize)]
struct DenseManifest {
    corpus_id: String,
    model: String,
    dimension: usize,
    collection: String,
    query_prefix: String,
    normalized: bool,
    embedding_precision: Option<String>,
    max_seq_length: Option<usize>,
    quantization: Option<String>,
    point_count: usize,
    signpost_rules: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub ok: bool,
    pub corpus: ManifestHealth,
    pub index: ManifestHealth,
    pub capabilities: Vec<String>,
    pub article_count: usize,
    pub chunk_count: usize,
    pub indexes: IndexHealth,
}

#[derive(Debug, Serialize)]
pub struct ManifestHealth {
    #[serde(rename = "manifestId")]
    pub manifest_id: String,
}

#[derive(Debug, Serialize)]
pub struct IndexHealth {
    pub bm25: IndexStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dense: Option<IndexStatus>,
}

#[derive(Debug, Serialize)]
pub struct IndexStatus {
    #[serde(rename = "manifestId")]
    pub manifest_id: String,
    pub status: String,
}

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(alias = "topK")]
    pub limit: Option<usize>,
    #[serde(alias = "chunkId")]
    pub chunk_id: Option<String>,
    #[serde(alias = "articleId")]
    pub article_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub mode: String,
    pub query: String,
    #[serde(rename = "latencyMs", skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<u128>,
    pub hits: Vec<SearchHit>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchHit {
    pub mode: String,
    pub score: Option<f32>,
    pub article_id: String,
    pub chunk_id: String,
    pub title: String,
    pub heading_path: Vec<String>,
    pub url: String,
    pub snippet: String,
    pub sources: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct EmbedResponse {
    embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct EmbedRequest<'a> {
    query: &'a str,
}

#[derive(Debug, Deserialize)]
struct QdrantSearchResponse {
    result: Vec<QdrantPoint>,
}

#[derive(Debug, Deserialize)]
struct QdrantPoint {
    score: Option<f32>,
    payload: Option<QdrantPayload>,
}

#[derive(Debug, Deserialize)]
struct QdrantPayload {
    article_id: Option<String>,
    chunk_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct QdrantSearchRequest<'a> {
    vector: &'a [f32],
    limit: usize,
    with_payload: bool,
}

#[derive(Debug, Deserialize)]
pub struct ReadRequest {
    #[serde(alias = "chunkId")]
    pub chunk_id: Option<String>,
    #[serde(alias = "articleId")]
    pub article_id: Option<String>,
    #[serde(alias = "maxChars")]
    pub max_chars: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct ReadResponse {
    pub article_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_id: Option<String>,
    pub title: String,
    pub heading_path: Vec<String>,
    pub url: String,
    pub text: String,
    pub truncated: bool,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

pub async fn serve(index_root: PathBuf, listen: SocketAddr) -> anyhow::Result<()> {
    let state = WikiState::open(index_root)?;
    let app = Router::new()
        .route("/health", get(health))
        .route("/search", post(search_bm25))
        .route("/literal_search", post(search_literal))
        .route("/semantic_search", post(search_dense))
        .route("/hybrid_search", post(search_hybrid))
        .route("/search/bm25", post(search_bm25))
        .route("/search/literal", post(search_literal))
        .route("/search/dense", post(search_dense))
        .route("/search/hybrid", post(search_hybrid))
        .route("/read", post(read))
        .layer(TraceLayer::new_for_http())
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(listen).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

impl WikiState {
    pub fn open(index_root: PathBuf) -> anyhow::Result<Self> {
        let manifest = corpus::load_manifest(&index_root)?;
        let bm25 = Arc::new(WikiIndex::open(&index_root)?);
        let chunks = Arc::new(ChunkStore::open(index_root.clone())?);
        let dense = DenseConfig::from_env(&manifest)?;
        Ok(Self {
            manifest,
            bm25,
            chunks,
            dense,
            http: reqwest::Client::new(),
        })
    }

    pub fn health(&self) -> HealthResponse {
        let manifest_id = self.manifest.corpus_id.clone();
        HealthResponse {
            ok: true,
            corpus: ManifestHealth {
                manifest_id: manifest_id.clone(),
            },
            index: ManifestHealth {
                manifest_id: manifest_id.clone(),
            },
            article_count: self.manifest.article_count,
            chunk_count: self.manifest.chunk_count,
            capabilities: self.capabilities(),
            indexes: IndexHealth {
                bm25: IndexStatus {
                    manifest_id: manifest_id.clone(),
                    status: "ready".to_string(),
                },
                dense: self.dense.as_ref().map(|dense| IndexStatus {
                    manifest_id: dense.manifest.corpus_id.clone(),
                    status: "ready".to_string(),
                }),
            },
        }
    }

    pub fn search_bm25(&self, query: &str, limit: usize) -> anyhow::Result<SearchResponse> {
        let started = Instant::now();
        let hits = self.bm25.search(query, limit)?;
        Ok(SearchResponse {
            mode: "bm25".to_string(),
            query: query.to_string(),
            latency_ms: Some(started.elapsed().as_millis()),
            hits,
        })
    }

    pub fn search_literal(
        &self,
        query: &str,
        limit: usize,
        chunk_id: Option<&str>,
        article_id: Option<&str>,
    ) -> anyhow::Result<SearchResponse> {
        let started = Instant::now();
        let hits = match (chunk_id, article_id) {
            (Some(_), Some(_)) => anyhow::bail!("provide either chunkId or articleId, not both"),
            (Some(chunk_id), None) => self.search_literal_chunk(query, chunk_id, limit)?,
            (None, Some(article_id)) => self.search_literal_article(query, article_id, limit)?,
            (None, None) => self.search_literal_indexed(query, limit)?,
        };
        Ok(SearchResponse {
            mode: "literal".to_string(),
            query: query.to_string(),
            latency_ms: Some(started.elapsed().as_millis()),
            hits,
        })
    }

    fn search_literal_chunk(
        &self,
        query: &str,
        chunk_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<SearchHit>> {
        let chunk = self.chunks.get(chunk_id)?;
        if chunk.text.contains(query) || chunk.title.contains(query) {
            Ok(vec![literal_hit(chunk, query)])
        } else {
            Ok(Vec::with_capacity(limit.min(1)))
        }
    }

    fn search_literal_article(
        &self,
        query: &str,
        article_id: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<SearchHit>> {
        let wanted = limit.max(1);
        let mut hits = Vec::new();
        for chunk in self.chunks.article_chunks(article_id)? {
            if chunk.text.contains(query) || chunk.title.contains(query) {
                hits.push(literal_hit(chunk, query));
                if hits.len() >= wanted {
                    break;
                }
            }
        }
        Ok(hits)
    }

    fn search_literal_indexed(&self, query: &str, limit: usize) -> anyhow::Result<Vec<SearchHit>> {
        const CANDIDATE_MULTIPLIER: usize = 20;
        let wanted = limit.max(1);
        let mut hits = Vec::new();
        let candidate_limit = wanted.saturating_mul(CANDIDATE_MULTIPLIER).max(50);
        for candidate in self.bm25.literal_candidates(query, candidate_limit)? {
            let chunk = self.chunks.get(&candidate.chunk_id)?;
            if chunk.text.contains(query) || chunk.title.contains(query) {
                hits.push(literal_hit(chunk, query));
                if hits.len() >= wanted {
                    break;
                }
            }
        }
        Ok(hits)
    }

    pub fn read_chunk(
        &self,
        chunk_id: &str,
        max_chars: Option<usize>,
    ) -> anyhow::Result<ReadResponse> {
        let chunk = self.chunks.get(chunk_id)?;
        Ok(read_response(chunk, max_chars))
    }

    pub fn read_article(
        &self,
        article_id: &str,
        max_chars: Option<usize>,
    ) -> anyhow::Result<ReadResponse> {
        self.chunks
            .read_article(article_id, max_chars.unwrap_or(4_000))
    }

    fn hit_for_chunk(
        &self,
        chunk_id: &str,
        score: Option<f32>,
        mode: &str,
        sources: Vec<String>,
    ) -> anyhow::Result<SearchHit> {
        let chunk = self.chunks.get(chunk_id)?;
        Ok(SearchHit {
            mode: mode.to_string(),
            score,
            article_id: chunk.article_id,
            chunk_id: chunk.chunk_id,
            title: chunk.title,
            heading_path: chunk.heading_path,
            url: chunk.url,
            snippet: bounded(&chunk.text, 240).0,
            sources,
        })
    }

    pub async fn search_dense(&self, query: &str, limit: usize) -> anyhow::Result<SearchResponse> {
        let started = Instant::now();
        let dense = self.dense.as_ref().context(
            "dense search is unavailable until Qdrant and Arctic embeddings are configured",
        )?;
        let embedding = self.embed_query(dense, query).await?;
        let qdrant_url = format!(
            "{}/collections/{}/points/search",
            dense.qdrant_url.trim_end_matches('/'),
            dense.qdrant_collection
        );
        let response: QdrantSearchResponse = self
            .http
            .post(qdrant_url)
            .json(&QdrantSearchRequest {
                vector: &embedding,
                limit: limit.max(1),
                with_payload: true,
            })
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        let hits = response
            .result
            .into_iter()
            .filter_map(|point| {
                let payload = point.payload?;
                let chunk_id = payload.chunk_id?;
                self.hit_for_chunk(&chunk_id, point.score, "dense", vec!["dense".to_string()])
                    .ok()
                    .map(|mut hit| {
                        if let Some(article_id) = payload.article_id {
                            hit.article_id = article_id;
                        }
                        hit
                    })
            })
            .collect();
        Ok(SearchResponse {
            mode: "dense".to_string(),
            query: query.to_string(),
            latency_ms: Some(started.elapsed().as_millis()),
            hits,
        })
    }

    pub async fn search_hybrid(&self, query: &str, limit: usize) -> anyhow::Result<SearchResponse> {
        let started = Instant::now();
        let bm25 = self.bm25.search(query, limit.max(10))?;
        let dense = self.search_dense(query, limit.max(10)).await?.hits;
        Ok(SearchResponse {
            mode: "hybrid".to_string(),
            query: query.to_string(),
            latency_ms: Some(started.elapsed().as_millis()),
            hits: reciprocal_rank_fusion(vec![bm25, dense], limit.max(1)),
        })
    }

    fn capabilities(&self) -> Vec<String> {
        let mut values = vec!["bm25".to_string(), "literal".to_string()];
        if self.dense.is_some() {
            values.push("dense".to_string());
            values.push("hybrid".to_string());
        }
        values
    }

    async fn embed_query(&self, dense: &DenseConfig, query: &str) -> anyhow::Result<Vec<f32>> {
        let response: EmbedResponse = self
            .http
            .post(&dense.embed_url)
            .json(&EmbedRequest { query })
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        if response.embedding.len() != 384 {
            anyhow::bail!(
                "embedding service returned {} dimensions, expected 384 for Snowflake Arctic Embed S",
                response.embedding.len()
            );
        }
        Ok(response.embedding)
    }
}

impl ChunkStore {
    fn open(index_root: PathBuf) -> anyhow::Result<Self> {
        let file = File::open(corpus::chunk_offsets_path(&index_root))?;
        let reader = BufReader::new(file);
        let mut offsets = HashMap::new();
        let mut article_offsets: HashMap<String, Vec<u64>> = HashMap::new();
        for line in reader.lines() {
            let line = line?;
            let Some((chunk_id, offset)) = line.split_once('\t') else {
                continue;
            };
            let offset = offset.parse()?;
            if let Some((article_id, _)) = chunk_id.split_once(':') {
                article_offsets
                    .entry(article_id.to_string())
                    .or_default()
                    .push(offset);
            }
            offsets.insert(chunk_id.to_string(), offset);
        }
        Ok(Self {
            index_root,
            offsets,
            article_offsets,
        })
    }

    fn get(&self, chunk_id: &str) -> anyhow::Result<ChunkRecord> {
        let offset = self
            .offsets
            .get(chunk_id)
            .with_context(|| format!("unknown chunk id: {chunk_id}"))?;
        corpus::read_chunk_at(&self.index_root, *offset)
    }

    fn read_article(&self, article_id: &str, max_chars: usize) -> anyhow::Result<ReadResponse> {
        let article_chunks = self.article_chunks(article_id)?;
        let mut text_chunks = Vec::new();
        let mut chars = 0usize;
        let mut truncated = false;

        for chunk in &article_chunks {
            let heading_chars = if chunk.heading_path.is_empty() {
                0
            } else {
                chunk.heading_path.join(" > ").chars().count() + 2
            };
            let next_chars = chars + heading_chars + chunk.text.chars().count() + 2;
            if next_chars > max_chars {
                let remaining = max_chars.saturating_sub(chars);
                if remaining > 0 {
                    text_chunks.push(article_chunk_text(chunk, remaining));
                }
                truncated = true;
                break;
            }
            chars = next_chars;
            text_chunks.push(article_chunk_text(chunk, max_chars));
        }

        let first = article_chunks
            .into_iter()
            .next()
            .with_context(|| format!("unknown article id: {article_id}"))?;
        Ok(ReadResponse {
            article_id: first.article_id,
            chunk_id: None,
            title: first.title,
            heading_path: Vec::new(),
            url: first.url,
            text: text_chunks.join("\n\n"),
            truncated,
        })
    }

    fn article_chunks(&self, article_id: &str) -> anyhow::Result<Vec<ChunkRecord>> {
        let offsets = self
            .article_offsets
            .get(article_id)
            .with_context(|| format!("unknown article id: {article_id}"))?;
        offsets
            .iter()
            .map(|offset| corpus::read_chunk_at(&self.index_root, *offset))
            .collect()
    }
}

fn read_response(chunk: ChunkRecord, max_chars: Option<usize>) -> ReadResponse {
    let (text, truncated) = bounded(&chunk.text, max_chars.unwrap_or(4_000));
    ReadResponse {
        article_id: chunk.article_id,
        chunk_id: Some(chunk.chunk_id),
        title: chunk.title,
        heading_path: chunk.heading_path,
        url: chunk.url,
        text,
        truncated,
    }
}

async fn health(State(state): State<WikiState>) -> Json<HealthResponse> {
    Json(state.health())
}

async fn search_bm25(
    State(state): State<WikiState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<ErrorResponse>)> {
    state
        .search_bm25(&req.query, req.limit.unwrap_or(10))
        .map(Json)
        .map_err(internal_error)
}

async fn search_literal(
    State(state): State<WikiState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<ErrorResponse>)> {
    state
        .search_literal(
            &req.query,
            req.limit.unwrap_or(10),
            req.chunk_id.as_deref(),
            req.article_id.as_deref(),
        )
        .map(Json)
        .map_err(internal_error)
}

async fn search_dense(
    State(state): State<WikiState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<ErrorResponse>)> {
    state
        .search_dense(&req.query, req.limit.unwrap_or(10))
        .await
        .map(Json)
        .map_err(service_error)
}

async fn search_hybrid(
    State(state): State<WikiState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, (StatusCode, Json<ErrorResponse>)> {
    state
        .search_hybrid(&req.query, req.limit.unwrap_or(10))
        .await
        .map(Json)
        .map_err(service_error)
}

async fn read(
    State(state): State<WikiState>,
    Json(req): Json<ReadRequest>,
) -> Result<Json<ReadResponse>, (StatusCode, Json<ErrorResponse>)> {
    state.read_request(req).map(Json).map_err(|err| {
        let text = err.to_string();
        let status = if text.contains("unknown chunk id") || text.contains("unknown article id") {
            StatusCode::NOT_FOUND
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        };
        (status, Json(ErrorResponse { error: text }))
    })
}

impl WikiState {
    fn read_request(&self, req: ReadRequest) -> anyhow::Result<ReadResponse> {
        match (req.chunk_id, req.article_id) {
            (Some(chunk_id), None) => self.read_chunk(&chunk_id, req.max_chars),
            (None, Some(article_id)) => self.read_article(&article_id, req.max_chars),
            (Some(_), Some(_)) => anyhow::bail!("provide either chunkId or articleId, not both"),
            (None, None) => anyhow::bail!("provide chunkId or articleId"),
        }
    }
}

fn internal_error(err: anyhow::Error) -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse {
            error: err.to_string(),
        }),
    )
}

fn service_error(err: anyhow::Error) -> (StatusCode, Json<ErrorResponse>) {
    let text = err.to_string();
    let status = if text.contains("dense search is unavailable") {
        StatusCode::SERVICE_UNAVAILABLE
    } else {
        StatusCode::INTERNAL_SERVER_ERROR
    };
    (status, Json(ErrorResponse { error: text }))
}

fn bounded(text: &str, max_chars: usize) -> (String, bool) {
    if text.chars().count() <= max_chars {
        return (text.to_string(), false);
    }
    if max_chars == 0 {
        return (String::new(), true);
    }
    let ellipsis = "...";
    let ellipsis_chars = ellipsis.chars().count();
    if max_chars <= ellipsis_chars {
        return (".".repeat(max_chars), true);
    }
    let mut value = text
        .chars()
        .take(max_chars - ellipsis_chars)
        .collect::<String>();
    value.push_str("...");
    (value, true)
}

fn article_chunk_text(chunk: &ChunkRecord, max_chars: usize) -> String {
    let heading = if chunk.heading_path.is_empty() {
        String::new()
    } else {
        format!("{}\n", chunk.heading_path.join(" > "))
    };
    bounded(&format!("{heading}{}", chunk.text), max_chars).0
}

fn literal_hit(chunk: ChunkRecord, query: &str) -> SearchHit {
    SearchHit {
        mode: "literal".to_string(),
        score: None,
        article_id: chunk.article_id,
        chunk_id: chunk.chunk_id,
        title: chunk.title,
        heading_path: chunk.heading_path,
        url: chunk.url,
        snippet: literal_snippet(&chunk.text, query, 240),
        sources: vec!["literal".to_string()],
    }
}

fn literal_snippet(text: &str, query: &str, max_chars: usize) -> String {
    let Some(byte_pos) = text.find(query) else {
        return bounded(text, max_chars).0;
    };
    let prefix_budget = max_chars.saturating_sub(query.chars().count()) / 2;
    let start = text[..byte_pos]
        .char_indices()
        .rev()
        .nth(prefix_budget)
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    let end_budget = max_chars.saturating_sub(text[start..byte_pos].chars().count());
    let end = text[byte_pos..]
        .char_indices()
        .nth(end_budget)
        .map(|(idx, _)| byte_pos + idx)
        .unwrap_or(text.len());
    let mut snippet = String::new();
    if start > 0 {
        snippet.push_str("...");
    }
    snippet.push_str(text[start..end].trim());
    if end < text.len() {
        snippet.push_str("...");
    }
    snippet
}

const HYBRID_MAX_HITS_PER_ARTICLE: usize = 2;

fn reciprocal_rank_fusion(result_sets: Vec<Vec<SearchHit>>, limit: usize) -> Vec<SearchHit> {
    let mut by_chunk: HashMap<String, (SearchHit, f32)> = HashMap::new();
    for hits in result_sets {
        for (rank, hit) in hits.into_iter().enumerate() {
            let fused = 1.0 / (60.0 + rank as f32 + 1.0);
            by_chunk
                .entry(hit.chunk_id.clone())
                .and_modify(|(existing, score)| {
                    *score += fused;
                    for source in &hit.sources {
                        if !existing.sources.contains(source) {
                            existing.sources.push(source.clone());
                        }
                    }
                    existing.mode = "hybrid".to_string();
                    existing.score = Some(*score);
                })
                .or_insert_with(|| {
                    let mut merged = hit;
                    merged.mode = "hybrid".to_string();
                    merged.score = Some(fused);
                    (merged, fused)
                });
        }
    }
    let mut hits = by_chunk.into_values().collect::<Vec<_>>();
    hits.sort_by(|(_, a), (_, b)| b.total_cmp(a));
    diversify_articles(hits, limit)
}

fn diversify_articles(ranked_hits: Vec<(SearchHit, f32)>, limit: usize) -> Vec<SearchHit> {
    let mut selected = Vec::new();
    let mut deferred = Vec::new();
    let mut per_article: HashMap<String, usize> = HashMap::new();

    for (hit, score) in ranked_hits {
        let count = per_article.get(&hit.article_id).copied().unwrap_or(0);
        if count < HYBRID_MAX_HITS_PER_ARTICLE {
            per_article.insert(hit.article_id.clone(), count + 1);
            selected.push(hit);
            if selected.len() == limit {
                return selected;
            }
        } else {
            deferred.push((hit, score));
        }
    }

    for (hit, _) in deferred {
        selected.push(hit);
        if selected.len() == limit {
            break;
        }
    }

    selected
}

impl DenseConfig {
    fn from_env(corpus_manifest: &CorpusManifest) -> anyhow::Result<Option<Self>> {
        let qdrant_url = env::var("WIKI_QDRANT_URL").ok();
        let qdrant_collection = env::var("WIKI_QDRANT_COLLECTION").ok();
        let embed_url = env::var("WIKI_EMBED_URL").ok();
        let manifest_path = env::var("WIKI_DENSE_MANIFEST").ok();
        if qdrant_url.is_none()
            && qdrant_collection.is_none()
            && embed_url.is_none()
            && manifest_path.is_none()
        {
            return Ok(None);
        }

        let qdrant_url = qdrant_url.context("missing WIKI_QDRANT_URL for dense search")?;
        let qdrant_collection =
            qdrant_collection.context("missing WIKI_QDRANT_COLLECTION for dense search")?;
        let embed_url = embed_url.context("missing WIKI_EMBED_URL for dense search")?;
        let manifest_path =
            manifest_path.context("missing WIKI_DENSE_MANIFEST for dense search")?;
        let manifest = DenseManifest::load(Path::new(&manifest_path))?;
        manifest.validate(corpus_manifest, &qdrant_collection)?;

        Ok(Some(Self {
            qdrant_url,
            qdrant_collection,
            embed_url,
            manifest,
        }))
    }
}

impl DenseManifest {
    fn load(path: &Path) -> anyhow::Result<Self> {
        let raw =
            std::fs::read_to_string(path).with_context(|| format!("loading {}", path.display()))?;
        Ok(serde_json::from_str(&raw)?)
    }

    fn validate(
        &self,
        corpus_manifest: &CorpusManifest,
        qdrant_collection: &str,
    ) -> anyhow::Result<()> {
        if self.corpus_id != corpus_manifest.corpus_id {
            anyhow::bail!(
                "dense manifest corpus mismatch: expected {}, got {}",
                corpus_manifest.corpus_id,
                self.corpus_id
            );
        }
        if self.model != "Snowflake/snowflake-arctic-embed-s" {
            anyhow::bail!(
                "dense manifest model mismatch: expected Snowflake/snowflake-arctic-embed-s, got {}",
                self.model
            );
        }
        if self.dimension != 384 {
            anyhow::bail!(
                "dense manifest dimension mismatch: expected 384, got {}",
                self.dimension
            );
        }
        if self.collection != qdrant_collection {
            anyhow::bail!(
                "dense manifest collection mismatch: expected {}, got {}",
                qdrant_collection,
                self.collection
            );
        }
        if self.query_prefix != "Represent this sentence for searching relevant passages: " {
            anyhow::bail!("dense manifest query prefix mismatch");
        }
        if !self.normalized {
            anyhow::bail!("dense manifest must record normalized=true");
        }
        if let Some(precision) = &self.embedding_precision {
            if precision != "float16" && precision != "float32" {
                anyhow::bail!("dense manifest embedding precision mismatch: got {precision}");
            }
        }
        if let Some(max_seq_length) = self.max_seq_length {
            if max_seq_length == 0 || max_seq_length > 512 {
                anyhow::bail!("dense manifest max_seq_length mismatch: got {max_seq_length}");
            }
        }
        if let Some(quantization) = &self.quantization {
            if quantization != "turbo-bits4" && quantization != "scalar-int8" {
                anyhow::bail!("dense manifest quantization mismatch: got {quantization}");
            }
        }
        if self.point_count == 0 {
            anyhow::bail!("dense manifest contains zero points");
        }
        if !self
            .signpost_rules
            .iter()
            .any(|rule| rule == "all_article_leads")
        {
            anyhow::bail!("dense manifest must include all_article_leads signpost rule");
        }
        Ok(())
    }
}

#[allow(dead_code)]
fn _assert_path_is_send_sync(_: &Path) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_manifest_rejects_wrong_model() {
        let corpus = CorpusManifest {
            manifest_version: 1,
            corpus_id: "corpus-1".to_string(),
            source_path: "wiki.jsonl".to_string(),
            article_count: 1,
            chunk_count: 1,
            chunker: "markdown-heading-v1".to_string(),
            bm25_index_dir: "tantivy".to_string(),
        };
        let manifest = DenseManifest {
            corpus_id: "corpus-1".to_string(),
            model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            dimension: 384,
            collection: "wiki_dense".to_string(),
            query_prefix: "Represent this sentence for searching relevant passages: ".to_string(),
            normalized: true,
            embedding_precision: Some("float16".to_string()),
            max_seq_length: Some(256),
            quantization: Some("turbo-bits4".to_string()),
            point_count: 10,
            signpost_rules: vec!["all_article_leads".to_string()],
        };

        let err = manifest.validate(&corpus, "wiki_dense").unwrap_err();
        assert!(err.to_string().contains("dense manifest model mismatch"));
    }

    #[test]
    fn hybrid_fusion_dedupes_sources_and_diversifies_articles() {
        let hits = reciprocal_rank_fusion(
            vec![
                vec![
                    hit("a1", "a1:c1", "bm25"),
                    hit("a1", "a1:c2", "bm25"),
                    hit("a1", "a1:c3", "bm25"),
                    hit("a2", "a2:c1", "bm25"),
                ],
                vec![hit("a1", "a1:c1", "dense"), hit("a3", "a3:c1", "dense")],
            ],
            4,
        );

        assert_eq!(hits[0].chunk_id, "a1:c1");
        assert!(hits[0].sources.contains(&"bm25".to_string()));
        assert!(hits[0].sources.contains(&"dense".to_string()));
        let a1_count = hits.iter().filter(|hit| hit.article_id == "a1").count();
        assert_eq!(a1_count, HYBRID_MAX_HITS_PER_ARTICLE);
        assert!(hits.iter().any(|hit| hit.article_id == "a2"));
        assert!(hits.iter().any(|hit| hit.article_id == "a3"));
    }

    fn hit(article_id: &str, chunk_id: &str, source: &str) -> SearchHit {
        SearchHit {
            mode: source.to_string(),
            score: Some(1.0),
            article_id: article_id.to_string(),
            chunk_id: chunk_id.to_string(),
            title: article_id.to_string(),
            heading_path: Vec::new(),
            url: String::new(),
            snippet: String::new(),
            sources: vec![source.to_string()],
        }
    }
}
