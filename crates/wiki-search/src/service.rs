use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use regex::Regex;
use serde::{Deserialize, Serialize};
use tower_http::trace::TraceLayer;

use crate::bm25::WikiIndex;
use crate::corpus::{self, ChunkRecord, CorpusManifest};

#[derive(Clone)]
pub struct WikiState {
    index_root: PathBuf,
    manifest: CorpusManifest,
    bm25: Arc<WikiIndex>,
    chunks: Arc<ChunkStore>,
    dense: Option<DenseConfig>,
    http: reqwest::Client,
}

struct ChunkStore {
    index_root: PathBuf,
    offsets: HashMap<String, u64>,
}

#[derive(Clone)]
struct DenseConfig {
    qdrant_url: String,
    qdrant_collection: String,
    embed_url: String,
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
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub mode: String,
    pub query: String,
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
    pub chunk_id: String,
    #[serde(alias = "maxChars")]
    pub max_chars: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct ReadResponse {
    pub article_id: String,
    pub chunk_id: String,
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
        Ok(Self {
            index_root,
            manifest,
            bm25,
            chunks,
            dense: DenseConfig::from_env(),
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
                dense: self.dense.as_ref().map(|_| IndexStatus {
                    manifest_id,
                    status: "ready".to_string(),
                }),
            },
        }
    }

    pub fn search_bm25(&self, query: &str, limit: usize) -> anyhow::Result<SearchResponse> {
        Ok(SearchResponse {
            mode: "bm25".to_string(),
            query: query.to_string(),
            hits: self.bm25.search(query, limit)?,
        })
    }

    pub fn search_literal(&self, query: &str, limit: usize) -> anyhow::Result<SearchResponse> {
        let re = Regex::new(&regex::escape(query))?;
        let mut hits = Vec::new();
        corpus::for_each_chunk(&self.index_root, |line| {
            if hits.len() >= limit.max(1) {
                return Ok(());
            }
            let chunk = line.chunk;
            if re.is_match(&chunk.text) || re.is_match(&chunk.title) {
                hits.push(SearchHit {
                    mode: "literal".to_string(),
                    score: None,
                    article_id: chunk.article_id,
                    chunk_id: chunk.chunk_id,
                    title: chunk.title,
                    heading_path: chunk.heading_path,
                    url: chunk.url,
                    snippet: bounded(&chunk.text, 240).0,
                    sources: vec!["literal".to_string()],
                });
            }
            Ok(())
        })?;
        Ok(SearchResponse {
            mode: "literal".to_string(),
            query: query.to_string(),
            hits,
        })
    }

    pub fn read_chunk(
        &self,
        chunk_id: &str,
        max_chars: Option<usize>,
    ) -> anyhow::Result<ReadResponse> {
        let chunk = self.chunks.get(chunk_id)?;
        Ok(read_response(chunk, max_chars))
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
            hits,
        })
    }

    pub async fn search_hybrid(&self, query: &str, limit: usize) -> anyhow::Result<SearchResponse> {
        let bm25 = self.bm25.search(query, limit.max(10))?;
        let dense = self.search_dense(query, limit.max(10)).await?.hits;
        Ok(SearchResponse {
            mode: "hybrid".to_string(),
            query: query.to_string(),
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
        for line in reader.lines() {
            let line = line?;
            let Some((chunk_id, offset)) = line.split_once('\t') else {
                continue;
            };
            offsets.insert(chunk_id.to_string(), offset.parse()?);
        }
        Ok(Self {
            index_root,
            offsets,
        })
    }

    fn get(&self, chunk_id: &str) -> anyhow::Result<ChunkRecord> {
        let offset = self
            .offsets
            .get(chunk_id)
            .with_context(|| format!("unknown chunk id: {chunk_id}"))?;
        corpus::read_chunk_at(&self.index_root, *offset)
    }
}

fn read_response(chunk: ChunkRecord, max_chars: Option<usize>) -> ReadResponse {
    let (text, truncated) = bounded(&chunk.text, max_chars.unwrap_or(4_000));
    ReadResponse {
        article_id: chunk.article_id,
        chunk_id: chunk.chunk_id,
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
        .search_literal(&req.query, req.limit.unwrap_or(10))
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
    state
        .read_chunk(&req.chunk_id, req.max_chars)
        .map(Json)
        .map_err(|err| {
            let status = if err.to_string().contains("unknown chunk id") {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            (
                status,
                Json(ErrorResponse {
                    error: err.to_string(),
                }),
            )
        })
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
    let mut value = text.chars().take(max_chars).collect::<String>();
    value.push_str("...");
    (value, true)
}

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
    hits.into_iter().take(limit).map(|(hit, _)| hit).collect()
}

impl DenseConfig {
    fn from_env() -> Option<Self> {
        Some(Self {
            qdrant_url: env::var("WIKI_QDRANT_URL").ok()?,
            qdrant_collection: env::var("WIKI_QDRANT_COLLECTION").ok()?,
            embed_url: env::var("WIKI_EMBED_URL").ok()?,
        })
    }
}

#[allow(dead_code)]
fn _assert_path_is_send_sync(_: &Path) {}
