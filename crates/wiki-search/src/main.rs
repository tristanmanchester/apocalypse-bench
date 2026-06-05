mod bm25;
mod corpus;
mod service;

use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::Context;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "wiki-search")]
#[command(about = "Local Wikipedia search service for apocalypse-bench")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Ingest {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    Index {
        #[arg(long)]
        index: PathBuf,
    },
    Optimize {
        #[arg(long)]
        index: PathBuf,
    },
    Search {
        #[arg(long)]
        index: PathBuf,
        #[arg(long)]
        query: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },
    Read {
        #[arg(long)]
        index: PathBuf,
        #[arg(long)]
        chunk_id: Option<String>,
        #[arg(long)]
        article_id: Option<String>,
    },
    Serve {
        #[arg(long)]
        index: PathBuf,
        #[arg(long, default_value = "127.0.0.1:8765")]
        listen: SocketAddr,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Ingest { input, out } => {
            let manifest = corpus::ingest_jsonl(&input, &out)
                .with_context(|| format!("ingesting {}", input.display()))?;
            bm25::build_index(&out, &manifest)?;
            println!("{}", serde_json::to_string_pretty(&manifest)?);
        }
        Command::Index { index } => {
            let manifest = corpus::load_manifest(&index)
                .with_context(|| format!("loading manifest from {}", index.display()))?;
            bm25::build_index(&index, &manifest)?;
            println!("{}", serde_json::to_string_pretty(&manifest)?);
        }
        Command::Optimize { index } => {
            let report = bm25::optimize_index(&index)
                .with_context(|| format!("optimizing Tantivy index at {}", index.display()))?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Search {
            index,
            query,
            limit,
        } => {
            let state = service::WikiState::open(index)?;
            let response = state.search_bm25(&query, limit)?;
            println!("{}", serde_json::to_string_pretty(&response)?);
        }
        Command::Read {
            index,
            chunk_id,
            article_id,
        } => {
            let state = service::WikiState::open(index)?;
            let response = if let Some(chunk_id) = chunk_id {
                state.read_chunk(&chunk_id, None)?
            } else if let Some(article_id) = article_id {
                state.read_article(&article_id, None)?
            } else {
                anyhow::bail!("provide --chunk-id or --article-id");
            };
            println!("{}", serde_json::to_string_pretty(&response)?);
        }
        Command::Serve { index, listen } => {
            service::serve(index, listen).await?;
        }
    }
    Ok(())
}
