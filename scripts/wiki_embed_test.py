import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import wiki_embed


class WikiEmbedTest(unittest.TestCase):
    def test_document_text_orders_title_abstract_and_body(self) -> None:
        text = wiki_embed.document_text(
            {
                "title": "Portable water purification",
                "abstract_text": "Portable methods for making water safer.",
                "text": "Boiling can inactivate many pathogens.",
            }
        )

        self.assertEqual(
            text,
            "Portable water purification\n\n"
            "Portable methods for making water safer.\n\n"
            "Boiling can inactivate many pathogens.",
        )

    def test_signpost_selection_includes_leads_and_practical_sections(self) -> None:
        self.assertTrue(wiki_embed.is_signpost({"chunk_kind": "lead"}, set()))
        self.assertTrue(
            wiki_embed.is_signpost(
                {
                    "chunk_kind": "section",
                    "title": "Portable water purification",
                    "heading_path": ["Techniques", "Disinfection"],
                    "text": "Chemical treatment can reduce pathogens.",
                },
                set(),
            )
        )
        self.assertFalse(
            wiki_embed.is_signpost(
                {
                    "chunk_kind": "section",
                    "title": "Poetry",
                    "heading_path": ["Reception"],
                    "text": "Literary criticism and publication history.",
                },
                set(),
            )
        )
        self.assertFalse(
            wiki_embed.is_signpost(
                {
                    "chunk_kind": "section",
                    "title": "Poetry",
                    "heading_path": ["Reception"],
                    "text": "This passage mentions water and fire only as metaphors.",
                },
                set(),
            )
        )

    def test_question_bank_phrases_add_targeted_non_practical_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            question_file = Path(temp_dir) / "AGR.jsonl"
            question_file.write_text(
                json.dumps(
                    {
                        "title": "Mushroom substrate",
                        "prompt": "How should improvised mycelium substrate be handled?",
                        "scenario": [],
                        "reference_facts": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            terms = wiki_embed.load_question_terms([question_file])

        self.assertIn("mycelium substrate", terms)
        self.assertTrue(
            wiki_embed.is_signpost(
                {
                    "chunk_kind": "section",
                    "title": "Fungi",
                    "heading_path": ["Mycelium substrate"],
                    "text": "Colonization notes.",
                },
                terms,
            )
        )
        self.assertFalse(
            wiki_embed.is_signpost(
                {
                    "chunk_kind": "section",
                    "title": "Fungi",
                    "heading_path": ["Growth"],
                    "text": "Mycelium can colonize organic substrate.",
                },
                terms,
            )
        )

    def test_iter_signposts_dedupes_and_respects_limit(self) -> None:
        rows = [
            {
                "article_id": "1",
                "chunk_id": "1:lead",
                "chunk_kind": "lead",
                "title": "Water",
                "heading_path": [],
                "text": "Water.",
            },
            {
                "article_id": "1",
                "chunk_id": "1:lead",
                "chunk_kind": "lead",
                "title": "Water",
                "heading_path": [],
                "text": "Duplicate.",
            },
            {
                "article_id": "2",
                "chunk_id": "2:history",
                "chunk_kind": "section",
                "title": "Poetry",
                "heading_path": ["History"],
                "text": "Publication history.",
            },
            {
                "article_id": "3",
                "chunk_id": "3:burns",
                "chunk_kind": "section",
                "title": "Burn",
                "heading_path": ["First aid"],
                "text": "Cool the burn and avoid infection.",
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            chunks_path = Path(temp_dir) / "chunks.jsonl"
            chunks_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            signposts = list(wiki_embed.iter_signposts(chunks_path, 2, set()))

        self.assertEqual([row["chunk_id"] for row in signposts], ["1:lead", "3:burns"])

    def test_dense_manifest_records_arctic_contract(self) -> None:
        manifest = wiki_embed.build_dense_manifest(
            corpus_manifest={"corpus_id": "corpus-123"},
            collection="wikipedia_arctic_s",
            source_chunks=Path("data/wiki/full/chunks.jsonl"),
            question_bank_paths=["data/question_bank"],
            point_count=42,
        )

        self.assertEqual(manifest["corpus_id"], "corpus-123")
        self.assertEqual(manifest["model"], "Snowflake/snowflake-arctic-embed-s")
        self.assertEqual(manifest["dimension"], 384)
        self.assertEqual(manifest["collection"], "wikipedia_arctic_s")
        self.assertEqual(
            manifest["query_prefix"],
            "Represent this sentence for searching relevant passages: ",
        )
        self.assertTrue(manifest["normalized"])
        self.assertEqual(manifest["embedding_precision"], "float32")
        self.assertEqual(manifest["max_seq_length"], 512)
        self.assertEqual(manifest["quantization"], "turbo-bits4")
        self.assertTrue(manifest["vectors_on_disk"])
        self.assertEqual(manifest["point_count"], 42)
        self.assertEqual(
            manifest["signpost_rules"],
            [
                "all_article_leads",
                "targeted_practical_sections",
                "benchmark_question_preselection",
            ],
        )


if __name__ == "__main__":
    unittest.main()
