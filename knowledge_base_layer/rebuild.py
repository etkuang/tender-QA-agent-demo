# coding: utf-8
# @Author: Wang Qingkang

import argparse
import json
import uuid

from knowledge_base_layer.config import settings
from knowledge_base_layer.embeddings import LocalEmbeddingGateway
from knowledge_base_layer.ingestion.pdf import PolicyPdfLoader, load_pdf_sources
from knowledge_base_layer.ingestion.pipeline import PolicyIngestionPipeline
from knowledge_base_layer.retrieval.milvus import MilvusStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild the policy Milvus index from raw PDFs.")
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--publish", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = load_pdf_sources(settings.pdf_manifest)
    loader = PolicyPdfLoader(settings)
    chunks = []
    for source in sources:
        chunks.extend(loader.load(source, settings.pdf_manifest.parent, args.data_version))
    pipeline = PolicyIngestionPipeline(
        MilvusStore(settings),
        LocalEmbeddingGateway(settings.embedding_model_name_or_path),
    )
    manifest = pipeline.ingest(
        chunks,
        settings.policy_collection_alias,
        args.data_version,
        uuid.uuid4().hex,
        publish=args.publish,
    )
    print(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2))

    failed = (
        any(issue.critical for issue in manifest.issues)
        or manifest.written_count != manifest.source_count
        or (args.publish and not manifest.published)
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()