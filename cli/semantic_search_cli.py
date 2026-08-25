import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, search_command, chunk_command, semantic_chunk_command
from constants import QUERY_RESULT_DEFAULT_LIMIT, CHUNK_DEFAULT_SIZE, OVERLAP_DEFAULT, SEMANTIC_CHUNK_DEFAULT_SIZE, SEMANTIC_OVERLAP_DEFAULT


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    _ = subparsers.add_parser("verify", help="Verify model")

    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Generate embedding for given text")
    embed_text_parser.add_argument(
        "text", type=str, help="Text for which to generate the embedding")

    _ = subparsers.add_parser(
        "verify_embeddings", help="Verify loaded or built embeddings")

    embed_query_parser = subparsers.add_parser(
        "embed_query", help="Generate embedding for given query")
    embed_query_parser.add_argument(
        "query", type=str, help="query for which to generate the embedding")

    search_parser = subparsers.add_parser("search", help="Semantic search")
    search_parser.add_argument(
        "query", type=str, help="String for which to search")
    search_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=QUERY_RESULT_DEFAULT_LIMIT,
        help="Limit number of results to this"
    )

    chunk_parser = subparsers.add_parser(
        "chunk", help="Chunk the given text into nearly fixed sized chunks.")
    chunk_parser.add_argument(
        "text", type=str, help="Text to break into chunks")
    chunk_parser.add_argument(
        "--chunk-size",
        type=int,
        nargs="?",
        default=CHUNK_DEFAULT_SIZE,
        help="Limit number of results to this"
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        nargs="?",
        default=OVERLAP_DEFAULT,
        help="Words to overlap per chunk"
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Chunk the given text into nearly fixed sized chunks of sentences.")
    semantic_chunk_parser.add_argument(
        "text", type=str, help="Text to break into chunks")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        nargs="?",
        default=SEMANTIC_CHUNK_DEFAULT_SIZE,
        help="Limit number of results to this"
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        nargs="?",
        default=SEMANTIC_OVERLAP_DEFAULT,
        help="Words to overlap per chunk"
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            search_command(args.query, args.limit)
        case "chunk":
            chunk_command(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk_command(
                args.text, args.max_chunk_size, args.overlap)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
