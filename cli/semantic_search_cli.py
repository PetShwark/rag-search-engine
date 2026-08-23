import argparse
from lib.semantic_search import verify_model, embed_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_command_parser = subparsers.add_parser("verify", help="Verify model")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate embedding for given text")
    embed_text_parser.add_argument("text", type=str, help="Text for which to generate the embedding")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()