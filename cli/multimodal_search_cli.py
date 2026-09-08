import argparse
from constants import DEBUG
from lib.multimodal_search import verify_image_embedding, image_search_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = \
        subparser.add_parser("verify_image_embedding", help="Verify image embedding")
    verify_image_embedding_parser.add_argument("image", type=str, help="The path to the image file")

    mmodal_search_parser = \
        subparser.add_parser("image_search", help="Movie search based on supplied image")
    mmodal_search_parser.add_argument("image", type=str, help="The path to the image file")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image)
        case "image_search":
            image_search_command(args.image)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
