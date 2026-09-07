import argparse
from lib.llm_stuff import describe_image_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Describe Image CLI")
    parser.add_argument("--image", required=True, help="path to an image file")
    parser.add_argument("--query", required=True, help="text query to rewrite based on the image")
    args = parser.parse_args()

    describe_image_command(args.query, args.image)

if __name__ == "__main__":
    main()