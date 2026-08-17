import argparse
import json

def search_movies_from_json_file(json_file: str, search_for: str, max_results: int = 5) -> list[str]:
    result: list[str] = []
    with open(json_file, "r") as file:
        try:
            movie_data = json.load(file)
        except json.JSONDecodeError, UnicodeDecodeError:
            print(f"Error decoding JSON file.")
    result_count = 0
    for movie in movie_data["movies"]:
        if result_count >= max_results:
            break
        if movie.get("title") and search_for in movie["title"]:
            result.append(movie["title"])
            result_count += 1
    return result
            

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            search_results = search_movies_from_json_file("./data/movies.json", args.query)
            for index, title in enumerate(search_results):
                print(f"{index+1}. {title}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()