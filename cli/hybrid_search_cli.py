import argparse
from lib.hybrid_search import normalize_command, weighted_search_command, rrf_search_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparser.add_parser("normalize", help="Normalize a list of floating point numbers.")
    normalize_parser.add_argument("numbers", type=float, nargs="*", help="Numbers to normalize")

    weighted_search_parser = subparser.add_parser(
        "weighted-search", 
        help="Perform a search using a hybrid of keyword and semantic searches using an 'alpha' factor that weighs keyword searches (⍺ = 1) over semantic (⍺ = 0).")
    weighted_search_parser.add_argument(
        "query",
        type=str,
        help="The query search string"
    )
    weighted_search_parser.add_argument(
        "--alpha",
        type=float,
        nargs="?",
        default=0.5,
        help="The ⍺ weight value between 0 and 1"
    )
    weighted_search_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=5,
        help="The maximum number of results to display"
    )

    rrf_search_parser = subparser.add_parser(
        "rrf-search", 
        help="Perform a search using a Reciprocal Rank Fusion hybrid ranking of keyword and semantic searches.")
    rrf_search_parser.add_argument(
        "query",
        type=str,
        help="The query search string"
    )
    rrf_search_parser.add_argument(
        "-k",
        type=float,
        nargs="?",
        default=60.0,
        help="The k weight value, higher gives lower ranked items more weight"
    )
    rrf_search_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=5,
        help="The maximum number of results to display"
    )
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method"
    )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual","batch","cross_encoder"],
        help="Query enhancement method"
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_command(args.numbers)
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case "rrf-search":
            print(f"Query: {args.query}")
            print(f"k is {args.k}")
            print(f"limit is {args.limit}")
            rrf_search_command(args.query, args.k, args.limit, args.enhance, args.rerank_method)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()