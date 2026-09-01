import argparse
from pathlib import Path
from json import load
from pydantic import BaseModel, ValidationError
from constants import DATA_FOLDER_PATH_NAME, GOLDEN_DATASET_FILE_NAME, MOVIE_JSON_FILE_NAME
from movies import load_movies
from lib.hybrid_search import HybridSearch


class GoldenDatasetRecord(BaseModel):
    query: str
    relevant_docs: list[str]


class GoldenDataset(BaseModel):
    test_cases: list[GoldenDatasetRecord]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
         "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    golden_dataset_path = Path(DATA_FOLDER_PATH_NAME) / GOLDEN_DATASET_FILE_NAME
    movies_dataset_path = Path(DATA_FOLDER_PATH_NAME) / MOVIE_JSON_FILE_NAME
    if golden_dataset_path.exists() and movies_dataset_path.exists():
        with open(golden_dataset_path, "r") as input_file:
            json_data_str = input_file.read()
        try:
            golden_dataset = GoldenDataset.model_validate_json(json_data_str)
        except ValidationError as e:
            print(e.json())
            return
        movies_data = load_movies(movies_dataset_path.__str__())
        hybrid_search = HybridSearch(movies_data.movies)
        test_k = 60
        print(f"k = {args.limit}\n\n")
        for test_item in golden_dataset.test_cases:
            search_results = hybrid_search.rrf_search(test_item.query, test_k, args.limit)
            # Trim relevant list down to args.limit length - into a set
            golden_test_titles = {
                title
                for i, title in enumerate(test_item.relevant_docs)
                if i < args.limit
            }
            retrieved_titles = {
                hybrid_search.idx.docmap[search_result["id"]].title
                for search_result in search_results
            }
            relevant_titles = golden_test_titles & retrieved_titles
            print(f"- Query: {test_item.query}")
            print(f"\t- Precision@{args.limit}: {(len(relevant_titles)/len(retrieved_titles)):.4f}")
            print(f"\tRetrieved: {', '.join(retrieved_titles)}")
            print(f"\tRelevant: {', '.join(relevant_titles)}")



if __name__ == "__main__":
     main()