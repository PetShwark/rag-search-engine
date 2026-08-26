from pydantic import BaseModel
from typing import List
from tqdm import tqdm

class Movie(BaseModel):
    id: int
    title: str
    description: str

class MoviesList(BaseModel):
    movies: List[Movie]

def load_movies(json_file_name: str) -> MoviesList:
    print(f"Reading '{json_file_name}' into string...")
    with open(json_file_name, "r", encoding="utf-8") as json_file:
        json_data = json_file.read()
    print(f"Validating string as MovieList object...")
    movie_list = MoviesList.model_validate_json(json_data)
    for movie in tqdm(movie_list.movies, desc="Decoding Unicode escapes"):
        movie.title = movie.title.encode("utf-8").decode("unicode_escape").strip()
        movie.description = movie.description.encode("utf-8").decode("unicode_escape").strip()
    print("Done.\n")
    return movie_list

