from pydantic import BaseModel
from typing import List

class Movie(BaseModel):
    id: int
    title: str
    description: str

class MoviesList(BaseModel):
    movies: List[Movie]

def load_movies(json_file_name: str) -> MoviesList:
    with open(json_file_name, "r") as json_file:
        json_data = json_file.read()
    return MoviesList.model_validate_json(json_data)


