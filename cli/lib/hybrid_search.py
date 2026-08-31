from pathlib import Path
from typing import TypedDict

from movies import load_movies
from inverted_index import InvertedIndex, DocID
from .semantic_search import ChunkedSemanticSearch, ChunkedSearchResult
from movies import Movie
from .llm_stuff import llm_spell_check, llm_rewrite_query, llm_expand_query


class HybridScoreData(TypedDict):
    document: str
    keyword_score: float
    semantic_score: float
    hybrid_score: float


class HybridScoreRecord(TypedDict):
    id: int
    data: HybridScoreData


class RRFScoreData(TypedDict):
    rrf_score: float
    bm25_rank: int
    sem_rank: int
    document: str


class RRFScoreRecord(TypedDict):
    id: int
    data: RRFScoreData


class HybridSearch:
    MOVIES_JSON_FILE = Path("./data") / "movies.json"

    def __init__(self, documents: list[Movie]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not InvertedIndex.INDEX_PICKLE_PATH.exists():
            self.idx.build(ChunkedSemanticSearch.MOVIES_JSON_FILE.__str__())
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> dict[DocID, float]:
        self.idx.load()
        return self.idx.bm25search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[HybridScoreRecord]:
        bm25_results = self._bm25_search(query, limit*500)
        norm_b25_results = dict(zip(bm25_results.keys(), normalize(list(bm25_results.values()))))
        assert len(bm25_results) == len(norm_b25_results), "Something went wrong with the normalization of the bm25_results, the dictionary changed size!"
        chunk_results = self.semantic_search.search_chunked(query, limit*500)
        norm_chunk_results = normalize_semantic_chunk_results(chunk_results)
        # Gather the results from each search into common dictionary (key=movie id)
        # We can't guarantee the same movies will appear in each set of scores, so we have to loop through
        # each individually and add records to the results as we encounter movies.
        results: dict[DocID, HybridScoreData] = {}
        # First loop only _adds_ result record because we're encountering for the first time
        for doc_id, norm_score in norm_b25_results.items():
            results[doc_id] = HybridScoreData(
                document=self.idx.docmap[doc_id].description,
                keyword_score=norm_score,
                semantic_score=0.0,
                hybrid_score=0.0
            )
        for norm_chunk_result in norm_chunk_results:
            if not results.get(norm_chunk_result["id"]):
                results[norm_chunk_result["id"]] = HybridScoreData(
                    document=norm_chunk_result["document"],
                    keyword_score=0.0,
                    semantic_score=norm_chunk_result["score"],
                    hybrid_score=0.0
                )
            else:
                # All the other fields should be there since the id exists
                results[norm_chunk_result["id"]]["semantic_score"] = norm_chunk_result["score"]
        # Loop through the gathered score records and calculate the hybrid scores
        for _, score_record in results.items():
            score_record["hybrid_score"] = \
                hybrid_score(
                    score_record["keyword_score"],
                    score_record["semantic_score"],
                    alpha)
        # Sort by hybrid scores, descending
        sorted_results = dict(sorted(results.items(), key=lambda x: x[1]["hybrid_score"], reverse=True))
        # Assemble return list of limit results
        new_limit = min(limit, len(sorted_results))
        return [
            HybridScoreRecord(id=doc_id, data=data) 
            for i, (doc_id, data) in enumerate(sorted_results.items()) 
            if i < new_limit
        ]


    def rrf_search(self, query: str, k: float, limit: int = 10) -> list[RRFScoreRecord]:
        results: list[RRFScoreRecord] = []
        bm25_results = self._bm25_search(query, limit*500)
        sorted_bm25_results = dict(sorted(bm25_results.items(), key=lambda x: x[1], reverse=True))
        ranked_bm25_results = {
            doc_id: (index+1) # Rank instead of bm25 score
            for index, (doc_id, _) in enumerate(sorted_bm25_results.items()) 
        }
        sem_results = self.semantic_search.search_chunked(query, limit*500)
        sorted_sem_results = list(sorted(sem_results, key=lambda x: x["score"], reverse=True))
        ranked_sem_results = [
            ChunkedSearchResult(
                id=result["id"],
                score=index+1, # Rank instead of semantic score
                document=result["document"],
                title=result["title"],
                metadata=result["metadata"]                
            )
            for index, result in enumerate(sorted_sem_results)
        ]
        accumulator: dict[DocID, RRFScoreData] = {}
        for doc_id, rank in ranked_bm25_results.items():
            accumulator[doc_id] = RRFScoreData(
                rrf_score=0.0,
                document=self.idx.docmap[doc_id].description,
                bm25_rank=rank,
                sem_rank=len(ranked_bm25_results)+1 # really last place
            )
        for sem_result in ranked_sem_results:
            if not accumulator.get(sem_result["id"]):
                accumulator[sem_result["id"]] = RRFScoreData(
                    rrf_score=0.0,
                    document=sem_result["document"],
                    bm25_rank=len(ranked_sem_results)+1, # really last place
                    sem_rank=int(sem_result["score"]) # Called score, but is rank
                )
            else:
                accumulator[sem_result["id"]]["sem_rank"] = int(sem_result["score"])
        # Loop through all the accumulated rank data and calculate the rrf_score for each record
        for _, record in accumulator.items():
            record["rrf_score"] = \
                (1.0 / (k + float(record["bm25_rank"]))) + \
                (1.0 / (k + float(record["sem_rank"])))
        # Sort results by rrf_score
        sorted_accum = dict(sorted(accumulator.items(), key=lambda x: x[1]["rrf_score"], reverse=True))
        # Return 'new_limit' number of results
        new_limit = min(limit, len(sorted_accum))
        return [
            RRFScoreRecord(
                id=doc_id,
                data=data
            )
            for i, (doc_id, data) in enumerate(sorted_accum.items())
            if i < new_limit
        ]


def normalize(nums: list[float]) -> list[float]:
    if len(nums) == 0:
        return []
    min_nums = min(nums)
    max_nums = max(nums)
    range_nums = max_nums - min_nums
    return list(map(lambda x: (x - min_nums)/(range_nums) if range_nums != 0 else 1.0, nums))


def normalize_semantic_chunk_results(input: list[ChunkedSearchResult]) -> list[ChunkedSearchResult]:
    """
    Takes a list of ChunkedSearcResults, normalizes the scores, and returns a new list with the
    ChunkSearchResults in the same order (id-wise), with the scores replaced with their normalized
    counterparts.
    """
    results: list[ChunkedSearchResult] = []
    normalized_scores_list = normalize(list(map(lambda result: result["score"], input)))
    # Put normalized scores back into the ChunkedSearchResult "records"; order should have been retained
    for index, chunk_result in enumerate(input):
        results.append(ChunkedSearchResult(
            id=chunk_result["id"],
            title=chunk_result["title"],
            document=chunk_result["document"],
            metadata=chunk_result["metadata"],
            score=normalized_scores_list[index]
        ))
    return results


def normalize_command(args) -> None:
    if len(args) == 0:
        print("No numbers to normalize.")
        return
    norm_nums = normalize(args)
    for num in norm_nums:
        print(f"* {num:.4f}")


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def weighted_search_command(query: str, alpha: float, limit: int) -> None:
    documents = load_movies(HybridSearch.MOVIES_JSON_FILE.__str__())
    hybrid_search = HybridSearch(documents.movies)
    search_results = hybrid_search.weighted_search(query, alpha, limit)
    for index, result in enumerate(search_results):
        print(f"{index+1}. {hybrid_search.idx.docmap[result['id']].title}")
        print(f"\tHybrid score: {result['data']['hybrid_score']}")
        print(f"\tBM25: {result['data']['keyword_score']}, Semantic: {result['data']['semantic_score']}")
        print(f"\t{result['data']['document']:.80}")


def rrf_search_command(query: str, k: float, limit: int, enhance: str) -> None:
    old_query = query
    new_query: str | None = None
    # If the enhance string exists, do some enhancing of the query
    match enhance:
        case "spell":
            new_query = llm_spell_check(query)
        case "rewrite":
            new_query = llm_rewrite_query(query)
        case "expand":
            new_query = llm_expand_query(query)
        case _:
            new_query = None
    if enhance:
        print(f"Enhanced query ({enhance}): '{old_query}' -> '{new_query}'\n")
    documents = load_movies(HybridSearch.MOVIES_JSON_FILE.__str__())
    hybrid_search = HybridSearch(documents.movies)
    search_results = hybrid_search.rrf_search(new_query if new_query else old_query, k, limit)
    for index, result in enumerate(search_results):
        print(f"{index+1}. {hybrid_search.idx.docmap[result['id']].title}")
        print(f"\tRRF score: {result['data']['rrf_score']}")
        print(f"\tBM25 rank: {result['data']['bm25_rank']}, Semantic rank: {result['data']['sem_rank']}")
        print(f"\t{result['data']['document']:.80}")
