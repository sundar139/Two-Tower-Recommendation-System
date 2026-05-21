"""Tests for explanation prompt and parsing behavior."""

from __future__ import annotations

from movie_recsys.serving.explanations import (
    RecommendationEvidenceItem,
    RecommendationExplanationContext,
    build_explanation_prompt,
    explain_recommendations,
    parse_or_clean_explanation,
)


class _DummyOllamaClient:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.last_prompt: str | None = None

    def generate_explanation(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.response_text


def _context() -> RecommendationExplanationContext:
    items = [
        RecommendationEvidenceItem(
            movie_id=318,
            rank_position=1,
            title="Shawshank Redemption, The (1994)",
            genres="Crime|Drama",
            release_year=1994,
            final_score=0.91,
            ranker_score=0.84,
            popularity_score=0.71,
            residual_score=0.29,
            scorer_policy="ranker_topk_popularity_backfill",
        ),
        RecommendationEvidenceItem(
            movie_id=296,
            rank_position=2,
            title="Pulp Fiction (1994)",
            genres="Comedy|Crime|Drama|Thriller",
            release_year=1994,
            final_score=0.83,
            ranker_score=0.76,
            popularity_score=0.63,
            residual_score=0.22,
            scorer_policy="ranker_topk_popularity_backfill",
        ),
    ]
    return RecommendationExplanationContext(
        user_id=709,
        user_idx=0,
        style="concise",
        include_debug=False,
        scorer_policy="ranker_topk_popularity_backfill",
        items=items,
        recent_titles=["Heat (1995)", "Fargo (1996)"],
        top_genres=["Drama", "Crime"],
    )


def test_prompt_includes_allowed_evidence_and_excludes_internal_fields() -> None:
    prompt = build_explanation_prompt(_context())

    assert "Shawshank Redemption, The (1994)" in prompt
    assert "Pulp Fiction (1994)" in prompt
    assert "Recent watched/liked titles: Heat (1995), Fargo (1996)" in prompt
    assert "User genre affinity summary: Drama, Crime" in prompt

    # Internal model state should not leak into the prompt.
    assert "embedding_vector" not in prompt
    assert "learned_weights" not in prompt
    assert "faiss_distance" not in prompt


def test_parse_or_clean_explanation_caps_to_two_sentences() -> None:
    text = " Great pick. It matches your recent genre pattern. Third sentence should be removed. "
    cleaned = parse_or_clean_explanation(text, max_sentences=2)

    assert cleaned == "Great pick. It matches your recent genre pattern."


def test_explain_recommendations_respects_max_items_and_order() -> None:
    client = _DummyOllamaClient(
        """{
  "overall_summary": "Strong fit with your recent tastes.",
  "item_explanations": [
    {"rank_position": 1, "movie_id": 318, "explanation": "Top ranked from genre affinity."},
    {"rank_position": 2, "movie_id": 296, "explanation": "High combined ranker/popularity signal."}
  ]
}"""
    )

    overall, per_item = explain_recommendations(
        context=_context(),
        client=client,  # type: ignore[arg-type]
        max_items=1,
    )

    assert overall == "Strong fit with your recent tastes."
    assert len(per_item) == 2
    assert per_item[0] == "Top ranked from genre affinity."
    assert per_item[1] is None
    assert client.last_prompt is not None
