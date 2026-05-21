"""Prompt building and response parsing for recommendation explanations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from movie_recsys.serving.ollama_client import OllamaClient


@dataclass(slots=True)
class RecommendationEvidenceItem:
    """Evidence fields allowed for local explanation generation."""

    movie_id: int
    rank_position: int
    title: str
    genres: str
    release_year: int | None
    final_score: float
    ranker_score: float | None
    popularity_score: float
    residual_score: float | None
    scorer_policy: str


@dataclass(slots=True)
class RecommendationExplanationContext:
    """Context passed to the explanation prompt builder and parser."""

    user_id: int | None
    user_idx: int | None
    style: Literal["concise", "detailed"]
    include_debug: bool
    scorer_policy: str
    items: list[RecommendationEvidenceItem]
    recent_titles: list[str]
    top_genres: list[str]


def _score_bucket(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score >= 0.67:
        return "high"
    if score >= 0.34:
        return "medium"
    return "low"


def _json_fence(content: dict[str, Any]) -> str:
    return json.dumps(content, indent=2, ensure_ascii=True)


def build_explanation_prompt(context: RecommendationExplanationContext) -> str:
    """Build an evidence-grounded prompt for per-item recommendation explanations."""

    items_payload: list[dict[str, Any]] = []
    for item in context.items:
        payload: dict[str, Any] = {
            "rank_position": item.rank_position,
            "movie_id": item.movie_id,
            "title": item.title,
            "genres": item.genres,
            "release_year": item.release_year,
            "final_score_level": _score_bucket(item.final_score),
            "ranker_score_level": _score_bucket(item.ranker_score),
            "popularity_score_level": _score_bucket(item.popularity_score),
            "residual_score_level": _score_bucket(item.residual_score),
            "scorer_policy": item.scorer_policy,
        }
        if context.include_debug:
            payload["debug_scores"] = {
                "final_score": item.final_score,
                "ranker_score": item.ranker_score,
                "popularity_score": item.popularity_score,
                "residual_score": item.residual_score,
            }
        items_payload.append(payload)

    request_contract = {
        "overall_summary": "string",
        "item_explanations": [
            {
                "rank_position": "integer",
                "movie_id": "integer",
                "explanation": "string <= 2 sentences",
            }
        ],
    }

    recent_titles_text = ", ".join(context.recent_titles[:8]) if context.recent_titles else "none"
    top_genres_text = ", ".join(context.top_genres[:5]) if context.top_genres else "unknown"

    return "\n".join(
        [
            "You are an assistant that explains already-ranked movie recommendations.",
            "Do not reorder, filter, or replace recommendations.",
            "Do not invent facts about movies.",
            "Do not claim the user will definitely like any movie.",
            "Use only the evidence provided.",
            "Use professional concise tone.",
            (
                "Do not reveal internal raw model details or exact numeric scores "
                "unless include_debug=true in the evidence."
            ),
            "Each item explanation must be at most 2 sentences.",
            f"Requested style: {context.style}",
            f"Scorer policy: {context.scorer_policy}",
            f"Recent watched/liked titles: {recent_titles_text}",
            f"User genre affinity summary: {top_genres_text}",
            "Return valid JSON only matching this schema:",
            _json_fence(request_contract),
            "Recommendation evidence:",
            _json_fence({"items": items_payload}),
        ]
    )


def parse_or_clean_explanation(text: str, *, max_sentences: int = 2) -> str:
    """Normalize and sentence-cap generated explanation text."""

    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return ""

    parts = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", compact) if segment.strip()]
    if not parts:
        return compact

    return " ".join(parts[:max_sentences])


def _extract_json_payload(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if match is None:
            raise
        payload = json.loads(match.group(0))

    if not isinstance(payload, dict):
        msg = "Explanation response must be a JSON object"
        raise ValueError(msg)
    return payload


def explain_recommendations(
    *,
    context: RecommendationExplanationContext,
    client: OllamaClient,
    max_items: int,
) -> tuple[str | None, list[str | None]]:
    """Generate an overall summary and per-item explanations without changing item order."""

    if max_items <= 0 or not context.items:
        return None, [None for _ in context.items]

    explain_count = min(max_items, len(context.items))
    scoped_context = RecommendationExplanationContext(
        user_id=context.user_id,
        user_idx=context.user_idx,
        style=context.style,
        include_debug=context.include_debug,
        scorer_policy=context.scorer_policy,
        items=context.items[:explain_count],
        recent_titles=context.recent_titles,
        top_genres=context.top_genres,
    )

    prompt = build_explanation_prompt(scoped_context)
    raw_text = client.generate_explanation(prompt)
    payload = _extract_json_payload(raw_text)

    overall = parse_or_clean_explanation(str(payload.get("overall_summary", "")), max_sentences=2)

    by_key: dict[tuple[int, int], str] = {}
    rows = payload.get("item_explanations", [])
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            rank_position = row.get("rank_position")
            movie_id = row.get("movie_id")
            explanation = row.get("explanation")
            if not isinstance(rank_position, int) or not isinstance(movie_id, int):
                continue
            if not isinstance(explanation, str):
                continue
            by_key[(rank_position, movie_id)] = parse_or_clean_explanation(
                explanation,
                max_sentences=2,
            )

    item_explanations: list[str | None] = []
    for item in context.items:
        if len(item_explanations) >= explain_count:
            item_explanations.append(None)
            continue

        resolved = by_key.get((item.rank_position, item.movie_id))
        if resolved:
            item_explanations.append(resolved)
            continue

        fallback = (
            f"Recommended at rank {item.rank_position} due to {item.genres or 'genre'} alignment "
            f"and {item.scorer_policy} scoring signals."
        )
        item_explanations.append(parse_or_clean_explanation(fallback, max_sentences=2))

    return (overall or None), item_explanations
