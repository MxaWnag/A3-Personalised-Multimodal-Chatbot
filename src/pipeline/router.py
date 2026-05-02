def route_query(query: str) -> str:
    q = query.lower()
    if any(k in q for k in ["image", "photo", "figure", "diagram", "shown"]):
        return "image"
    if any(k in q for k in ["compare", "combine", "summarize", "summary", "multi"]):
        return "hybrid"
    return "text"
