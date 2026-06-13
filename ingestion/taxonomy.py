def classify_category(repo: dict, tags: list[str]) -> str:
    react_ui_match = any("react" in t.lower() or "ui" in t.lower() for t in tags)
    react_agent_match = any("agent" in t.lower() or "llm" in t.lower() for t in tags)

    if react_agent_match and not react_ui_match:
        return "AI Agent"
    if react_ui_match and not react_agent_match:
        return "Frontend"
        
    for tag in tags:
        if tag.lower() in ["backend", "api", "database"]:
            return "Backend"
        if tag.lower() in ["frontend", "ui", "ux", "react", "vue", "angular"]:
            return "Frontend"
        if tag.lower() in ["ai", "ml", "machine learning", "data science", "llm"]:
            return "AI/ML"
        if tag.lower() in ["devops", "ci/cd", "infrastructure", "kubernetes", "docker"]:
            return "DevOps"
            
    # Default fallback
    return "Miscellaneous"
