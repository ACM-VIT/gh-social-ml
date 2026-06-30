import logging
import os
import re
import time
import threading
import requests

logger = logging.getLogger(__name__)

class GeminiRateLimiter:
    """Thread-safe rate limiter to prevent exceeding the Gemini/Gemma API requests-per-minute (RPM) quota."""
    def __init__(self, rpm_limit: float = 15.0):
        self.rpm_limit = rpm_limit
        self.lock = threading.Lock()
        self.last_request_time = 0.0

    def wait_if_needed(self) -> None:
        if self.rpm_limit <= 0:
            return
        
        spacing = 60.0 / self.rpm_limit
        with self.lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < spacing:
                sleep_time = spacing - elapsed
                logger.debug(f"Rate limiter: sleeping for {sleep_time:.2f}s to respect RPM limit of {self.rpm_limit}")
                time.sleep(sleep_time)
            self.last_request_time = time.time()


# Instantiate a global rate limiter. Default to 15 RPM (typical free tier constraint)
_RPM_LIMIT = float(os.getenv("GEMINI_RPM_LIMIT", "15"))
rate_limiter = GeminiRateLimiter(rpm_limit=_RPM_LIMIT)


def generate_readme_markdown(clean_text: str) -> str:
    """
    Generate a clean, structured Markdown document from the cleaned README plain text
    using the Gemma 4 E4B model. Enforces rate limits and automatically retries with 
    exponential backoff on HTTP 429 (Resource Exhausted).
    """
    if not clean_text or not clean_text.strip():
        return ""

    # Load environment variables
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMMA_API_KEY") or os.getenv("GOOGLE_API_KEY")
    api_url = os.getenv("GEMINI_API_URL") or "https://generativelanguage.googleapis.com/v1beta"
    model = os.getenv("GEMMA_MODEL_ID") or "gemma-4-26b-a4b-it"

    if not api_key:
        logger.warning(
            "No API key found for Gemma README Markdown generation. "
            "Please set GEMINI_API_KEY, GEMMA_API_KEY, or GOOGLE_API_KEY in the environment. "
            "Skipping Markdown generation."
        )
        return ""

    headers = {
        "Content-Type": "application/json",
    }
    
    prompt = (
        "You are an expert technical writer. Convert the following plain text version of a GitHub repository README "
        "into a clean, well-structured, and highly readable Markdown document. Use logical headings, subheadings, "
        "bullet points, and an improved overall structure.\n\n"
        "Instructions:\n"
        "- Do NOT add or introduce new facts, library features, or hallucinate information not present in the source text.\n"
        "- Preserve all important technical details, configurations, installation instructions, code snippets, and commands.\n"
        "- Remove unnecessary boilerplate or redundancies to make it clean and readable.\n"
        "- Return ONLY the generated Markdown text. Do NOT include any introductory or concluding comments, greetings, or conversational remarks.\n\n"
        f"Source text:\n{clean_text}"
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 4096
        }
    }

    url = f"{api_url.rstrip('/')}/models/{model}:generateContent?key={api_key}"
    max_retries = 3
    backoff_factor = 2.0

    for attempt in range(max_retries + 1):
        # Enforce RPM spacing before making the call
        rate_limiter.wait_if_needed()

        try:
            logger.info(f"Generating README markdown using Gemma model '{model}' via Cloud API (Attempt {attempt + 1}/{max_retries + 1})...")
            # Set a 60-second timeout to accommodate model reasoning/thoughts latency
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            
            if response.status_code == 200:
                res_json = response.json()
                try:
                    text_out = res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                    # Clean up potential markdown code fences wrapping the response
                    if text_out.startswith("```"):
                        text_out = re.sub(r"^```[a-zA-Z]*\n?", "", text_out)
                        if text_out.endswith("```"):
                            text_out = text_out[:-3].strip()
                    return text_out.strip()
                except (KeyError, IndexError) as e:
                    logger.error(f"Failed to parse Gemini API response payload: {e}")
                    return ""
            
            elif response.status_code == 429:
                if attempt < max_retries:
                    sleep_time = backoff_factor ** attempt * 2.0
                    logger.warning(f"Gemini API rate limit exceeded (HTTP 429). Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                    continue
                else:
                    logger.error("Gemini API rate limit exceeded (HTTP 429) and max retries exhausted.")
                    return ""
            
            else:
                logger.error(f"Gemini API returned error status {response.status_code}: {response.text}")
                return ""
                
        except Exception as exc:
            if attempt < max_retries:
                logger.warning(f"Error calling Gemini API: {exc}. Retrying...")
                time.sleep(1.0)
                continue
            else:
                logger.error(f"Error calling Gemini API after max retries: {exc}")
                return ""
    
    return ""
