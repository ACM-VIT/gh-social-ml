"""User Onboarding Pipeline for Git Social ML Engine.

This module handles the Day-1 Initial Interest Vector generation for new users,
taking raw profile data (skills, tech_stack, interests, bio) and generating
embeddings that are stored in Qdrant for similarity-based matching.
"""

import os
import logging
import uuid
import copy
from typing import Any, Dict

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False

# ── Isolated Configuration with Fallbacks ─────────────────────────────────────
# These fallbacks allow the script to work independently until global config.py is merged.
# Once merged, these can be replaced with: from config import EMBEDDING_MODEL, VECTOR_DIMENSION

EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Safely parse VECTOR_DIMENSION with fallback to prevent crashes on invalid values
try:
    VECTOR_DIMENSION: int = int(os.getenv("VECTOR_DIMENSION", "384"))
except (ValueError, TypeError):
    logger.warning("Invalid VECTOR_DIMENSION in environment, using default 384")
    VECTOR_DIMENSION: int = 384
QDRANT_URL: str | None = os.getenv("QDRANT_URL", None)
QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY", None)
USER_PROFILES_COLLECTION: str = "user_profiles"

logger = logging.getLogger("pipeline.user_onboarding")


def _synthesize_user_context_impl(user_data: Dict[str, Any]) -> str:
    """Pure helper function to flatten user profile data into a single dense text string.

    This function contains the core synthesis logic without any dependencies on
    the SentenceTransformer model, allowing it to be used independently.

    Args:
        user_data: Dictionary containing user profile fields:
            - skills: List[str] - User's technical skills
            - tech_stack: List[str] - User's technology stack preferences
            - interests: List[str] - User's interests
            - bio: str - User's biography/description

    Returns:
        A single dense text string combining all user profile information.
    """
    skills = user_data.get("skills", [])
    tech_stack = user_data.get("tech_stack", [])
    interests = user_data.get("interests", [])
    bio = user_data.get("bio", "")

    # Convert lists to comma-separated strings with explicit string casting
    # to handle ints, floats, None, or other non-string types safely
    if isinstance(skills, list):
        skills_str = ", ".join(str(item) if item is not None else "" for item in skills)
    else:
        skills_str = str(skills) if skills is not None else ""
    
    if isinstance(tech_stack, list):
        tech_stack_str = ", ".join(str(item) if item is not None else "" for item in tech_stack)
    else:
        tech_stack_str = str(tech_stack) if tech_stack is not None else ""
    
    if isinstance(interests, list):
        interests_str = ", ".join(str(item) if item is not None else "" for item in interests)
    else:
        interests_str = str(interests) if interests is not None else ""

    # Handle bio field - ensure it's a string
    bio_str = str(bio) if bio is not None else ""

    # Synthesize into a dense, coherent text representation
    context_parts = []
    
    if skills_str:
        context_parts.append(f"Skills: {skills_str}")
    
    if tech_stack_str:
        context_parts.append(f"Tech Stack: {tech_stack_str}")
    
    if interests_str:
        context_parts.append(f"Interests: {interests_str}")
    
    if bio_str:
        context_parts.append(f"Bio: {bio_str}")

    synthesized_context = ". ".join(context_parts)
    
    if not synthesized_context:
        logger.warning("User data is empty or missing all fields.")
        return ""

    return synthesized_context


class UserOnboardingPipeline:
    """Pipeline for generating and storing user interest vectors.

    This class handles the complete workflow of:
    1. Synthesizing user profile data into a dense text representation
    2. Generating embeddings using SentenceTransformer
    3. Storing vectors in Qdrant for similarity search
    """

    def __init__(self, embedding_model: str | None = None) -> None:
        """Initialize the user onboarding pipeline.

        Args:
            embedding_model: Name of the SentenceTransformer model to use.
                Defaults to EMBEDDING_MODEL environment variable or 'all-MiniLM-L6-v2'.

        Raises:
            ImportError: If sentence-transformers is not installed.
        """
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run 'pip install sentence-transformers' to enable embeddings."
            )

        self.model_name = embedding_model or EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)
        logger.info(f"Initialized SentenceTransformer with model: {self.model_name}")

    def synthesize_user_context(self, user_data: Dict[str, Any]) -> str:
        """Flatten user profile data into a single dense text string.

        This method delegates to the pure helper function to avoid loading
        the SentenceTransformer model when only text synthesis is needed.

        Args:
            user_data: Dictionary containing user profile fields:
                - skills: List[str] - User's technical skills
                - tech_stack: List[str] - User's technology stack preferences
                - interests: List[str] - User's interests
                - bio: str - User's biography/description

        Returns:
            A single dense text string combining all user profile information.

        Example:
            >>> user_data = {
            ...     "skills": ["Python", "Machine Learning"],
            ...     "tech_stack": ["PyTorch", "FastAPI"],
            ...     "interests": ["AI", "Open Source"],
            ...     "bio": "ML engineer passionate about NLP"
            ... }
            >>> context = synthesize_user_context(user_data)
            >>> print(context)
            'Skills: Python, Machine Learning. Tech Stack: PyTorch, FastAPI. ...
        """
        return _synthesize_user_context_impl(user_data)

    def generate_interest_vector(self, user_data: Dict[str, Any]) -> list[float]:
        """Generate Day-1 Initial Interest Vector from user profile data.

        This method synthesizes the user context and generates an embedding
        vector using the configured SentenceTransformer model.

        Args:
            user_data: Dictionary containing user profile fields (skills, tech_stack,
                interests, bio).

        Returns:
            A list of floats representing the user's interest vector.

        Raises:
            ValueError: If user_data is invalid, empty, or synthesis fails.
        """
        # Validate user_data is a dictionary
        if not isinstance(user_data, dict):
            raise ValueError("user_data must be a dictionary.")

        context = self.synthesize_user_context(user_data)
        
        if not context:
            raise ValueError("Cannot generate vector from empty user data.")

        try:
            embedding = self.model.encode(context)
            if embedding is None:
                raise ValueError("Model.encode returned None.")
            vector = embedding.tolist()
            
            # Validate vector dimension matches expected dimension
            if len(vector) != VECTOR_DIMENSION:
                raise ValueError(
                    f"Generated vector dimension {len(vector)} does not match "
                    f"expected VECTOR_DIMENSION {VECTOR_DIMENSION}. "
                    "This indicates a model mismatch or configuration error."
                )
            
            # Validate vector elements are numbers
            import math
            for i, val in enumerate(vector):
                if not isinstance(val, (int, float)):
                    raise ValueError(f"Vector element at index {i} is not a number: {val}")
                if math.isnan(val) or math.isinf(val):
                    raise ValueError(f"Vector element at index {i} is NaN or Inf: {val}")
            
            return vector
        except Exception as exc:
            logger.error(f"Failed to generate embedding: {exc}")
            raise

    def save_to_qdrant(
        self,
        user_id: str,
        vector: list[float],
        payload: Dict[str, Any] | None = None,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
    ) -> bool:
        """Save user interest vector to Qdrant database.

        This method ensures the 'user_profiles' collection exists and stores
        the vector alongside the user's ID and any additional payload data.

        Args:
            user_id: Unique identifier for the user (used as point ID in Qdrant).
            vector: The interest vector to store.
            payload: Optional dictionary of additional metadata to store with the vector.
                If None, a default payload with user_id will be created.
            qdrant_url: Qdrant server URL. Defaults to QDRANT_URL environment variable.
            qdrant_api_key: Qdrant API key. Defaults to QDRANT_API_KEY environment variable.

        Returns:
            True if the vector was successfully saved, False otherwise.

        Raises:
            ImportError: If qdrant-client is not installed.
            ValueError: If user_id or vector validation fails.
        """
        if not HAS_QDRANT:
            raise ImportError(
                "qdrant-client is not installed. "
                "Run 'pip install qdrant-client' to enable Qdrant storage."
            )

        # Validate user_id
        if not user_id or not isinstance(user_id, str):
            raise ValueError("user_id must be a non-empty string.")

        # Validate vector
        if not isinstance(vector, list) or len(vector) == 0:
            raise ValueError("vector must be a non-empty list.")

        # Validate vector elements are numbers and not NaN/Inf
        import math
        for i, val in enumerate(vector):
            if not isinstance(val, (int, float)):
                raise ValueError(f"Vector element at index {i} is not a number: {val}")
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Vector element at index {i} is NaN or Inf: {val}")

        # Use provided parameters or fall back to environment variables
        url = qdrant_url or QDRANT_URL
        api_key = qdrant_api_key or QDRANT_API_KEY

        if not url:
            logger.error(
                "QDRANT_URL is not set. Cannot connect to Qdrant. "
                "Set it in your .env file or pass it as a parameter."
            )
            return False

        # Ensure payload contains user_id (copy to avoid mutating caller's object)
        if payload is None:
            payload = {}
        else:
            payload = copy.copy(payload)
        payload["user_id"] = user_id

        try:
            # Initialize Qdrant client with timeout for production safety
            client = QdrantClient(url=url, api_key=api_key, timeout=30.0)

            # Check if collection exists, create if not
            collections = client.get_collections().collections
            collection_names = [col.name for col in collections]

            if USER_PROFILES_COLLECTION not in collection_names:
                logger.info(f"Creating collection '{USER_PROFILES_COLLECTION}'...")
                try:
                    client.create_collection(
                        collection_name=USER_PROFILES_COLLECTION,
                        vectors_config=VectorParams(
                            size=VECTOR_DIMENSION,
                            distance=Distance.COSINE,
                        ),
                    )
                    logger.info(f"Collection '{USER_PROFILES_COLLECTION}' created successfully.")
                except Exception as exc:
                    # Handle race condition: collection may have been created by another process
                    if "already exists" in str(exc).lower() or "collection" in str(exc).lower():
                        logger.info(f"Collection '{USER_PROFILES_COLLECTION}' already exists (race condition handled).")
                    else:
                        raise

            # Convert user_id to deterministic UUID for Qdrant compatibility
            point_uuid = uuid.uuid5(uuid.NAMESPACE_URL, f"user:{user_id}")

            # Insert the vector as a new point
            point = PointStruct(
                id=str(point_uuid),
                vector=vector,
                payload=payload,
            )

            client.upsert(
                collection_name=USER_PROFILES_COLLECTION,
                points=[point],
            )

            logger.info(f"Successfully saved user '{user_id}' interest vector to Qdrant.")
            return True

        except Exception as exc:
            logger.error(f"Failed to save vector to Qdrant: {exc}")
            return False

    def onboard_user(
        self,
        user_id: str,
        user_data: Dict[str, Any],
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
    ) -> bool:
        """Complete onboarding workflow: generate vector and save to Qdrant.

        This is a convenience method that combines vector generation and storage
        into a single operation.

        Args:
            user_id: Unique identifier for the user.
            user_data: Dictionary containing user profile fields (skills, tech_stack,
                interests, bio).
            qdrant_url: Optional Qdrant server URL.
            qdrant_api_key: Optional Qdrant API key.

        Returns:
            True if onboarding succeeded, False otherwise.
        """
        try:
            vector = self.generate_interest_vector(user_data)
            return self.save_to_qdrant(
                user_id=user_id,
                vector=vector,
                payload=user_data,
                qdrant_url=qdrant_url,
                qdrant_api_key=qdrant_api_key,
            )
        except Exception as exc:
            logger.error(f"User onboarding failed for '{user_id}': {exc}")
            return False


# ── Convenience Functions ───────────────────────────────────────────────────────

def synthesize_user_context(user_data: Dict[str, Any]) -> str:
    """Flatten user profile data into a single dense text string.

    This is a standalone convenience function that calls the pure helper
    directly without loading the SentenceTransformer model.

    Args:
        user_data: Dictionary containing user profile fields.

    Returns:
        A single dense text string combining all user profile information.
    """
    return _synthesize_user_context_impl(user_data)


def generate_interest_vector(user_data: Dict[str, Any]) -> list[float]:
    """Generate Day-1 Initial Interest Vector from user profile data.

    This is a standalone convenience function. For repeated use, instantiate
    the UserOnboardingPipeline class directly.

    Args:
        user_data: Dictionary containing user profile fields.

    Returns:
        A list of floats representing the user's interest vector.
    """
    pipeline = UserOnboardingPipeline()
    return pipeline.generate_interest_vector(user_data)


def save_user_vector_to_qdrant(
    user_id: str,
    vector: list[float],
    payload: Dict[str, Any] | None = None,
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
) -> bool:
    """Save user interest vector to Qdrant database.

    This is a standalone convenience function. For repeated use, instantiate
    the UserOnboardingPipeline class directly.

    Args:
        user_id: Unique identifier for the user.
        vector: The interest vector to store.
        payload: Optional dictionary of additional metadata.
        qdrant_url: Optional Qdrant server URL.
        qdrant_api_key: Optional Qdrant API key.

    Returns:
        True if the vector was successfully saved, False otherwise.
    """
    pipeline = UserOnboardingPipeline()
    return pipeline.save_to_qdrant(
        user_id=user_id,
        vector=vector,
        payload=payload,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
    )


def onboard_user(
    user_id: str,
    user_data: Dict[str, Any],
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
) -> bool:
    """Complete onboarding workflow: generate vector and save to Qdrant.

    This is a standalone convenience function. For repeated use, instantiate
    the UserOnboardingPipeline class directly.

    Args:
        user_id: Unique identifier for the user.
        user_data: Dictionary containing user profile fields.
        qdrant_url: Optional Qdrant server URL.
        qdrant_api_key: Optional Qdrant API key.

    Returns:
        True if onboarding succeeded, False otherwise.
    """
    pipeline = UserOnboardingPipeline()
    return pipeline.onboard_user(
        user_id=user_id,
        user_data=user_data,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
    )


# ── Example Usage ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load environment variables from .env file for local testing
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        logger.warning("python-dotenv not installed. Skipping .env loading.")
    
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Example user data
    example_user = {
        "user_id": "user_123",
        "skills": ["Python", "Machine Learning", "Data Engineering"],
        "tech_stack": ["PyTorch", "FastAPI", "PostgreSQL", "Docker"],
        "interests": ["AI/ML", "Open Source", "Cloud Computing", "MLOps"],
        "bio": "Senior ML Engineer with 5+ years experience building scalable ML pipelines.",
    }

    # Perform onboarding
    success = onboard_user(
        user_id=example_user["user_id"],
        user_data=example_user,
    )

    if success:
        print(f"✅ User '{example_user['user_id']}' onboarded successfully!")
    else:
        print(f"❌ Failed to onboard user '{example_user['user_id']}'. Check logs for details.")
