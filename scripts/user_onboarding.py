"""User Onboarding Pipeline for Git Social ML Engine.

This module handles the Day-1 Initial Interest Vector generation for new users,
taking raw profile data (skills, tech_stack, interests, bio) and generating
embeddings that are stored in Qdrant for similarity-based matching.
"""

import os
import logging
import uuid
import copy
import math
from typing import Any, Dict

# Initialize logger immediately after imports to be available for env parsing
logger = logging.getLogger("pipeline.user_onboarding")

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
    dim_value = os.getenv("VECTOR_DIMENSION", "384")
    VECTOR_DIMENSION: int = int(dim_value)
    # Validate dimension is positive and reasonable
    if VECTOR_DIMENSION <= 0:
        logger.warning(f"VECTOR_DIMENSION must be positive, got {VECTOR_DIMENSION}. Using default 384.")
        VECTOR_DIMENSION = 384
    elif VECTOR_DIMENSION > 10000:
        logger.warning(f"VECTOR_DIMENSION unusually large ({VECTOR_DIMENSION}). Using default 384.")
        VECTOR_DIMENSION = 384
except (ValueError, TypeError):
    logger.warning(f"Invalid VECTOR_DIMENSION in environment: {dim_value}. Using default 384")
    VECTOR_DIMENSION: int = 384

QDRANT_URL: str | None = os.getenv("QDRANT_URL", None)
QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY", None)
USER_PROFILES_COLLECTION: str = "user_profiles"
# Optional explicit target for multi-vector collections
TARGET_VECTOR_NAME: str | None = os.getenv("TARGET_VECTOR_NAME", None)


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

    Raises:
        ValueError: If user_data is not a dictionary.
    """
    # Validate user_data is a dictionary to prevent AttributeError on .get()
    if not isinstance(user_data, dict):
        raise ValueError("user_data must be a dictionary.")

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
        raise ValueError("User data is empty or missing all fields. Cannot synthesize context.")

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

        The SentenceTransformer model is loaded lazily on first use to allow
        the pipeline to be instantiated in environments without ML dependencies
        (e.g., for saving precomputed vectors to Qdrant).

        Args:
            embedding_model: Name of the SentenceTransformer model to use.
                Defaults to EMBEDDING_MODEL environment variable or 'all-MiniLM-L6-v2'.
        """
        self.model_name = embedding_model or EMBEDDING_MODEL
        self._model = None  # Lazy-loaded model

    def _get_model(self):
        """Lazy-load the SentenceTransformer model on first access.

        Returns:
            The loaded SentenceTransformer model.

        Raises:
            ImportError: If sentence-transformers is not installed.
        """
        if self._model is None:
            if not HAS_SENTENCE_TRANSFORMERS:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Run 'pip install sentence-transformers' to enable embeddings."
                )
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Lazy-loaded SentenceTransformer with model: {self.model_name}")
        return self._model

    @property
    def model(self):
        """Property to access the lazily-loaded model."""
        return self._get_model()

    def synthesize_user_context(self, user_data: Dict[str, Any]) -> str:
        """Flatten user profile data into a single dense text string.

        This method delegates to the pure helper function to avoid loading
        the SentenceTransformer model when only text synthesis is needed.

        Args:
            user_data: Dictionary containing user profile fields.

        Returns:
            A single dense text string combining all user profile information.
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
            model = self._get_model()  # Lazy load
            embedding = model.encode(context)
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
            True if the vector was successfully saved.

        Raises:
            ImportError: If qdrant-client is not installed.
            ValueError: If user_id, vector validation fails, or QDRANT_URL is not set.
            Exception: If Qdrant operation fails (propagated for caller handling).
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

        # Validate vector dimension matches expected dimension
        if len(vector) != VECTOR_DIMENSION:
            raise ValueError(
                f"Vector dimension {len(vector)} does not match "
                f"expected VECTOR_DIMENSION {VECTOR_DIMENSION}. "
                "This indicates a model mismatch or configuration error."
            )

        # Validate vector elements are numbers and not NaN/Inf
        for i, val in enumerate(vector):
            if not isinstance(val, (int, float)):
                raise ValueError(f"Vector element at index {i} is not a number: {val}")
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Vector element at index {i} is NaN or Inf: {val}")

        # Use provided parameters or fall back to environment variables
        url = qdrant_url or QDRANT_URL
        api_key = qdrant_api_key or QDRANT_API_KEY

        if not url:
            raise ValueError(
                "QDRANT_URL is not set. Cannot connect to Qdrant. "
                "Set it in your .env file or pass it as a parameter."
            )

        # Ensure payload contains user_id (deep copy to avoid mutating caller's nested objects)
        if payload is None:
            payload = {}
        else:
            payload = copy.deepcopy(payload)
        payload["user_id"] = user_id

        try:
            # Initialize Qdrant client with timeout for production safety
            client = QdrantClient(url=url, api_key=api_key, timeout=30.0)
            
            # Default vector_name to None for standard collections
            vector_name = None

            # Check if collection exists, create if not
            collections_response = client.get_collections()
            if collections_response is None or collections_response.collections is None:
                raise ValueError("Failed to retrieve collections from Qdrant.")
            collections = collections_response.collections
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
                    if "already exists" in str(exc).lower():
                        logger.info(f"Collection '{USER_PROFILES_COLLECTION}' already exists (race condition handled).")
                        logger.info(f"Validating schema for race-condition-created collection...")
                        collection_info = client.get_collection(USER_PROFILES_COLLECTION)
                        if collection_info is None or collection_info.config is None or collection_info.config.params is None:
                            raise ValueError(f"Failed to retrieve configuration for collection '{USER_PROFILES_COLLECTION}'.")
                        
                        vectors_config = collection_info.config.params.vectors
                        if vectors_config is None:
                            raise ValueError(f"Collection '{USER_PROFILES_COLLECTION}' has no vectors configuration.")
                        
                        if isinstance(vectors_config, dict):
                            if "size" in vectors_config:
                                existing_size = vectors_config.get("size")
                                existing_distance = vectors_config.get("distance")
                            else:
                                if not vectors_config:
                                    raise ValueError(f"Collection '{USER_PROFILES_COLLECTION}' has empty named-vector configuration.")
                                
                                # GREPTILE FIX: Explicit vector selection logic factoring in distance too
                                if TARGET_VECTOR_NAME:
                                    if TARGET_VECTOR_NAME not in vectors_config:
                                        raise ValueError(f"Target vector '{TARGET_VECTOR_NAME}' not found in collection.")
                                    vector_name = TARGET_VECTOR_NAME
                                else:
                                    expected_dist_str = str(Distance.COSINE).upper()
                                    matching_vectors = [
                                        name for name, conf in vectors_config.items() 
                                        if getattr(conf, "size", None) == VECTOR_DIMENSION and 
                                           expected_dist_str in str(getattr(conf, "distance", "")).upper()
                                    ]
                                    if len(matching_vectors) == 1:
                                        vector_name = matching_vectors[0]
                                    elif len(matching_vectors) == 0:
                                        raise ValueError(f"No vectors match dimension {VECTOR_DIMENSION} and distance {Distance.COSINE} in collection '{USER_PROFILES_COLLECTION}'.")
                                    else:
                                        raise ValueError(f"Ambiguous vector targets. Multiple vectors match dimension {VECTOR_DIMENSION} and distance {Distance.COSINE}. Set TARGET_VECTOR_NAME.")
                                
                                target_config = vectors_config[vector_name]
                                existing_size = getattr(target_config, "size", None)
                                existing_distance = getattr(target_config, "distance", None)
                        else:
                            existing_size = getattr(vectors_config, "size", None)
                            existing_distance = getattr(vectors_config, "distance", None)
                        
                        if existing_size != VECTOR_DIMENSION:
                            raise ValueError(
                                f"Collection schema mismatch: existing vector size is {existing_size}, but expected {VECTOR_DIMENSION}."
                            )
                        
                        expected_distance = str(Distance.COSINE).upper()
                        actual_distance = str(existing_distance).upper() if existing_distance else ""
                        if expected_distance not in actual_distance and actual_distance not in expected_distance:
                            raise ValueError(
                                f"Collection schema mismatch: existing distance metric is {existing_distance}, but expected {Distance.COSINE}."
                            )
                    else:
                        raise
            else:
                # Collection exists - validate schema matches expected configuration
                logger.info(f"Collection '{USER_PROFILES_COLLECTION}' already exists. Validating schema...")
                collection_info = client.get_collection(USER_PROFILES_COLLECTION)
                if collection_info is None or collection_info.config is None or collection_info.config.params is None:
                    raise ValueError(f"Failed to retrieve configuration for collection '{USER_PROFILES_COLLECTION}'.")
                
                vectors_config = collection_info.config.params.vectors
                if vectors_config is None:
                    raise ValueError(f"Collection '{USER_PROFILES_COLLECTION}' has no vectors configuration.")
                
                if isinstance(vectors_config, dict):
                    if "size" in vectors_config:
                        existing_size = vectors_config.get("size")
                        existing_distance = vectors_config.get("distance")
                    else:
                        if not vectors_config:
                            raise ValueError(f"Collection '{USER_PROFILES_COLLECTION}' has empty named-vector configuration.")
                        
                        # GREPTILE FIX: Explicit vector selection logic factoring in distance too
                        if TARGET_VECTOR_NAME:
                            if TARGET_VECTOR_NAME not in vectors_config:
                                raise ValueError(f"Target vector '{TARGET_VECTOR_NAME}' not found in collection.")
                            vector_name = TARGET_VECTOR_NAME
                        else:
                            expected_dist_str = str(Distance.COSINE).upper()
                            matching_vectors = [
                                name for name, conf in vectors_config.items() 
                                if getattr(conf, "size", None) == VECTOR_DIMENSION and 
                                   expected_dist_str in str(getattr(conf, "distance", "")).upper()
                            ]
                            if len(matching_vectors) == 1:
                                vector_name = matching_vectors[0]
                            elif len(matching_vectors) == 0:
                                raise ValueError(f"No vectors match dimension {VECTOR_DIMENSION} and distance {Distance.COSINE} in collection '{USER_PROFILES_COLLECTION}'.")
                            else:
                                raise ValueError(f"Ambiguous vector targets. Multiple vectors match dimension {VECTOR_DIMENSION} and distance {Distance.COSINE}. Set TARGET_VECTOR_NAME.")
                        
                        target_config = vectors_config[vector_name]
                        existing_size = getattr(target_config, "size", None)
                        existing_distance = getattr(target_config, "distance", None)
                else:
                    existing_size = getattr(vectors_config, "size", None)
                    existing_distance = getattr(vectors_config, "distance", None)
                
                if existing_size != VECTOR_DIMENSION:
                    raise ValueError(
                        f"Collection schema mismatch: existing vector size is {existing_size}, but expected {VECTOR_DIMENSION}."
                    )
                
                expected_distance = str(Distance.COSINE).upper()
                actual_distance = str(existing_distance).upper() if existing_distance else ""
                if expected_distance not in actual_distance and actual_distance not in expected_distance:
                    raise ValueError(
                        f"Collection schema mismatch: existing distance metric is {existing_distance}, but expected {Distance.COSINE}."
                    )
                
                logger.info(f"Collection '{USER_PROFILES_COLLECTION}' schema validated successfully.")

            # Convert user_id to deterministic UUID for Qdrant compatibility
            point_uuid = uuid.uuid5(uuid.NAMESPACE_URL, f"user:{user_id}")

            # Insert the vector as a new point
            # Apply dynamic handling for named versus unnamed vectors
            final_vector = {vector_name: vector} if vector_name is not None else vector

            point = PointStruct(
                id=str(point_uuid),
                vector=final_vector,
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
            # Re-raise to allow caller to handle specific errors
            raise

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
    return _synthesize_user_context_impl(user_data)


def generate_interest_vector(user_data: Dict[str, Any]) -> list[float]:
    if not isinstance(user_data, dict):
        raise ValueError("user_data must be a dictionary.")
    
    pipeline = UserOnboardingPipeline()
    return pipeline.generate_interest_vector(user_data)


def save_user_vector_to_qdrant(
    user_id: str,
    vector: list[float],
    payload: Dict[str, Any] | None = None,
    qdrant_url: str | None = None,
    qdrant_api_key: str | None = None,
) -> bool:
    if not user_id or not isinstance(user_id, str):
        raise ValueError("user_id must be a non-empty string.")
    if not isinstance(vector, list) or len(vector) == 0:
        raise ValueError("vector must be a non-empty list.")
    
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
    if not user_id or not isinstance(user_id, str):
        raise ValueError("user_id must be a non-empty string.")
    if not isinstance(user_data, dict):
        raise ValueError("user_data must be a dictionary.")
    
    try:
        pipeline = UserOnboardingPipeline()
        return pipeline.onboard_user(
            user_id=user_id,
            user_data=user_data,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
        )
    except Exception as exc:
        logger.error(f"User onboarding failed for '{user_id}': {exc}")
        return False


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