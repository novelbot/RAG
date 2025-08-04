
import sys
sys.path.insert(0, "src")

from src.core.config import get_config
from src.embedding.factory import get_embedding_manager

# Test configuration loading
config = get_config()
print(f"✅ Config loaded: embedding={config.embedding.provider}:{config.embedding.model}")

# Test embedding manager
embedding_manager = get_embedding_manager([config.embedding])
print(f"✅ Embedding manager initialized")

# Test embedding generation
from src.embedding.base import EmbeddingRequest
test_request = EmbeddingRequest(
    input=["test"],
    model=config.embedding.model,
    encoding_format="float"
)
response = embedding_manager.generate_embeddings(test_request)
print(f"✅ Embedding test successful: dimension={len(response.embeddings[0])}")
