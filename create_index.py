from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

# 🔥 改成你的
QDRANT_URL = "https://1db6d8ba-525a-4ac3-a0db-8543aefe8461.eu-central-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6ODM2Mzk4MDUtYTVmNS00MzUyLWE2NWEtZWNlMWUxNWYxZTE3In0.SStK2mFTKzbEvbWc2r8B2s7TiXE68ETTKrPvmrkiJ7A"

COLLECTION_NAME = "error_codes"

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="error_code",
    field_schema=PayloadSchemaType.KEYWORD
)

print("✅ error_code index 建立完成")