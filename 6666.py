from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url="https://1db6d8ba-525a-4ac3-a0db-8543aefe8461.eu-central-1-0.aws.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6MTBjNGY0Y2ItODgwOS00MDdhLTk1YmYtNTZhZmQwZDQ1NTQ5In0.Yd48yAYq4XSaaqMDLO6HULnfl72b6FjRqCIZKh8c5NI",

    check_compatibility=False
)

print(qdrant_client.get_collections())