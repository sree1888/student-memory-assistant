from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

print("Program started")

# STEP 1: Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# STEP 2: Sample long-term memory
notes = [
    "Binary Search works on sorted arrays and divides the search space into halves",
    "Stack follows LIFO principle and is used in function calls",
    "Queue follows FIFO principle and is used in scheduling",
    "Time complexity of merge sort is O(n log n)"
]

# STEP 3: Embeddings
embeddings = model.encode(notes)

# STEP 4: Local Qdrant (FREE)
client = QdrantClient(":memory:")

# STEP 5: Create collection
client.create_collection(
    collection_name="student_memory",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# STEP 6: Store data
points = [
    PointStruct(id=i, vector=embeddings[i], payload={"text": notes[i]})
    for i in range(len(notes))
]

client.upsert(collection_name="student_memory", points=points)

print("✅ Student notes stored successfully in Qdrant (Local)!")

# STEP 7: SEARCH (THIS NOW WORKS)
query = "Explain binary search algorithm"
query_vector = model.encode(query)

results = client.search(
    collection_name="student_memory",
    query_vector=query_vector,
    limit=2
)

print("\n🔍 Retrieved relevant memory:")
for r in results:
    print("-", r.payload["text"])
