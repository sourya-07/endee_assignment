import time
from gradio_client import Client

print("Waiting for app to start...")
time.sleep(2)

print("Connecting to local Gradio client...")
client = Client("http://127.0.0.1:8080/")

print("=== Test 1: Create Subjects ===")
print("Creating 'History'...")
res1 = client.predict(name="History", api_name="/create_subject")
print(res1)

print("Creating 'Science'...")
res2 = client.predict(name="Science", api_name="/create_subject")
print(res2)

print("\n=== Test 2: Ingest Content ===")
print("Ingesting to 'History'...")
res3 = client.predict(
    subject="History",
    files=["/Users/souryagupta/assi/endee_assignment/tests/test_docs/history.txt"],
    links_text="",
    reset=True,
    api_name="/ingest_subject_docs"
)
print(res3)

print("Ingesting to 'Science'...")
res4 = client.predict(
    subject="Science",
    files=["/Users/souryagupta/assi/endee_assignment/tests/test_docs/science.txt"],
    links_text="",
    reset=True,
    api_name="/ingest_subject_docs"
)
print(res4)

print("\n=== Test 3: Querying Subjects (Isolation Test) ===")
print("Querying 'Science' about History (should fail/return no relevant context)...")
res5 = client.predict(
    subject="Science",
    question="What is the longest wall in the world?",
    top_k=2,
    category_filter="All",
    use_reranker=False,
    api_name="/query_rag"
)
print("Science Output:", res5[0])

print("\nQuerying 'History' about History (should succeed)...")
res6 = client.predict(
    subject="History",
    question="What is the longest wall in the world?",
    top_k=2,
    category_filter="All",
    use_reranker=False,
    api_name="/query_rag"
)
print("History Output:", res6[0])

print("\nQuerying 'History' about Science (should fail)...")
res7 = client.predict(
    subject="History",
    question="At what temperature does water boil?",
    top_k=2,
    category_filter="All",
    use_reranker=False,
    api_name="/query_rag"
)
print("History Output:", res7[0])

print("\nQuerying 'Science' about Science (should succeed)...")
res8 = client.predict(
    subject="Science",
    question="At what temperature does water boil?",
    top_k=2,
    category_filter="All",
    use_reranker=False,
    api_name="/query_rag"
)
print("Science Output:", res8[0])
