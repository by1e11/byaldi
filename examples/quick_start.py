from byaldi import RAGMultiModalModel
# Optionally, you can specify an `index_root`, which is where it'll save the index. It defaults to ".byaldi/".
RAG = RAGMultiModalModel.from_pretrained(
    "vidore/colqwen2.5-v0.2"
)

RAG.index(
    input_path="./examples/docs/ACL-3.pdf",
    index_name="attention",
    store_collection_with_index=True,
    overwrite=True
)

query = "What are the key contributions of this paper?"

results = RAG.search(query, k=1)

pass