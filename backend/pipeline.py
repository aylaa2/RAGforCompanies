import config
import retrieval
import generate

def retrieve_chunks(query: str) -> list[dict]:
    """Helper function to run the existing retrieval stages for a given query."""
    # 1. Dense semantic search always runs (the baseline).
    results = retrieval.semantic_search(query, config.TOP_K_RETRIEVE)

    # 2. Optionally add keyword search and fuse the two ranked lists (RRF).
    if config.USE_BM25:
        bm25 = retrieval.bm25_search(query, config.TOP_K_RETRIEVE)
        results = retrieval.fuse(results, bm25)

    # 3. Optionally rerank with the cross-encoder, else just truncate.
    if config.USE_RERANKER:
        results = retrieval.rerank(query, results, config.TOP_K_FINAL)
    else:
        results = results[: config.TOP_K_FINAL]
        
    return results

def answer_question(query: str, max_iterations: int = 3) -> dict:
    """
    Run the iterative pipeline.
    Returns {"answer": str, "chunks": list[dict]}.
    """
    all_chunks = {}
    current_search_query = query
    
    for iteration in range(max_iterations):
        # 1. Retrieve based on the current search query
        new_chunks = retrieve_chunks(current_search_query)
        
        # Add new chunks to our accumulated context (deduplicating by ID)
        for chunk in new_chunks:
            if chunk["id"] not in all_chunks:
                all_chunks[chunk["id"]] = chunk
                
        context_list = list(all_chunks.values())
        
        # 2. Ask the LLM to generate an answer or request more context
        answer = generate.generate_answer(query, context_list)
        
        # 3. Check if the LLM wants to search for more info
        if answer.startswith("CAUTĂ:"):
            # Extract the new search term and loop again
            current_search_query = answer.replace("CAUTĂ:", "").strip()
            print(f"Iterația {iteration + 1}: LLM a cerut informații suplimentare -> {current_search_query}")
            continue
        else:
            # The LLM provided a final answer
            return {"answer": answer, "chunks": context_list}

    # 4. Fallback: If we hit max iterations, return what we have
    print("Avertisment: S-a atins limita maximă de iterații.")
    return {"answer": answer, "chunks": list(all_chunks.values())}