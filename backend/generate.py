import config

# We update the system prompt to allow a specific "SEARCH:" fallback
SYSTEM = (
    "Ești asistentul de asigurări al companiei Scutul Carpatic. "
    "Răspunzi în limba română, clar și concis, cu cuvinte simple și uzuale. "
    "Dacă nu poți formula un răspuns complet pe baza contextului furnizat și ai nevoie "
    "de mai multe informații despre un concept anume, răspunde STRICT cu: CAUTĂ: <concept>. "
    "Nu folosi liniuța de pauză lungă; folosește virgule, puncte sau două puncte."
)

def build_prompt(query: str, chunks: list[dict]) -> str:
    if chunks:
        context = "\n\n".join(
            f"[Sursa: {c['source']}]\n{c['text']}" for c in chunks
        )
    else:
        context = "(niciun fragment relevant găsit)"

    return (
        f"{SYSTEM}\n\n"
        "Folosește DOAR informațiile din contextul de mai jos pentru a răspunde. "
        "Dacă contextul este insuficient pentru o parte din răspuns, cere o nouă căutare "
        "folosind 'CAUTĂ: <ce lipsește>'. Nu inventa cifre sau clauze.\n\n"
        f"=== CONTEXT ===\n{context}\n=== SFÂRȘIT CONTEXT ===\n\n"
        f"Întrebare inițială/Curentă: {query}\n"
        "Răspuns:"
    )

def generate_answer(query: str, chunks: list[dict]) -> str:
    """Build the prompt and ask the local LLM (Ollama) for an answer."""
    prompt = build_prompt(query, chunks)
    try:
        import ollama
        resp = ollama.chat(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_predict": 350}, # Lowered temperature for stricter formatting
        )
        return resp["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError(
            f"Nu am putut obține un răspuns de la modelul local '{config.LLM_MODEL}'. "
            f"Verifică: 1) Ollama pornit (ollama serve), 2) modelul descărcat "
            f"(ollama pull {config.LLM_MODEL}). Detaliu: {e}"
        )