"""
Generation: retrieved chunks + question -> answer.

Uses a local LLM via Ollama (config.LLM_MODEL). Start Ollama and pull the
model first, e.g.:  ollama pull qwen2.5:3b

Two modes:
  - plain (allow_search=False): answer only from context, or say "nu știu".
  - search-enabled (allow_search=True): used by the iterative pipeline — the
    model may reply "CAUTĂ: <concept>" to request another retrieval round.
"""

import config

SYSTEM_BASE = (
    "Ești asistentul de asigurări al companiei Scutul Carpatic. "
    "Răspunzi în limba română, clar și concis, cu cuvinte simple și uzuale. "
    "Nu folosi liniuța de pauză lungă; folosește virgule, puncte sau două puncte."
)
SEARCH_INSTRUCTION = (
    " Dacă nu poți răspunde complet pe baza contextului și ai nevoie de mai multe "
    "informații despre un concept anume, răspunde STRICT cu: CAUTĂ: <concept>."
)


def build_prompt(query: str, chunks: list[dict], allow_search: bool = False) -> str:
    """
    Build a grounded prompt: a context block with each chunk's source, plus
    strict instructions to answer ONLY from that context (and to admit when
    the answer isn't there). When allow_search is on, the model may instead
    request more context with "CAUTĂ: <concept>".
    """
    if chunks:
        context = "\n\n".join(f"[Sursa: {c['source']}]\n{c['text']}" for c in chunks)
    else:
        context = "(niciun fragment relevant găsit)"

    system = SYSTEM_BASE + (SEARCH_INSTRUCTION if allow_search else "")
    if allow_search:
        miss = ("Dacă lipsește o parte din informație, cere o nouă căutare cu "
                "'CAUTĂ: <ce lipsește>'. ")
    else:
        miss = ("Dacă răspunsul nu se află în context, spune exact: "
                "\"Nu am găsit această informație în documentele disponibile.\" ")

    return (
        f"{system}\n\n"
        "Folosește DOAR informațiile din contextul de mai jos pentru a răspunde. "
        f"{miss}Nu inventa cifre sau clauze. Menționează sursa relevantă.\n\n"
        f"=== CONTEXT ===\n{context}\n=== SFÂRȘIT CONTEXT ===\n\n"
        f"Întrebare: {query}\n"
        "Răspuns:"
    )


def generate_answer(query: str, chunks: list[dict], allow_search: bool = False) -> str:
    """Build the prompt and ask the local LLM (Ollama) for an answer."""
    prompt = build_prompt(query, chunks, allow_search)
    try:
        import ollama
        resp = ollama.chat(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_predict": 350},
        )
        return resp["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError(
            f"Nu am putut obține un răspuns de la modelul local '{config.LLM_MODEL}'. "
            f"Verifică: 1) Ollama pornit (ollama serve), 2) modelul descărcat "
            f"(ollama pull {config.LLM_MODEL}). Detaliu: {e}"
        )
