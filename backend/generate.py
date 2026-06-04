"""
Generation: retrieved chunks + question -> answer.

Uses a local LLM via Ollama (config.LLM_MODEL). Start Ollama and pull the
model first, e.g.:  ollama pull llama3.1
"""

import config

SYSTEM = (
    "Ești asistentul de asigurări al companiei Scutul Carpatic. "
    "Răspunzi în limba română, clar și concis, cu cuvinte simple și uzuale. "
    "Nu folosi linia de pauză („—"); folosește virgule, puncte sau două puncte."
)


def build_prompt(query: str, chunks: list[dict]) -> str:
    """
    Build a grounded prompt: a context block with each chunk's source, plus
    strict instructions to answer ONLY from that context (and to admit when
    the answer isn't there — saying "nu știu" is intended behavior).
    """
    if chunks:
        context = "\n\n".join(
            f"[Sursa: {c['source']}]\n{c['text']}" for c in chunks
        )
    else:
        context = "(niciun fragment relevant găsit)"

    return (
        f"{SYSTEM}\n\n"
        "Folosește DOAR informațiile din contextul de mai jos pentru a răspunde. "
        "Dacă răspunsul nu se află în context, spune exact: "
        "\"Nu am găsit această informație în documentele disponibile.\" "
        "Nu inventa cifre sau clauze. Menționează sursa relevantă.\n\n"
        f"=== CONTEXT ===\n{context}\n=== SFÂRȘIT CONTEXT ===\n\n"
        f"Întrebare: {query}\n"
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
            # temperature low for factual answers; num_predict caps the
            # answer length so generation can't run on for ages on CPU.
            options={"temperature": 0.2, "num_predict": 350},
        )
        return resp["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError(
            f"Nu am putut obține un răspuns de la modelul local '{config.LLM_MODEL}'. "
            f"Verifică: 1) Ollama pornit (ollama serve), 2) modelul descărcat "
            f"(ollama pull {config.LLM_MODEL}). Detaliu: {e}"
        )
