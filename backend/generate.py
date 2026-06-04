"""
Generation: retrieved chunks + question -> answer.
"""

import config


def build_prompt(query: str, chunks: list[dict]) -> str:
    """
    TODO:
      - format chunks into a readable context block
      - INCLUDE EACH CHUNK'S SOURCE — citations are the key trust feature
      - instruct the model to answer ONLY from context, and to say it
        doesn't know if the answer isn't there ("I don't know" is a feature)
    """
    raise NotImplementedError


def generate_answer(query: str, chunks: list[dict]) -> str:
    """
    TODO:
      - prompt = build_prompt(query, chunks)
      - call the LLM (config.LLM_MODEL)
      - return the text answer
    """
    raise NotImplementedError
