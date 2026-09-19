"""Controlled failures used only by the evaluation harness."""


def failing_retriever(_query):
    raise RuntimeError("intentional evaluation retrieval failure")