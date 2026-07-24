# Adapters are imported lazily by client.py (e.g. `from .adapters.langgraph import ...`
# inside a method body) so importing `mcpgram` never eagerly imports optional
# framework packages you might not have installed.
