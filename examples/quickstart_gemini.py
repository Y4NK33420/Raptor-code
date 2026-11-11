"""Build a RAPTOR index over demo/sample.txt (Cinderella) with Gemini and ask a question.

Needs GEMINI_API_KEY in the environment or a .env file.

    python examples/quickstart_gemini.py
"""

from pathlib import Path

from raptor import RaptorConfig, RetrievalAugmentation

config = RaptorConfig()
config.chunking.max_tokens = 150
config.builder.max_layers = 3

ra = RetrievalAugmentation(config)  # Gemini models + disk cache, from config.models
tree = ra.add_documents(Path("demo/sample.txt").read_text(encoding="utf-8"))
print(f"Built {len(tree.all_nodes)} nodes across {tree.num_layers + 1} layers")
ra.save("artifacts/indexes/cinderella")

question = "How did Cinderella reach her happy ending?"
result = ra.retrieve_nodes(question)
print("Retrieved layers:", result.layer_counts())
print("\nAnswer:\n" + ra.qa_model.answer_question(result.context, question))
