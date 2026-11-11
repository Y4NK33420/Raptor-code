"""Live smoke test against Gemini. Run with: pytest -m live"""

from pathlib import Path

import pytest

from raptor import RaptorConfig, RetrievalAugmentation
from raptor.settings import get_settings

pytestmark = pytest.mark.live

SAMPLE = Path(__file__).resolve().parents[1] / "demo" / "sample.txt"


@pytest.fixture(autouse=True)
def _need_key():
    if get_settings().gemini_api_key is None:
        pytest.skip("GEMINI_API_KEY not set")


def test_build_and_answer_cinderella(tmp_path):
    cfg = RaptorConfig()
    cfg.chunking.max_tokens = 150
    cfg.builder.max_layers = 3
    ra = RetrievalAugmentation(cfg)  # Gemini models with the disk cache
    tree = ra.add_documents(SAMPLE.read_text(encoding="utf-8"))
    assert tree.num_layers >= 1
    ra.save(tmp_path / "cinderella")

    result = ra.retrieve_nodes("How did Cinderella reach her happy ending?")
    assert result.nodes and any(r.layer >= 1 for r in result.nodes)
    answer = ra.answer_question("Who helped Cinderella go to the festival?")
    assert len(answer) > 20
