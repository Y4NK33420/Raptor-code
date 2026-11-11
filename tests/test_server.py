"""API tests for ``raptor.server`` with offline fake models and fixture data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from raptor.server.app import create_app, parse_option  # noqa: E402
from raptor.server.config import ServerSettings  # noqa: E402
from raptor.server.fakes import fake_bundle  # noqa: E402
from raptor.server.runs import mcnemar_exact  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.split("\n\n"):
        name, data = None, []
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
        if name:
            events.append((name, json.loads("\n".join(data) or "null")))
    return events


def stream(client: TestClient, method: str, url: str, **kw) -> list[tuple[str, dict]]:
    with client.stream(method, url, **kw) as r:
        assert r.status_code == 200, r.read()
        assert r.headers["content-type"].startswith("text/event-stream")
        return sse("".join(r.iter_text()))


@pytest.fixture(scope="module")
def quality_dir(tmp_path_factory, document) -> Path:
    d = tmp_path_factory.mktemp("quality")
    rows = []
    for s in range(2):
        rows.append(
            {
                "article_id": f"A{s}",
                "set_unique_id": f"A{s}_SET",
                "title": f"Story {s}",
                "author": "Anon",
                "topic": "Testing",
                "article": document,
                "questions": [
                    {
                        "question_unique_id": f"A{s}_SET_{i}",
                        "question": "Where do the sailor and the ship meet?",
                        "options": [
                            "In the kitchen",
                            "At the harbor",
                            "On the moon",
                            "In a forest",
                        ],
                        "gold_label": 2,
                        "difficult": i % 2,
                    }
                    for i in range(3)
                ],
            }
        )
    (d / "QuALITY.v1.0.1.htmlstripped.dev").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )
    return d


@pytest.fixture(scope="module")
def settings(tmp_path_factory, quality_dir) -> ServerSettings:
    tmp = tmp_path_factory.mktemp("server")
    dist = tmp / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><div id=root></div>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    return ServerSettings(
        demo_dir=ROOT / "demo" / "trees",
        indexes_dir=tmp / "indexes",
        runs_dirs=[FIXTURES / "runs"],
        code_dir=tmp / "code",
        code_stub_repos=[FIXTURES / "code_repo"],
        quality_dir=quality_dir,
        derived_dir=tmp / "derived",
        web_dist=dist,
        projection_method="pca",
    )


@pytest.fixture(scope="module")
def client(settings) -> TestClient:
    with TestClient(create_app(settings, models=fake_bundle())) as c:
        yield c


# --------------------------------------------------------------------- meta/static
def test_info_and_presets(client):
    info = client.get("/api/info").json()
    assert info["fake_models"] is True
    assert info["code_backend"] in {"stub", "raptor.code"}
    presets = {p["id"]: p for p in client.get("/api/presets").json()}
    assert {"balanced", "quick", "fine", "shallow"} <= set(presets)
    assert presets["quick"]["config"]["chunking"]["max_tokens"] == 250


def test_frontend_is_served_with_spa_fallback(client):
    assert "id=root" in client.get("/").text
    assert "id=root" in client.get("/explore/some-index").text
    assert client.get("/assets/app.js").text == "console.log(1)"
    assert client.get("/api/does-not-exist").status_code == 404


def test_openapi_has_typed_models(client):
    spec = client.get("/openapi.json").json()
    for name in ("IndexInfo", "TreeData", "RunCompare", "CodeSymbol", "EventCatalog"):
        assert name in spec["components"]["schemas"]


# ------------------------------------------------------------------------- indexes
def test_lists_demo_indexes(client):
    ids = {i["id"]: i for i in client.get("/api/indexes").json()}
    assert "cinderella_gemini" in ids
    cin = ids["cinderella_gemini"]
    assert cin["origin"] == "demo" and cin["num_nodes"] == sum(cin["layer_counts"].values())
    assert client.get("/api/indexes/nope").status_code == 404
    assert client.get("/api/indexes/..%2F..%2Fetc").status_code == 404


def test_tree_node_and_projection(client):
    tree = client.get("/api/indexes/cinderella_gemini/tree").json()
    assert len(tree["nodes"]) == 27
    by_id = {n["id"]: n for n in tree["nodes"]}
    edge = tree["edges"][0]
    assert edge["target"] in by_id[edge["source"]]["children"]
    assert edge["source"] in by_id[edge["target"]]["parents"]

    root = next(n for n in tree["nodes"] if n["layer"] == tree["num_layers"])
    node = client.get(f"/api/indexes/cinderella_gemini/nodes/{root['id']}").json()
    assert node["children"] == root["children"] and node["text"]
    leaf = client.get(f"/api/indexes/cinderella_gemini/nodes/{root['children'][0]}").json()
    assert root["id"] in leaf["parents"] and leaf["id"] not in leaf["siblings"]
    assert client.get("/api/indexes/cinderella_gemini/nodes/9999").status_code == 404

    proj = client.get("/api/indexes/cinderella_gemini/projection").json()
    assert len(proj["points"]) == 27
    assert all(0 <= p["x"] <= 1 and 0 <= p["y"] <= 1 for p in proj["points"])
    leaf_pt = next(p for p in proj["points"] if p["layer"] == 0)
    assert leaf_pt["cluster"] in leaf_pt["parents"]
    # cached on disk the second time
    assert client.get("/api/indexes/cinderella_gemini/projection").json() == proj


def test_query_streams_retrieval_then_tokens(client):
    events = stream(
        client,
        "POST",
        "/api/indexes/cinderella_gemini/query",
        json={"question": "Who helps Cinderella go to the ball?", "top_k": 5},
    )
    names = [e for e, _ in events]
    assert names[0] == "retrieval" and names[-1] == "done"
    assert "token" in names and names.index("answer") > names.index("token")
    retrieval = events[0][1]
    assert 1 <= len(retrieval["nodes"]) <= 5
    assert sum(retrieval["layer_counts"].values()) == len(retrieval["nodes"])
    scores = [n["score"] for n in retrieval["nodes"]]
    assert scores == sorted(scores, reverse=True)
    answer = next(d for e, d in events if e == "answer")["answer"]
    assert answer == "".join(d["text"] for e, d in events if e == "token")


def test_compare_free_text_uses_same_budget(client):
    events = stream(
        client,
        "POST",
        "/api/indexes/cinderella_gemini/compare",
        json={"question": "What did the bird throw down?", "top_k": 6, "max_tokens": 900},
    )
    assert events[0][0] == "question" and events[-1][0] == "done"
    retrieval = {d["side"]: d for e, d in events if e == "retrieval"}
    assert set(retrieval) == {"flat", "raptor"}
    assert set(retrieval["flat"]["layer_counts"]) == {"0"}  # flat = leaves only
    for side in retrieval.values():
        assert len(side["nodes"]) <= 6 and side["context_tokens"] <= 900
    assert {d["side"] for e, d in events if e == "answer"} == {"flat", "raptor"}


def test_build_query_and_delete_index(client, document):
    r = client.post(
        "/api/indexes",
        data={
            "source_type": "text",
            "text": document,
            "name": "Topics doc",
            "preset": "balanced",
            "config": json.dumps(
                {"chunking": {"max_tokens": 40}, "clustering": {"reduction_dimension": 5}}
            ),
        },
    )
    assert r.status_code == 202, r.text
    job_id, index_id = r.json()["job_id"], r.json()["index_id"]
    events = stream(client, "GET", f"/api/jobs/{job_id}/events")
    names = [e for e, _ in events]
    assert names[0] == "queued" and names[-1] == "complete", events[-1]
    chunks = next(d for e, d in events if e == "chunks")
    clusters = [d for e, d in events if e == "clusters"]
    summaries = [d for e, d in events if e == "summary"]
    progress = [d["stage"] for e, d in events if e == "progress"]
    assert progress[0] == "chunking" and "summarizing" in progress
    assert chunks["count"] > 10
    # every summary is matched to a parent announced by a clusters event
    announced = {p["id"] for c in clusters for p in c["parents"]}
    assert summaries and all(s["id"] in announced for s in summaries)
    assert clusters[0]["parents"][0]["children"][0] < chunks["count"]
    complete = events[-1][1]
    assert complete["index_id"] == index_id

    info = client.get(f"/api/indexes/{index_id}").json()
    assert info["origin"] == "user" and info["name"] == "Topics doc"
    assert info["num_nodes"] == complete["num_nodes"]
    tree = client.get(f"/api/indexes/{index_id}/tree").json()
    assert {n["id"] for n in tree["nodes"]} >= announced & {n["id"] for n in tree["nodes"]}
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "done"

    # replay from a Last-Event-ID only sends later events
    replay = stream(client, "GET", f"/api/jobs/{job_id}/events", headers={"Last-Event-ID": "2"})
    assert len(replay) == len(events) - 3

    q = stream(client, "POST", f"/api/indexes/{index_id}/query", json={"question": "rocket orbit"})
    assert q[0][0] == "retrieval" and q[0][1]["nodes"]

    assert client.delete("/api/indexes/cinderella_gemini").status_code == 403
    assert client.delete(f"/api/indexes/{index_id}").status_code == 204
    assert client.get(f"/api/indexes/{index_id}").status_code == 404


def test_build_validation(client):
    assert client.post("/api/indexes", data={"source_type": "text", "text": " "}).status_code == 422
    bad = client.post(
        "/api/indexes",
        data={"source_type": "file"},
        files={"file": ("x.exe", b"MZ....", "application/octet-stream")},
    )
    assert bad.status_code == 415
    assert (
        client.post(
            "/api/indexes", data={"source_type": "text", "text": "x" * 100, "preset": "nope"}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/indexes", data={"source_type": "quality", "set_id": "missing"}
        ).status_code
        == 404
    )
    ok = client.post(
        "/api/indexes",
        data={"source_type": "file", "preset": "shallow"},
        files={
            "file": (
                "notes.md",
                b"# Notes\n\nOne. Two. Three. Four. Five sentences here.",
                "text/markdown",
            )
        },
    )
    assert ok.status_code == 202
    events = stream(client, "GET", f"/api/jobs/{ok.json()['job_id']}/events")
    assert events[-1][0] == "complete"
    info = client.get(f"/api/indexes/{ok.json()['index_id']}").json()
    assert info["name"] == "notes" and info["source"]["filename"] == "notes.md"


# ------------------------------------------------------------------------- quality
def test_quality_stories_and_mcq_compare(client):
    stories = client.get("/api/quality/stories").json()
    assert [s["set_id"] for s in stories] == ["A0_SET", "A1_SET"]
    detail = client.get("/api/quality/stories/A0_SET").json()
    assert len(detail["questions"]) == 3 and detail["article"]
    assert client.get("/api/quality/stories/zzz").status_code == 404

    r = client.post(
        "/api/indexes", data={"source_type": "quality", "set_id": "A0_SET", "preset": "quick"}
    )
    index_id = r.json()["index_id"]
    assert stream(client, "GET", f"/api/jobs/{r.json()['job_id']}/events")[-1][0] == "complete"
    assert index_id in client.get("/api/quality/stories").json()[0]["index_ids"]

    events = stream(
        client,
        "POST",
        f"/api/indexes/{index_id}/compare",
        json={"question_id": "A0_SET_1", "set_id": "A0_SET", "top_k": 8},
    )
    q = events[0][1]
    assert q["gold"] == 2 and len(q["options"]) == 4 and q["difficult"] is True
    answers = {d["side"]: d for e, d in events if e == "answer"}
    for a in answers.values():
        assert a["predicted"] in {1, 2, 3, 4}
        assert a["correct"] == (a["predicted"] == 2)
    assert (
        client.post(f"/api/indexes/{index_id}/compare", json={"question_id": "nope"}).status_code
        == 404
    )


def test_parse_option():
    assert parse_option("Option 3. Because…", 4) == 3
    assert parse_option("I pick option (2)", 4) == 2
    assert parse_option("Option 9", 4) is None
    assert parse_option("no idea", 4) is None


# ---------------------------------------------------------------------------- runs
def test_runs_list_detail_compare(client):
    runs = client.get("/api/runs").json()
    assert len(runs) >= 5
    assert all(r["origin"] == "fixture" for r in runs)
    flat = next(r for r in runs if r["method"] == "flat")
    raptor = next(r for r in runs if r["name"] == "raptor-baseline")
    assert flat["ci95"][0] <= flat["accuracy"] <= flat["ci95"][1]

    detail = client.get(f"/api/runs/{raptor['run_id']}").json()
    assert len(detail["results"]) == raptor["n_questions"]
    assert detail["config"]["method"] == "raptor" and "chunking" in detail["config_yaml"]

    cmp = client.get(
        "/api/runs/compare", params={"a": flat["run_id"], "b": raptor["run_id"]}
    ).json()
    n = cmp["fixed"] + cmp["broken"] + cmp["both_right"] + cmp["both_wrong"]
    assert n == cmp["n_paired"] == len(cmp["questions"])
    assert cmp["delta"] == pytest.approx(cmp["accuracy_b"] - cmp["accuracy_a"], abs=1e-3)
    assert 0 <= cmp["mcnemar_p"] <= 1
    assert cmp["delta_ci95"][0] <= cmp["delta"] <= cmp["delta_ci95"][1]
    fixed = [q for q in cmp["questions"] if q["status"] == "fixed"]
    assert all(q["b_predicted"] == q["gold"] != q["a_predicted"] for q in fixed)
    assert client.get("/api/runs/nope").status_code == 404
    assert (
        client.get("/api/runs/compare", params={"a": "nope", "b": raptor["run_id"]}).status_code
        == 404
    )


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 1024)
    assert mcnemar_exact(8, 2) == pytest.approx(0.109375)


# ---------------------------------------------------------------------------- code
def test_code_explorer_endpoints(client):
    repos = client.get("/api/code").json()
    assert repos and repos[0]["n_files"] >= 5
    cid = repos[0]["id"]
    files = {f["path"]: f for f in client.get(f"/api/code/{cid}/files").json()}
    assert "shop/pricing.py" in files
    syms = {s["symbol"]: s for s in files["shop/pricing.py"]["symbols"]}
    assert "sales_tax" in syms and syms["sales_tax"]["start_line"] > 1

    src = client.get(f"/api/code/{cid}/file", params={"path": "shop/pricing.py"}).json()
    assert src["language"] == "python" and "def sales_tax" in src["text"]
    for bad in ("../../../pyproject.toml", "shop/../../README.md", "nope.py"):
        assert client.get(f"/api/code/{cid}/file", params={"path": bad}).status_code == 404

    quote_id = syms["quote"]["id"]
    sym = client.get(f"/api/code/{cid}/symbol/{quote_id}").json()
    assert sym["hit"]["symbol"] == "quote"
    assert {c["symbol"] for c in sym["callees"]} >= {"sales_tax", "coupon_discount"}
    assert any(c["symbol"] == "checkout" for c in sym["callers"])

    graph = client.get(f"/api/code/{cid}/graph").json()
    node_ids = {n["id"] for n in graph["nodes"]}
    assert graph["edges"] and all(
        e["src"] in node_ids and e["dst"] in node_ids for e in graph["edges"]
    )
    calls_only = client.get(f"/api/code/{cid}/graph", params={"kinds": "calls"}).json()
    assert {e["kind"] for e in calls_only["edges"]} == {"calls"}

    events = stream(
        client, "POST", f"/api/code/{cid}/ask", json={"question": "How is sales tax computed?"}
    )
    names = [e for e, _ in events]
    assert names[0] == "token" and "citations" in names and names[-1] == "done"
    cites = next(d for e, d in events if e == "citations")["citations"]
    assert cites and all(c["path"] in files and c["start_line"] >= 1 for c in cites)
    assert any(c["symbol"] == "sales_tax" for c in cites)
    assert client.get("/api/code/missing/files").status_code == 404
