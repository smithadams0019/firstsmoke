from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient
from helpers import build_app, exploding_analyzer, slow_analyzer


def run_to_completion(client: TestClient, payload: bytes, filename="a.png", params=None):
    """Submit and block until the job leaves the running state."""
    response = client.post(
        "/api/jobs",
        files={"file": (filename, payload, "image/png")},
        data={"params": json.dumps(params or {})},
    )
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    deadline = time.time() + 20
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never finished")


class TestHealthAndVersion:
    def test_healthz_is_cheap_and_names_the_product(self, client):
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "product": "demo"}

    def test_version_reports_opencv_5_git_sha_and_models(self, client):
        payload = client.get("/version").json()
        assert payload["opencv_version"].startswith("5."), "the entry requirement, visible"
        assert "git_sha" in payload
        assert payload["product"]["slug"] == "demo"
        assert "models" in payload
        assert payload["opencv_build"]["threads"] >= 1

    def test_version_surfaces_model_licences(self, tmp_path):
        from visioncore import YOLOX_TINY

        with TestClient(build_app(tmp_path=tmp_path, models={"detector": YOLOX_TINY})) as client:
            models = client.get("/version").json()["models"]
        assert models["detector"]["licence"] == "Apache-2.0"

    def test_config_gives_the_ui_what_it_needs_to_render_itself(self, client):
        payload = client.get("/api/config").json()
        assert payload["product"]["title"] == "Demo Product"
        assert payload["accent"] == "#ff6b35"
        assert ".png" in payload["accepts"]
        assert payload["params"][0]["name"] == "threshold"

    def test_every_response_carries_a_request_id(self, client):
        assert client.get("/healthz").headers["X-Request-Id"]

    def test_a_supplied_request_id_is_echoed_back(self, client):
        response = client.get("/healthz", headers={"X-Request-Id": "abc123"})
        assert response.headers["X-Request-Id"] == "abc123"


class TestUpload:
    def test_a_job_runs_and_returns_a_run_record(self, client, sample_png):
        job = run_to_completion(client, sample_png)
        assert job["status"] == "done"
        record = job["result"]
        assert record["product"] == "demo"
        assert record["metrics"]["contour_count"] == 2, "two drawn squares, known by construction"
        assert record["env"]["opencv_version"].startswith("5.")
        assert record["timings"]["total_ms"] > 0
        assert {s["name"] for s in record["timings"]["stages"]} >= {"decode", "threshold", "contours"}

    def test_evidence_is_attached_and_fetchable(self, client, sample_png):
        job = run_to_completion(client, sample_png)
        evidence = job["result"]["evidence"]
        assert len(evidence) == 1
        response = client.get(evidence[0]["uri"])
        assert response.status_code == 200
        assert len(response.content) > 100

    def test_missing_evidence_is_a_clean_404(self, client, sample_png):
        job = run_to_completion(client, sample_png)
        response = client.get(f"/api/jobs/{job['job_id']}/evidence/nope.jpg")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "NOT_FOUND"

    def test_evidence_path_traversal_is_not_possible(self, client, sample_png):
        job = run_to_completion(client, sample_png)
        response = client.get(
            f"/api/jobs/{job['job_id']}/evidence/..%2F..%2Fetc%2Fpasswd"
        )
        assert response.status_code in (400, 404)

    def test_params_reach_the_analyzer_and_can_produce_a_refusal(self, client, sample_png):
        job = run_to_completion(client, sample_png, params={"refuse": 1})
        assert job["status"] == "done", "a refusal is a result, not a failure"
        assert job["result"]["refused"] is True
        assert job["result"]["refusals"][0]["code"] == "TOO_OBLIQUE"

    def test_jobs_are_listable(self, client, sample_png):
        run_to_completion(client, sample_png)
        jobs = client.get("/api/jobs").json()["jobs"]
        assert len(jobs) == 1
        assert jobs[0]["status"] == "done"


class TestErrorShape:
    def test_unsupported_file_type_is_rejected_with_the_standard_shape(self, client):
        response = client.post("/api/jobs", files={"file": ("x.exe", b"MZ", "application/x-msdownload")})
        assert response.status_code == 415
        error = response.json()["error"]
        assert error["code"] == "UNSUPPORTED_MEDIA"
        assert error["request_id"]
        assert ".png" in error["details"]["accepted"]

    def test_oversize_upload_is_rejected(self, tmp_path):
        with TestClient(build_app(tmp_path=tmp_path, max_upload_bytes=1024)) as client:
            response = client.post(
                "/api/jobs", files={"file": ("big.png", b"x" * 4096, "image/png")}
            )
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "TOO_LARGE"

    def test_empty_upload_is_rejected(self, client):
        response = client.post("/api/jobs", files={"file": ("a.png", b"", "image/png")})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "BAD_REQUEST"

    def test_malformed_params_json_is_rejected_clearly(self, client, sample_png):
        response = client.post(
            "/api/jobs",
            files={"file": ("a.png", sample_png, "image/png")},
            data={"params": "{not json"},
        )
        assert response.status_code == 400
        assert "not valid JSON" in response.json()["error"]["message"]

    def test_params_must_be_an_object(self, client, sample_png):
        response = client.post(
            "/api/jobs",
            files={"file": ("a.png", sample_png, "image/png")},
            data={"params": "[1,2,3]"},
        )
        assert response.status_code == 400

    def test_unknown_job_is_a_404_with_the_standard_shape(self, client):
        response = client.get("/api/jobs/doesnotexist")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "NOT_FOUND"

    def test_an_analyzer_crash_becomes_a_rendered_error_not_a_stack_trace(self, tmp_path, sample_png):
        with TestClient(build_app(exploding_analyzer, tmp_path=tmp_path)) as client:
            job = run_to_completion(client, sample_png)
        assert job["status"] == "failed"
        assert job["error"]["code"] == "ANALYSIS_FAILED"
        assert "analyzer blew up" in job["error"]["message"]
        assert job["result"] is None


class TestProgressStream:
    def test_sse_stream_carries_progress_then_ends(self, tmp_path, sample_png):
        with TestClient(build_app(slow_analyzer, tmp_path=tmp_path)) as client:
            created = client.post("/api/jobs", files={"file": ("a.png", sample_png, "image/png")})
            job_id = created.json()["job_id"]
            events = []
            with client.stream("GET", f"/api/jobs/{job_id}/events") as stream:
                assert stream.headers["content-type"].startswith("text/event-stream")
                for line in stream.iter_lines():
                    # SSE sends "event: <type>" before its "data:" line, so break
                    # on the parsed payload or the last event is never collected.
                    if line.startswith("data:"):
                        events.append(json.loads(line[5:]))
                        if events[-1]["type"] == "end":
                            break
        types = [e["type"] for e in events]
        assert "progress" in types
        assert types[-1] == "end"
        percents = [e["percent"] for e in events if e["type"] == "progress"]
        assert percents == sorted(percents)
        assert any(e["type"] == "status" and e["status"] == "done" for e in events)

    def test_a_late_subscriber_gets_the_whole_history_replayed(self, client, sample_png):
        job = run_to_completion(client, sample_png)
        events = []
        with client.stream("GET", f"/api/jobs/{job['job_id']}/events") as stream:
            for line in stream.iter_lines():
                if line.startswith("data:"):
                    events.append(json.loads(line[5:]))
                    if events[-1]["type"] == "end":
                        break
        assert [e["type"] for e in events].count("progress") >= 2
        assert events[-1]["type"] == "end"

    def test_streaming_an_unknown_job_is_a_404(self, client):
        assert client.get("/api/jobs/nope/events").status_code == 404


class TestUiShell:
    def test_index_renders_with_the_product_substituted(self, client):
        html = client.get("/").text
        assert "Demo Product" in html
        assert "{{PRODUCT_TITLE}}" not in html
        assert "#ff6b35" in html

    def test_shell_assets_are_served(self, client):
        for path in ("/shell/css/theme.css", "/shell/css/shell.css",
                     "/shell/js/main.js", "/shell/js/api.js", "/shell/js/render.js",
                     "/shell/js/theme.js"):
            assert client.get(path).status_code == 200, path

    def test_theme_exposes_custom_properties_for_products_to_override(self, client):
        css = client.get("/shell/css/theme.css").text
        for token in ("--accent", "--bg", "--fg", "--font-display", "--space-4", "--radius"):
            assert token in css
        assert 'prefers-color-scheme: dark' in css
        assert ':root[data-theme="dark"]' in css

    def test_the_shell_has_no_build_step(self, client):
        js = client.get("/shell/js/main.js").text
        assert "import" in js
        assert "require(" not in js

    def test_a_product_can_override_the_index_and_serve_its_own_assets(self, tmp_path):
        static = tmp_path / "static"
        (static / "css").mkdir(parents=True)
        (static / "index.html").write_text("<h1>{{PRODUCT_TITLE}} custom</h1>")
        (static / "css" / "product.css").write_text(":root { --accent: #123456; }")
        with TestClient(build_app(tmp_path=tmp_path, static_dir=static)) as client:
            assert "Demo Product custom" in client.get("/").text
            assert client.get("/assets/css/product.css").status_code == 200
