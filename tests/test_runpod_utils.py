from __future__ import annotations


def test_rename_instance_uses_graphql_and_succeeds(monkeypatch):
    import runpod_service.runpod_utils as utils

    # Set required env
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-123")
    monkeypatch.setenv("RUNPOD_API_KEY", "key")

    def fake_post(url, json=None, timeout=0):  # type: ignore[no-redef]
        class R:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "data": {
                        "podRename": {
                            "id": "pod-123",
                            "name": json["variables"]["input"]["name"],
                        }
                    }
                }

        return R()

    monkeypatch.setattr(utils.requests, "post", fake_post)
    ok = utils.rename_instance("new-name")
    assert ok is True


def test_get_instance_name_queries_graphql(monkeypatch):
    import runpod_service.runpod_utils as utils

    # Set required env
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-123")
    monkeypatch.setenv("RUNPOD_API_KEY", "key")

    def fake_post(url, json=None, timeout=0):  # type: ignore[no-redef]
        class R:
            def raise_for_status(self):
                return None

            def json(self):
                return {"data": {"pod": {"id": "pod-123", "name": "my-pod"}}}

        return R()

    monkeypatch.setattr(utils.requests, "post", fake_post)
    name = utils.get_instance_name()
    assert name == "my-pod"


def test_stop_runpod_uses_sdk_and_finishes_wandb(monkeypatch):
    import runpod_service.runpod_utils as utils

    # Set required env
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-123")
    monkeypatch.setenv("RUNPOD_API_KEY", "key")

    # Provide SDK stop_pod and track calls
    calls = {"stopped": [], "finished": 0}

    class RP:
        api_key = None

        @staticmethod
        def stop_pod(pod_id):
            calls["stopped"].append(pod_id)

    monkeypatch.setattr(utils, "runpod", RP)

    class WBWrapper:
        @staticmethod
        def finish():
            calls["finished"] += 1

    class WB:
        wrapper = WBWrapper()

    monkeypatch.setattr(utils, "wandb", WB)

    ok = utils.stop_runpod()
    assert ok is True
    assert calls["stopped"] == ["pod-123"]
    assert calls["finished"] == 1
