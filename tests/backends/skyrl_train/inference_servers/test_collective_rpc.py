"""Native weight-sync wake fan-out without a Ray cluster or GPU."""

import builtins
from unittest.mock import AsyncMock, call

import pytest

from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)


def _client():
    return RemoteInferenceClient(proxy_url="http://unused", server_urls=["http://a", "http://b"], data_parallel_size=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("tags", [["weights"], ["kv_cache"], ["weights", "kv_cache"]])
async def test_weight_sync_wake_fans_out_and_preserves_responses(tags):
    client = _client()
    responses = {url: {"status": 200, "body": {"results": [url]}} for url in client.server_urls}

    async def respond(url, endpoint, payload, method, params):
        return url, responses[url]

    client._call_server = AsyncMock(side_effect=respond)
    actual = await client.wake_for_weight_sync(tags)
    payload = {"method": "skyrl_wake_for_weight_sync", "kwargs": {"tags": tags}}
    assert client._call_server.await_args_list == [
        call(url, "/collective_rpc", payload, "POST", None) for url in client.server_urls
    ]
    for url in client.server_urls:
        assert actual[url] is responses[url]


@pytest.mark.asyncio
async def test_weight_sync_wake_propagates_partial_worker_failure():
    client = _client()

    async def respond(url, *args):
        if url == "http://b":
            raise RuntimeError("worker refused")
        return url, {"status": 200, "body": {"results": []}}

    client._call_server = AsyncMock(side_effect=respond)
    with pytest.raises(RuntimeError, match="worker refused"):
        await client.wake_for_weight_sync(["kv_cache"])
    assert client._call_server.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["wake_up", "wake_for_weight_sync"])
async def test_default_wake_does_not_import_isoexec(monkeypatch, method):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "isoexec" or name.startswith("isoexec."):
            raise AssertionError("default wake must not import IsoExec")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    client = _client()
    result = {"native": "response"}
    client._call_all_servers = AsyncMock(return_value=result)
    assert await getattr(client, method)(tags=["kv_cache"]) is result
    assert client._call_all_servers.await_count == 1
