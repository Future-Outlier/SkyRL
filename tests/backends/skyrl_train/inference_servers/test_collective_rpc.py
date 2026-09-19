"""Generic RPC fan-out and native wake seams without a Ray cluster or GPU."""

import builtins
from unittest.mock import AsyncMock, call

import pytest

from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)


def _client(*, uses_isoexec=False):
    return RemoteInferenceClient(
        proxy_url="http://unused", server_urls=["http://a", "http://b"], data_parallel_size=1, uses_isoexec=uses_isoexec
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kwargs", [None, {}, {"tags": ["kv_cache"]}])
async def test_collective_rpc_fans_out_and_preserves_responses(kwargs):
    client = _client()
    responses = {url: {"status": 200, "body": {"results": [url]}} for url in client.server_urls}

    async def respond(url, endpoint, payload, method, params):
        return url, responses[url]

    client._call_server = AsyncMock(side_effect=respond)
    actual = await client.collective_rpc("extension_method", kwargs)
    payload = {"method": "extension_method"}
    if kwargs is not None:
        payload["kwargs"] = kwargs
    assert client._call_server.await_args_list == [
        call(url, "/collective_rpc", payload, "POST", None) for url in client.server_urls
    ]
    for url in client.server_urls:
        assert actual[url] is responses[url]


@pytest.mark.asyncio
async def test_collective_rpc_propagates_partial_worker_failure():
    client = _client()

    async def respond(url, *args):
        if url == "http://b":
            raise RuntimeError("worker refused")
        return url, {"status": 200, "body": {"results": []}}

    client._call_server = AsyncMock(side_effect=respond)
    with pytest.raises(RuntimeError, match="worker refused"):
        await client.collective_rpc("extension_method")
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


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["wake_up", "wake_for_weight_sync"])
async def test_wake_failure_never_runs_package_verifier(monkeypatch, method):
    adapter = pytest.importorskip("isoexec.integrations.skyrl.inference")
    verify = AsyncMock()
    monkeypatch.setattr(adapter, "verify_after_wake", verify)
    client = _client(uses_isoexec=True)
    client._call_all_servers = AsyncMock(side_effect=RuntimeError("native wake failed"))
    with pytest.raises(RuntimeError, match="native wake failed"):
        await getattr(client, method)(tags=["kv_cache"])
    verify.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["wake_up", "wake_for_weight_sync"])
async def test_package_verification_failure_aborts_native_wake_result(method):
    pytest.importorskip("isoexec.integrations.skyrl.inference")
    client = _client(uses_isoexec=True)
    client._call_all_servers = AsyncMock(side_effect=[{"native": "ok"}, RuntimeError("bad receipt")])
    with pytest.raises(RuntimeError, match="bad receipt"):
        await getattr(client, method)(tags=["kv_cache"])
    assert client._call_all_servers.await_count == 2
