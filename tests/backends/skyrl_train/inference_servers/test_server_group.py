from skyrl.backends.skyrl_train.inference_servers.server_group import ServerGroup


def test_server_group_forwards_extra_env_vars():
    group = object.__new__(ServerGroup)
    group._use_expandable_segments = False
    group._extra_env_vars = {
        "VLLM_PLUGINS": "isoexec",
        "VLLM_ISOEXEC_REGISTERED": "1",
    }

    assert group._engine_runtime_env() == {"env_vars": group._extra_env_vars}
