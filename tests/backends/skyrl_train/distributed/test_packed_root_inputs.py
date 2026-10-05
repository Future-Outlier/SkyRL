"""Production packed-root preprocessing with native group/type stubs on CPU.

The SkyRL capability helper, config accessor and preprocessor are real code.
These checks do not execute Megatron scheduling or distributed collectives.
"""

import importlib
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from skyrl.backends.skyrl_train.distributed.megatron.packing_utils import (
    should_pack_root_inputs,
)


@pytest.fixture
def production_preprocessor(monkeypatch: pytest.MonkeyPatch):
    names = (
        "megatron",
        "megatron.core",
        "megatron.core.parallel_state",
        "megatron.core.distributed",
        "megatron.core.optimizer",
        "megatron.core.packed_seq_params",
        "megatron.core.transformer",
        "megatron.core.transformer.module",
        "megatron.core.transformer.moe",
        "megatron.core.transformer.moe.moe_utils",
        "megatron.core.utils",
    )
    modules = {name: ModuleType(name) for name in names}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    setattr(modules["megatron.core"], "parallel_state", modules["megatron.core.parallel_state"])
    setattr(modules["megatron.core.distributed"], "DistributedDataParallel", nn.Module)
    setattr(modules["megatron.core.optimizer"], "ChainedOptimizer", nn.Module)
    setattr(modules["megatron.core.packed_seq_params"], "PackedSeqParams", SimpleNamespace)
    setattr(modules["megatron.core.transformer.module"], "Float16Module", nn.Module)
    for name in (
        "clear_aux_losses_tracker",
        "get_moe_layer_wise_logging_tracker",
        "reduce_aux_losses_tracker_across_ranks",
    ):
        setattr(modules["megatron.core.transformer.moe.moe_utils"], name, lambda *args, **kwargs: None)
    for name, value in (
        ("get_tensor_model_parallel_world_size", 2),
        ("get_context_parallel_world_size", 1),
        ("get_context_parallel_rank", 0),
    ):
        setattr(modules["megatron.core.parallel_state"], name, lambda value=value: value)

    def config_attribute(model: Any, attr: str, *, allow_none: bool = True) -> Any:
        assert attr == "config" and allow_none is False
        return getattr(model, attr)

    setattr(modules["megatron.core.utils"], "get_attr_wrapped_model", config_attribute)
    setattr(modules["megatron.core.utils"], "unwrap_model", lambda model: model)
    name = "skyrl.backends.skyrl_train.distributed.megatron.megatron_utils"
    parent = importlib.import_module(name.rpartition(".")[0])
    monkeypatch.setattr(parent, "megatron_utils", None, raising=False)
    # Reload only this pure utility module, so its native imports cannot retain
    # another test's group stubs. The original module is restored at teardown.
    monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module(name)
    try:
        yield module
    finally:
        sys.modules.pop(name, None)


@pytest.mark.parametrize(
    ("pp_rank", "capability", "is_vlm", "packed"),
    (
        (0, False, False, True),
        (1, True, False, True),
        (2, True, False, True),
        (2, None, False, False),
        (2, None, True, True),
    ),
    ids=("legacy-first", "later-rank1", "later-rank2", "legacy-later", "vision-language-later"),
)
def test_production_preprocessor_preserves_packed_roots_and_legacy_control(
    production_preprocessor: Any,
    pp_rank: int,
    capability: bool | None,
    is_vlm: bool,
    packed: bool,
):
    config = SimpleNamespace()
    if capability is not None:
        config.requires_packed_root_inputs = capability
    # Native DDP stores the stage's same config object. The real SkyRL accessor
    # delegates the "config", allow_none=False read to the native API above.
    stage = SimpleNamespace(config=config)
    model = SimpleNamespace(module=stage, config=config)
    carrier = production_preprocessor.get_model_config(model)
    assert carrier is config
    tokens = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]], dtype=torch.int64)
    mask = torch.tensor([[True, True, True, False, False], [True, True, False, False, False]])
    result, params = production_preprocessor.preprocess_packed_seqs(
        tokens,
        mask,
        pre_process=should_pack_root_inputs(
            is_pipeline_first_stage=pp_rank == 0,
            is_vlm=is_vlm,
            requires_packed_root_inputs=getattr(carrier, "requires_packed_root_inputs", False),
        ),
    )
    assert params.qkv_format == "thd"
    assert params.cu_seqlens_q.dtype == torch.int32
    assert params.cu_seqlens_q.tolist() == params.cu_seqlens_q_padded.tolist() == [0, 4, 6]
    assert params.cu_seqlens_kv.tolist() == params.cu_seqlens_kv_padded.tolist() == [0, 4, 6]
    assert params.max_seqlen_q == params.max_seqlen_kv == 4 and params.total_tokens == 6
    if packed:
        assert tuple(result.shape) == (1, 6) and result.dtype == tokens.dtype
        assert result.tolist() == [[1, 2, 3, 0, 4, 5]]
    else:
        assert result is tokens and tuple(result.shape) == (2, 5)
    assert tokens.tolist() == [[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]]
