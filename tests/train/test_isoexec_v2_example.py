"""The shared example resolves real HF configs through the V2 driver boundary."""

import json
import runpy
import shlex
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from skyrl.train.config import SkyRLTrainConfig

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/train/isoexec_v2/run.py"
example = runpy.run_path(str(EXAMPLE))
MODELS, training_overrides = example["MODELS"], example["training_overrides"]


@pytest.mark.parametrize("name", MODELS)
def test_checkpoint_config_selects_the_shared_architecture(name, tmp_path):
    pytest.importorskip("isoexec.model.v2")
    from isoexec.integrations.skyrl.full_distribution_config import (
        validate_full_distribution_config,
    )
    from isoexec.integrations.skyrl.model_folder import (
        prepare_checkpoints,
        source_config,
    )

    config_package = MODELS[name][1] + "." + name
    raw = json.loads(files(config_package).joinpath("config.json").read_text())
    checkpoint = tmp_path / "private checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(raw))
    # Artifact resolution checks the header; this test constructs only on meta.
    save_file({"probe": torch.ones(1)}, str(checkpoint / "model.safetensors"))
    cfg = SkyRLTrainConfig.from_cli_overrides(
        training_overrides(name, tmp_path / "data", tmp_path / "run", checkpoint=str(checkpoint))
    )
    validate_full_distribution_config(cfg)
    prepare_checkpoints(cfg)
    selected = source_config(cfg.isoexec_model, cfg.trainer, str(checkpoint))
    assert selected is not None
    source, effective = selected
    assert effective == raw
    assert source.package == MODELS[name][1]
    model = source.build_meta(effective)
    assert type(model).__name__ in raw["architectures"]
    assert all(parameter.is_meta for parameter in model.parameters())
    engine = cfg.generator.inference_engine
    assert not engine.enforce_eager
    assert engine.engine_init_kwargs["compilation_config"] == {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}
    assert engine.engine_init_kwargs["max_model_len"] == 2048
    assert cfg.trainer.max_prompt_length == 512
    assert cfg.generator.sampling_params.max_generate_length == 1024


def test_dry_run_keeps_paths_and_native_overrides_without_starting_training(tmp_path):
    data = tmp_path / "data with spaces"
    result = subprocess.run(
        [sys.executable, str(EXAMPLE), "--dry-run", "--data-dir", str(data), "trainer.max_training_steps=2"],
        check=True,
        capture_output=True,
        text=True,
    )
    command = shlex.split(result.stdout)
    assert command[:3] == [
        "env",
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV=0",
        "SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S=1200",
    ]
    assert command[3:6] == [sys.executable, "-m", "skyrl.train.entrypoints.main_base"]
    cfg = SkyRLTrainConfig.from_cli_overrides(command[6:])
    assert cfg.data.train_data == [str(data / "train.parquet")]
    assert cfg.trainer.max_training_steps == 2
    assert cfg.isoexec_model is not None
    assert cfg.isoexec_model["revision"] == MODELS["qwen3_5_0_8b"][2]
    assert cfg.isoexec_model["source"]["package"] == "isoexec.models.v2.qwen3_5"
    assert not any("model_factory" in argument for argument in command)
