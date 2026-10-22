import runpy
from pathlib import Path

import pytest

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import validate_logprob_comparison


def test_modal_example_enables_full_logprob_comparison(monkeypatch):
    """Check the CLI profile; native runtime validation requires Linux/CUDA dependencies."""
    pytest.importorskip("modal")
    pytest.importorskip("isoexec")
    monkeypatch.delenv("MODAL_ISOEXEC_ONLY", raising=False)
    example = Path(__file__).parents[2] / "examples/train_integrations/modal/main.py"
    namespace = runpy.run_path(str(example))
    assert namespace["isoexec_image"] is None
    assert namespace["run_isoexec_full_distribution"] is None
    command = namespace["isoexec_full_distribution_command"]()
    entrypoint = command.index("skyrl.train.entrypoints.main_base")
    overrides = command[entrypoint + 1 :]

    cfg = SkyRLTrainConfig.from_cli_overrides(overrides)
    validate_logprob_comparison(cfg)

    assert cfg.trainer.enable_isoexec is True
    assert cfg.trainer.rollout_logprob_comparison == "full"
    assert cfg.generator.inference_engine.logprob_output == "full"
    assert cfg.generator.eval_sampling_params.logprobs is None


def test_modal_example_registers_isoexec_function_only_when_selected(monkeypatch):
    pytest.importorskip("modal")
    monkeypatch.setenv("MODAL_ISOEXEC_ONLY", "1")
    example = Path(__file__).parents[2] / "examples/train_integrations/modal/main.py"

    namespace = runpy.run_path(str(example))

    assert namespace["isoexec_image"] is not None
    assert namespace["run_isoexec_full_distribution"] is not None
