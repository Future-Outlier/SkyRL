import pytest

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import (
    prepare_runtime_environment,
    validate_logprob_comparison,
)
from tests.train.util import example_dummy_config


def _full_mode_config():
    cfg = example_dummy_config()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.enable_isoexec = True
    cfg.trainer.rollout_logprob_comparison = "full"
    cfg.trainer.remove_microbatch_padding = True
    cfg.generator.inference_engine.logprob_output = "full"
    cfg.generator.batched = True
    return cfg


def test_isoexec_is_opt_in_and_preserves_default_runtime_environment():
    cfg = example_dummy_config()

    assert cfg.trainer.enable_isoexec is False
    assert "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES" not in prepare_runtime_environment(cfg)


def test_isoexec_runtime_uses_physical_gpu_namespace():
    cfg = example_dummy_config()
    cfg.trainer.enable_isoexec = True

    assert prepare_runtime_environment(cfg)["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] == "1"


def test_isoexec_runtime_forwards_isoexec_overrides_only_when_enabled(monkeypatch):
    monkeypatch.setenv("ISOEXEC_FULL_DISTRIBUTION_EVIDENCE_DIR", "/evidence/rows")
    monkeypatch.setenv("ISOEXEC", "1")
    cfg = example_dummy_config()

    assert "ISOEXEC_FULL_DISTRIBUTION_EVIDENCE_DIR" not in prepare_runtime_environment(cfg)

    cfg.trainer.enable_isoexec = True
    env_vars = prepare_runtime_environment(cfg)

    assert env_vars["ISOEXEC_FULL_DISTRIBUTION_EVIDENCE_DIR"] == "/evidence/rows"
    assert env_vars["ISOEXEC"] == "1"


def test_full_mode_fields_are_cli_overridable():
    cfg = SkyRLTrainConfig.from_cli_overrides(
        [
            "trainer.enable_isoexec=true",
            "trainer.rollout_logprob_comparison=full",
            "trainer.remove_microbatch_padding=true",
            "generator.inference_engine.logprob_output=full",
        ]
    )

    assert cfg.trainer.enable_isoexec is True
    assert cfg.trainer.rollout_logprob_comparison == "full"
    assert cfg.trainer.remove_microbatch_padding is True
    assert cfg.generator.inference_engine.logprob_output == "full"


def test_action_mode_preserves_evaluation_logprobs():
    cfg = example_dummy_config()
    assert cfg.generator.eval_sampling_params.logprobs == 1

    validate_logprob_comparison(cfg)

    assert cfg.generator.eval_sampling_params.logprobs == 1


def test_logprob_modes_must_match():
    cfg = example_dummy_config()
    cfg.trainer.rollout_logprob_comparison = "full"
    with pytest.raises(ValueError, match="must agree"):
        validate_logprob_comparison(cfg)


def _forbid_isoexec_import(monkeypatch):
    import builtins

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "isoexec" or name.startswith("isoexec."):
            raise AssertionError("The disabled integration must not import IsoExec")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def test_action_mode_does_not_import_isoexec(monkeypatch):
    _forbid_isoexec_import(monkeypatch)
    validate_logprob_comparison(example_dummy_config())


def test_full_mode_requires_opt_in_before_importing_the_package(monkeypatch):
    _forbid_isoexec_import(monkeypatch)
    cfg = _full_mode_config()
    cfg.trainer.enable_isoexec = False
    with pytest.raises(ValueError, match="trainer.enable_isoexec=true"):
        validate_logprob_comparison(cfg)


def test_full_mode_delegates_policy_to_the_installed_package(monkeypatch):
    import sys
    from types import SimpleNamespace

    calls = []
    monkeypatch.setitem(
        sys.modules,
        "isoexec.integrations.skyrl.full_distribution_config",
        SimpleNamespace(validate_full_distribution_config=calls.append),
    )
    cfg = _full_mode_config()
    validate_logprob_comparison(cfg)
    assert calls == [cfg]


def test_package_refusal_propagates_from_the_native_validation_seam(monkeypatch):
    import sys
    from types import SimpleNamespace

    def refuse(cfg):
        raise ValueError("package policy refusal")

    monkeypatch.setitem(
        sys.modules,
        "isoexec.integrations.skyrl.full_distribution_config",
        SimpleNamespace(validate_full_distribution_config=refuse),
    )
    with pytest.raises(ValueError, match="package policy refusal"):
        validate_logprob_comparison(_full_mode_config())
