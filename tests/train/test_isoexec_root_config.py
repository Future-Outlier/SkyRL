"""Keep IsoExec user settings at the root and reject unsupported entrypoints."""

import pytest

from skyrl.train.config import SkyRLTrainConfig


@pytest.mark.parametrize(
    "key", ["enable_isoexec", "isoexec_model", "trainer.enable_isoexec", "trainer.isoexec_model", "trainer.isoexec"]
)
def test_old_or_nested_fields_are_rejected(key):
    with pytest.raises(ValueError, match="Invalid fields"):
        SkyRLTrainConfig.from_cli_overrides([key + "=true"])


@pytest.mark.parametrize("overrides", [[], ["isoexec=null"]])
def test_absent_or_null_isoexec_disables_integration(overrides):
    assert SkyRLTrainConfig.from_cli_overrides(overrides).isoexec is None


def test_cli_preserves_mapping_for_package_validation():
    cfg = SkyRLTrainConfig.from_cli_overrides(
        [
            "isoexec.model_package=isoexec.models.qwen3_5",
            "isoexec.full_logprob_comparision=true",
        ]
    )
    assert cfg.isoexec == {"model_package": "isoexec.models.qwen3_5", "full_logprob_comparision": True}
    # The adapter must reject this; the framework must not silently turn it off.
    assert SkyRLTrainConfig.from_cli_overrides(["isoexec={}"]).isoexec == {}


def test_tinker_refuses_isoexec_without_the_rl_comparison_lifecycle():
    backend = pytest.importorskip("skyrl.backends.skyrl_train_backend")
    overrides = backend.MegatronBackendOverrides(
        **{
            "isoexec": {"model_package": "isoexec.models.qwen3_5", "full_logprob_comparision": True},
        }
    )
    with pytest.raises(ValueError, match="Tinker backend is not supported"):
        backend._build_skyrl_train_config("model", overrides)
