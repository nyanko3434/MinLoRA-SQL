"""requirements-ci.txt is requirements.txt without the CUDA wheels. Keep the two from drifting."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUDA_ONLY = ("nvidia-", "triton==")


def _pins(name):
    lines = (ROOT / name).read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith(("#", "--"))]


def test_ci_requirements_match_the_pinned_ones():
    full, ci = _pins("requirements.txt"), _pins("requirements-ci.txt")
    torch_full = [p for p in full if p.startswith("torch==")]
    torch_ci = [p for p in ci if p.startswith("torch==")]
    assert len(torch_full) == len(torch_ci) == 1
    # Same torch release, different build (+cu121 locally, +cpu in CI).
    assert torch_full[0].split("+")[0] == torch_ci[0].split("+")[0]
    assert torch_ci[0].endswith("+cpu")

    expected = [p for p in full if not p.startswith(CUDA_ONLY) and not p.startswith("torch==")]
    assert [p for p in ci if not p.startswith("torch==")] == expected
