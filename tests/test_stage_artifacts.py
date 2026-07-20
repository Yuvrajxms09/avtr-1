import json
from pathlib import Path

from safetensors.torch import load_file
import torch

from avtr1_renderer.stage_artifacts import StageArtifactSession, record_geometry_stage


def test_stage_artifacts_preserve_chunk_boundaries_and_manifest(tmp_path: Path) -> None:
    output = tmp_path / "stages"
    source = torch.randn(1, 21, 3)
    first = torch.randn(5, 21, 3)
    second = torch.randn(5, 21, 3)
    with StageArtifactSession(
        output,
        metadata={"motion_fingerprint_sha256": "abc"},
        capture_pixels=False,
    ):
        record_geometry_stage("source_keypoints", source)
        record_geometry_stage("driving_raw", first)
        record_geometry_stage("source_keypoints", source)
        record_geometry_stage("driving_raw", second)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    tensors = load_file(output / "geometry.safetensors")
    assert manifest["metadata"]["motion_fingerprint_sha256"] == "abc"
    assert manifest["geometry_stages"]["driving_raw"]["chunks"] == 2
    assert manifest["geometry_stages"]["driving_raw"]["chunk_lengths"] == [5, 5]
    assert manifest["geometry_stages"]["driving_raw"]["chunk_lengths_sha256"]
    torch.testing.assert_close(tensors["driving_raw"], torch.cat([first, second]))
    assert tensors["driving_raw__chunk_lengths"].tolist() == [5, 5]
