"""Object registry: mutated variants (001..) load as their own shapes, alongside 000."""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wrappers" / "isaaclab"))
from envs import _object_registry as reg  # noqa: E402

# The VR dataset server only accepts these characters in case ids / file names.
VR_SEGMENT = re.compile(r"^[A-Za-z0-9_.-]+$")


@pytest.fixture
def objects_root(tmp_path, monkeypatch):
    monkeypatch.setattr(reg, "data_path", lambda *parts: tmp_path.joinpath(*parts))
    reg._ycb_shape_names_cached.cache_clear()
    yield tmp_path / "data" / "objects"
    reg._ycb_shape_names_cached.cache_clear()


def touch(root, split, family, *files):
    d = root / split / family
    d.mkdir(parents=True, exist_ok=True)
    for f in files:
        (d / f).touch()


def test_variant_name_round_trip():
    assert reg.variant_name("torus", 0) == "torus"
    assert reg.variant_name("ycb_065-a_cups", 3) == "ycb_065-a_cups.v003"
    for name in ["torus", "ycb_065-a_cups.v003", "ycb_006_mustard_bottle.v012"]:
        assert reg.variant_name(*reg.split_variant(name)) == name
        assert VR_SEGMENT.match(name)


def test_prim_names_unique_and_valid():
    names = [reg.variant_name(f, i) for f in ["torus", "ycb_065-a_cups"] for i in range(8)]
    prims = [reg.prim_name(n) for n in names]
    assert len(set(prims)) == len(prims)
    assert all(re.match(r"^[A-Za-z][A-Za-z0-9]*$", p) for p in prims)


def test_family_variants_lists_spawn_ready_instances(objects_root):
    # 004 was dropped by the graspability gate; 005 has no USD yet.
    touch(objects_root, "train", "torus", "000.usd", "001.usd", "002.usd", "003.usd", "005.obj")
    assert reg.family_variants("torus") == ["torus", "torus.v001", "torus.v002", "torus.v003"]
    assert reg.family_variants("torus", max_variants=2) == ["torus", "torus.v001"]
    assert reg.family_variants("torus", max_variants=1) == ["torus"]
    assert reg.family_variants("torus", max_variants=0) == reg.family_variants("torus")


def test_shape_asset_resolves_variant_files_and_split(objects_root):
    (objects_root / "eval").mkdir(parents=True)
    (objects_root / "eval" / "manifest.json").write_text('[{"family": "ycb_011_banana"}]')
    touch(objects_root, "eval", "ycb_011_banana", "000.usd", "002.usd")

    assert reg.shape_asset("torus.v002", "_pc.npy") == objects_root / "train" / "torus" / "002_pc.npy"
    assert reg.shape_asset("torus", ".obj") == objects_root / "train" / "torus" / "000.obj"
    assert reg.shape_asset("ycb_011_banana.v002", ".usd") == objects_root / "eval" / "ycb_011_banana" / "002.usd"
    assert reg.expand_variants(["ycb_011_banana"]) == ["ycb_011_banana", "ycb_011_banana.v002"]
