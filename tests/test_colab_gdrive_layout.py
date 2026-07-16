"""Verify the Colab notebook's Google-Drive layout WITHOUT Colab, by checking the
target **rclone remote** (default ``i:``) that Colab's ``/content/drive/MyDrive`` mirrors.

The Colab run addresses everything under ``/content/drive/MyDrive/…`` (see
``PathConfig.for_colab``). That same content lives on the ``i:`` rclone remote, so the exact
Drive paths the run needs can be validated up front by mapping
``/content/drive/MyDrive/<X>`` → ``<remote><X>`` and probing with ``rclone lsf``.

This is the *local pre-flight for the Colab pre-flight cell*: run it before you open the
notebook on Colab and you know in seconds whether your Drive has every checkpoint + corpus.

Usage
-----
    pytest tests/test_colab_gdrive_layout.py -q            # full parametrized suite
    python -m tests.test_colab_gdrive_layout               # CLI report (default remote i:)
    python -m tests.test_colab_gdrive_layout --remote ber: --al ft-gpt2-small
    GGE_RCLONE_REMOTE=i: pytest tests/test_colab_gdrive_layout.py

Remote resolution: ``--remote`` CLI arg  >  ``$GGE_RCLONE_REMOTE``  >  default ``i:``.
Tests SKIP (not fail) when rclone is absent or the remote is unreachable/rate-limited, so an
offline machine doesn't red-fail; a reachable-but-missing path is a real FAILURE.
"""
from __future__ import annotations

import os
import shutil
import subprocess

try:
    import pytest  # optional — only the pytest suite needs it; the CLI report runs without it
except ImportError:  # pragma: no cover
    pytest = None

from gen_gec_errant.registry import MODEL_REGISTRY, DATASET_REGISTRY, PathConfig

GDRIVE_PREFIX = "/content/drive/MyDrive/"
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
_PATHS = PathConfig.for_colab()


def default_remote() -> str:
    return os.environ.get("GGE_RCLONE_REMOTE", "i:")


def gdrive_to_rclone(gdrive_path, remote: str | None = None) -> str:
    """Map a Colab ``/content/drive/MyDrive/<X>`` path to ``<remote><X>``."""
    remote = remote or default_remote()
    s = str(gdrive_path)
    if not s.startswith(GDRIVE_PREFIX):
        raise ValueError(f"not a MyDrive path (cannot map to rclone): {s}")
    return remote + s[len(GDRIVE_PREFIX):]


def rclone_available() -> bool:
    return shutil.which("rclone") is not None


def rclone_lsf(path: str, timeout: int = 60):
    """Entries (basenames) at an rclone path, or ``None`` if unreachable/missing/error."""
    try:
        out = subprocess.run(["rclone", "lsf", path], capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if out.returncode != 0:
        return None
    return [ln.strip().rstrip("/") for ln in out.stdout.splitlines() if ln.strip()]


# AL keys that live on Drive (fine-tuned checkpoints); controls load from the HF Hub, not Drive.
AL_KEYS = sorted(k for k, m in MODEL_REGISTRY.items() if m.is_learner_tuned and m.gdrive_subpath)
CONTROL_KEYS = sorted(k for k, m in MODEL_REGISTRY.items() if not m.gdrive_subpath)


def al_checkpoint_rclone(model, remote: str | None = None) -> str:
    return gdrive_to_rclone(_PATHS.model_gdrive_path(model), remote)


def check_al(key: str, remote: str | None = None):
    """Return (status, detail). status ∈ {'ok','missing','unreachable'}."""
    rpath = al_checkpoint_rclone(MODEL_REGISTRY[key], remote)
    entries = rclone_lsf(rpath)
    if entries is None:
        return "unreachable", rpath
    files = set(entries)
    if "config.json" in files and any(w in files for w in WEIGHT_FILES):
        return "ok", rpath
    return "missing", f"{rpath} (has: {sorted(files)[:6]})"


def check_dataset(dkey: str, remote: str | None = None):
    ds = DATASET_REGISTRY[dkey]
    rpath = gdrive_to_rclone(_PATHS.dataset_path(ds), remote)
    parent, fname = rpath.rsplit("/", 1)
    entries = rclone_lsf(parent + "/")
    if entries is None:
        return "unreachable", rpath
    return ("ok", rpath) if fname in set(entries) else ("missing", rpath)


# ── the pytest suite (defined only when pytest is importable) ────────────────
if pytest is not None:

    def test_gdrive_to_rclone_mapping():
        assert gdrive_to_rclone("/content/drive/MyDrive/_p/x", "i:") == "i:_p/x"
        assert gdrive_to_rclone(_PATHS.data_root, "ber:").startswith("ber:phd-experimental-data/")
        with pytest.raises(ValueError):
            gdrive_to_rclone("/not/a/drive/path")

    def test_controls_are_hf_not_drive():
        # controls must resolve to an HF id (no Drive dependency) — else the pairing is wrong.
        for k in CONTROL_KEYS:
            assert MODEL_REGISTRY[k].hf_model_id and MODEL_REGISTRY[k].gdrive_subpath is None

    # remote-dependent tests (skip when rclone/remote unavailable)
    @pytest.mark.skipif(not rclone_available(), reason="rclone not installed")
    @pytest.mark.parametrize("key", AL_KEYS)
    def test_al_checkpoint_on_drive(key):
        status, detail = check_al(key)
        if status == "unreachable":
            pytest.skip(f"remote unreachable (offline/rate-limited): {detail}")
        assert status == "ok", f"{key}: checkpoint incomplete/missing on Drive — {detail}"

    @pytest.mark.skipif(not rclone_available(), reason="rclone not installed")
    @pytest.mark.parametrize("dkey", sorted(DATASET_REGISTRY))
    def test_dataset_on_drive(dkey):
        status, detail = check_dataset(dkey)
        if status == "unreachable":
            pytest.skip(f"remote unreachable: {detail}")
        assert status == "ok", f"{dkey}: corpus missing on Drive — {detail}"


# ── broker-manifest tests (network-free: the broker's addressing must agree with
#    this suite's gdrive→rclone mapping, and verification must behave) ──────────
if pytest is not None:
    from gen_gec_errant import brokers as bk

    def test_manifest_covers_al_checkpoints_and_datasets():
        man = bk.build_manifest()
        for k in AL_KEYS:  # every AL checkpoint present as checkpoints/<key>
            assert f"checkpoints/{k}" in man and man[f"checkpoints/{k}"].kind == "checkpoint"
        for dk in DATASET_REGISTRY:  # every dataset present as corpora/<filename>
            res = f"corpora/{DATASET_REGISTRY[dk].filename}"
            assert res in man and man[res].kind == "corpus"
        for ck in CONTROL_KEYS:  # controls are NOT broker resources (HF Hub by id)
            assert f"checkpoints/{ck}" not in man

    def test_broker_drive_relpath_matches_gdrive_to_rclone():
        # the broker's remote addressing must equal this suite's mapping (no drift)
        man = bk.build_manifest()
        for k in AL_KEYS:
            spec = man[f"checkpoints/{k}"]
            assert "i:" + bk._drive_relpath(spec) == al_checkpoint_rclone(MODEL_REGISTRY[k], "i:")
        rb = bk.RcloneBroker(man, remote="i:")
        for dk in DATASET_REGISTRY:
            spec = man[f"corpora/{DATASET_REGISTRY[dk].filename}"]
            assert rb._remote_path(spec) == gdrive_to_rclone(_PATHS.dataset_path(DATASET_REGISTRY[dk]), "i:")

    def test_verify_checkpoint_and_corpus(tmp_path):
        man = bk.build_manifest()
        ck = man[f"checkpoints/{AL_KEYS[0]}"]
        co = next(v for v in man.values() if v.kind == "corpus")
        d = tmp_path / "ckpt"; d.mkdir()
        assert not bk.verify(ck, d)                       # empty dir
        (d / "config.json").write_text("{}")
        assert not bk.verify(ck, d)                       # weights still missing
        (d / "model.safetensors").write_bytes(b"x")
        assert bk.verify(ck, d)                           # config + weights ⇒ ok
        f = tmp_path / "c.csv"
        assert not bk.verify(co, f)                       # absent
        f.write_text("text\nhello\n")
        assert bk.verify(co, f)                           # nonempty file ⇒ ok

    def test_local_broker_verifies_in_place_or_raises(tmp_path):
        man = bk.build_manifest()
        co = next(v for v in man.values() if v.kind == "corpus")
        (tmp_path / "splits").mkdir()
        corpus = tmp_path / "splits" / DATASET_REGISTRY[co.registry_key].filename
        corpus.write_text("text\nx\n")
        ok_paths = PathConfig(data_root=tmp_path / "splits", models_root=tmp_path / "m", output_root=tmp_path / "o")
        assert bk.make_broker("local", man, paths=ok_paths).acquire(co.name, tmp_path / "ignored") == corpus
        bad_paths = PathConfig(data_root=tmp_path / "empty", models_root=tmp_path / "m", output_root=tmp_path / "o")
        with pytest.raises(RuntimeError):
            bk.make_broker("local", man, paths=bad_paths).acquire(co.name, tmp_path / "ignored")


# ── CLI report ───────────────────────────────────────────────────────────────
def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Check the Colab Drive layout on an rclone remote.")
    ap.add_argument("--remote", default=default_remote(), help="rclone remote (default: i: / $GGE_RCLONE_REMOTE)")
    ap.add_argument("--al", nargs="*", default=AL_KEYS, help="AL registry keys to check (default: all)")
    ap.add_argument("--datasets", nargs="*", default=sorted(DATASET_REGISTRY), help="dataset keys (default: all)")
    args = ap.parse_args(argv)

    if not rclone_available():
        print("rclone not installed — cannot probe the remote."); return 2

    icon = {"ok": "✅", "missing": "❌", "unreachable": "⚠️ "}
    print(f"== Colab Drive layout on remote '{args.remote}' ==")
    fail = 0
    print("- Artificial-learner checkpoints (must exist on Drive):")
    for k in args.al:
        st, d = check_al(k, args.remote)
        print(f"  {icon[st]} {k:16s} {d}")
        fail += st == "missing"
    print("- Datasets:")
    for dk in args.datasets:
        st, d = check_dataset(dk, args.remote)
        print(f"  {icon[st]} {dk:16s} {d}")
        fail += st == "missing"
    print(f"- Matched controls (load from HF Hub, no Drive needed): {', '.join(CONTROL_KEYS) or '(none)'}")
    print(f"\n{'FAILED' if fail else 'OK'} — {fail} missing path(s). (⚠️ = unreachable, not counted.)")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
