from __future__ import annotations

import importlib.util
import tarfile
from pathlib import Path

import pytest


def _script():
    path = Path(__file__).parents[1] / "scripts" / "validate_submission.py"
    spec = importlib.util.spec_from_file_location("kaggriculture_validate_submission", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_requires_the_complete_submission_surface(tmp_path) -> None:
    module = _script()
    archive = tmp_path / "submission.tar.gz"
    main = tmp_path / "main.py"
    main.write_text("def agent(obs): return {}\n", encoding="utf-8")
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(main, arcname="main.py")

    with pytest.raises(ValueError, match="incomplete"):
        module._extract(archive, tmp_path / "extracted")


def test_extract_rejects_unbound_extra_members_before_extraction(tmp_path) -> None:
    module = _script()
    root = tmp_path / "root"
    for relative in module.REQUIRED_MEMBERS | {"unexpected.bin"}:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"payload")
    archive = tmp_path / "submission.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                bundle.add(path, arcname=path.relative_to(root).as_posix())

    with pytest.raises(ValueError, match="unbound"):
        module._extract(archive, tmp_path / "extracted")


def test_builtin_and_public_opponents_resolve_strictly(tmp_path, monkeypatch) -> None:
    module = _script()
    assert module._opponent("starter") == "starter"
    opponent = tmp_path / "opponent.py"
    opponent.write_text("def agent(obs): return {}\n", encoding="utf-8")
    assert module._opponent(str(opponent)) == str(opponent.resolve())
    monkeypatch.setattr(module, "PUBLIC_V27_OPPONENT", tmp_path / "missing.py")
    with pytest.raises(FileNotFoundError):
        module._opponent("v27")
