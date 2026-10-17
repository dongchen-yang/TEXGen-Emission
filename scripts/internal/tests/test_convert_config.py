"""Tests for scripts/convert_config.py --ckpt: only real pickled class references are rewritten.

    python -m pytest scripts/internal/tests/test_convert_config.py -q     (texgen env, from the repo root)
"""
import os
import pickle
import struct
import sys
import types
import zipfile

import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import convert_config as cc  # noqa: E402

OLD = ("spuv.systems.lightgen_system", "LightGenSystem")
NEW = ("spuv.systems.texgen_train", "TEXGenDiffusion")


class _Thing:
    pass


def _fake_class(monkeypatch, module, name):
    """A class that pickles as module.name, with stand-in packages so unpickling never imports spuv."""
    parts = module.split(".")
    for i in range(1, len(parts)):
        pkg = ".".join(parts[:i])
        if not getattr(sys.modules.get(pkg), "_fake", False):
            monkeypatch.setitem(sys.modules, pkg, _package(pkg))
    mod = types.ModuleType(module)
    cls = type(name, (_Thing,), {"__module__": module})
    setattr(mod, name, cls)
    monkeypatch.setitem(sys.modules, module, mod)
    return cls


def _package(name):
    pkg = types.ModuleType(name)
    pkg.__path__ = []
    setattr(pkg, "_fake", True)
    return pkg


def test_an_old_pickled_class_is_rewritten_and_loads_as_the_new_one(tmp_path, monkeypatch):
    old_cls = _fake_class(monkeypatch, *OLD)
    src, dst = str(tmp_path / "old.ckpt"), str(tmp_path / "new.ckpt")
    torch.save({"state_dict": {"w": torch.arange(4.0)}, "hyper_parameters": {"cls": old_cls}, "obj": old_cls()}, src)
    assert cc.convert_checkpoint(src, dst) == 1   # pickle writes a class's GLOBAL once, then refers back to it
    new_cls = _fake_class(monkeypatch, *NEW)
    monkeypatch.delitem(sys.modules, OLD[0])
    ck = torch.load(dst, weights_only=False)
    assert type(ck["obj"]) is new_cls and ck["hyper_parameters"]["cls"] is new_cls
    assert torch.equal(ck["state_dict"]["w"], torch.arange(4.0))


def test_strings_that_hold_the_old_bytes_or_path_are_left_alone(tmp_path, monkeypatch):
    # A string holding the exact bytes of the old GLOBAL opcode, and a config string naming the old
    # class path, are data: the copy must load and equal the original, with no abort.
    old_cls = _fake_class(monkeypatch, *OLD)
    src, dst = str(tmp_path / "strings.ckpt"), str(tmp_path / "out.ckpt")
    payload = {"raw": f"c{OLD[0]}\n{OLD[1]}\n", "cfg": {"system_cls": f"{OLD[0]}.{OLD[1]}"},
               "w": torch.ones(3)}
    torch.save({**payload, "obj": old_cls()}, src)
    assert cc.convert_checkpoint(src, dst) == 1
    _fake_class(monkeypatch, *NEW)
    ck = torch.load(dst, weights_only=False)
    assert ck["raw"] == payload["raw"] and ck["cfg"] == payload["cfg"] and torch.equal(ck["w"], payload["w"])


def test_a_checkpoint_with_only_strings_is_copied_unchanged(tmp_path):
    src, dst = str(tmp_path / "plain.ckpt"), str(tmp_path / "out.ckpt")
    payload = {"meta": f"{OLD[0]}.{OLD[1]}", "raw": f"c{OLD[0]}\n{OLD[1]}\n", "w": torch.ones(2)}
    torch.save(payload, src)
    assert cc.convert_checkpoint(src, dst) == 0
    ck = torch.load(dst, weights_only=False)
    assert ck["meta"] == payload["meta"] and ck["raw"] == payload["raw"] and torch.equal(ck["w"], payload["w"])


def test_converting_in_place_is_refused(tmp_path):
    src = str(tmp_path / "x.ckpt")
    torch.save({"w": torch.ones(1)}, src)
    with pytest.raises(SystemExit):
        cc.convert_checkpoint(src, src)
    os.symlink(src, str(tmp_path / "link.ckpt"))
    with pytest.raises(SystemExit):
        cc.convert_checkpoint(src, str(tmp_path / "link.ckpt"))


def test_a_protocol_4_pickle_naming_an_old_class_is_refused(tmp_path, monkeypatch):
    old_cls = _fake_class(monkeypatch, *OLD)
    src = str(tmp_path / "p4.ckpt")
    torch.save({"obj": old_cls()}, src, pickle_protocol=4)
    with pytest.raises(SystemExit):
        cc.convert_checkpoint(src, str(tmp_path / "out.ckpt"))


def test_every_data_pkl_of_the_archive_is_converted_and_counted(tmp_path, monkeypatch):
    old_cls = _fake_class(monkeypatch, *OLD)
    pkl = pickle.dumps(old_cls(), protocol=2)
    src, dst = str(tmp_path / "two.zip"), str(tmp_path / "out.zip")
    with zipfile.ZipFile(src, "w") as z:
        z.writestr("a/data.pkl", pkl)
        z.writestr("b/data.pkl", pkl)
    assert cc.convert_checkpoint(src, dst) == 2
    new_cls = _fake_class(monkeypatch, *NEW)
    with zipfile.ZipFile(dst) as z:
        assert all(type(pickle.loads(z.read(f"{d}/data.pkl"))) is new_cls for d in "ab")


def test_a_framed_pickle_with_a_plain_global_is_refused(tmp_path, monkeypatch):
    # protocol 4 framing around a GLOBAL (no STACK_GLOBAL): a splice would break the frame length
    old_cls = _fake_class(monkeypatch, *OLD)
    payload = f"c{OLD[0]}\n{OLD[1]}\n.".encode()
    data = b"\x80\x04\x95" + struct.pack("<Q", len(payload)) + payload
    assert pickle.loads(data) is old_cls
    src = str(tmp_path / "framed.zip")
    with zipfile.ZipFile(src, "w") as z:
        z.writestr("archive/data.pkl", data)
    with pytest.raises(SystemExit):
        cc.convert_checkpoint(src, str(tmp_path / "out.zip"))
