"""Convert a config or a checkpoint written by an older version of this code to the current names.

Every published checkpoint carries the config it was trained with (<run>/configs/parsed.yaml), and
both name classes and keys the current code no longer has. Convert them before use:

    python scripts/convert_config.py <run>/configs/parsed.yaml <copy>/configs/parsed.yaml
    python scripts/convert_config.py --ckpt <run>/ckpts/<file>.ckpt <copy>/ckpts/<file>.ckpt

Inference reads the config beside the checkpoint (<run>/ckpts/../configs/parsed.yaml), so write the
results into a copy of the run directory and keep the originals as the record of the run. A
checkpoint is converted without loading it: its pickle is parsed and only the class references it
holds (GLOBAL opcodes) are rewritten; strings that name the old paths, and every tensor, stay as stored.
"""
import argparse
import os
import pickletools
import shutil
import zipfile

# key -> {old value: new value}
RENAMED = {
    "data_cls": {"spuv.data.lightgen_uv.LightGenDataModule": "spuv.data.mesh_uv.ObjaverseDataModule"},
    "system_cls": {"spuv.systems.lightgen_system.LightGenSystem": "spuv.systems.texgen_train.TEXGenDiffusion"},
    "system.backbone_cls": {
        "spuv.models.sparse_networks.lightgen_pointuvnet.LightGenPointUVNet":
            "spuv.models.sparse_networks.texgen_network.PointUVNet",
    },
}
# keys the current code no longer has (alpha is always read)
REMOVED = ["data.use_alpha"]
# classes a checkpoint pickles: (module, name) -> (module, name). The system's Config is pickled as
# an attribute of the system class, so the class is what moves.
PICKLED = {("spuv.systems.lightgen_system", "LightGenSystem"): ("spuv.systems.texgen_train", "TEXGenDiffusion")}
CLASS_RENAMES = {f"{om}.{on}": f"{nm}.{nn}" for (om, on), (nm, nn) in PICKLED.items()}


def convert(cfg):
    from omegaconf import OmegaConf
    for key, names in RENAMED.items():
        value = OmegaConf.select(cfg, key)
        if value in names:
            OmegaConf.update(cfg, key, names[value])
    for key in REMOVED:
        parent, _, leaf = key.rpartition(".")
        node = OmegaConf.select(cfg, parent) if parent else cfg
        if node is not None and leaf in node:
            del node[leaf]
    return cfg


def _rewrite_globals(data):
    """data.pkl with its GLOBAL opcodes (protocol 2, torch.save's default) for the old classes rewritten;
    returns (new bytes, how many changed). The pickle is parsed and only real opcodes are touched, so a
    string that happens to hold the same bytes, or names the old class path, is left as it is."""
    old_modules = {om for om, _ in PICKLED}
    ops = list(pickletools.genops(data))
    # protocol 4 frames its payload (a splice would break the frame length) and may name classes with
    # STACK_GLOBAL; neither is handled, so such a pickle that mentions an old module is refused
    if any(op.name in ("FRAME", "STACK_GLOBAL") for op, _, _ in ops) and any(m.encode() in data for m in old_modules):
        raise SystemExit("a protocol-4 pickle (FRAME or STACK_GLOBAL) that mentions an old module: not handled")
    out, last, n = [], 0, 0
    for k, (op, arg, pos) in enumerate(ops):
        if op.name != "GLOBAL":
            continue
        new = PICKLED.get(tuple(arg.split(" ", 1)))
        if new is None:
            continue
        end = ops[k + 1][2] if k + 1 < len(ops) else len(data)
        out += [data[last:pos], f"c{new[0]}\n{new[1]}\n".encode()]
        last, n = end, n + 1
    out.append(data[last:])
    return b"".join(out), n


def convert_checkpoint(src, dst):
    """Copy a torch zip checkpoint, rewriting the class references its data.pkl pickles; returns how many
    changed. Every other entry is copied as stored. Written to <dst>.part, then renamed."""
    if os.path.realpath(src) == os.path.realpath(dst):
        raise SystemExit("--ckpt writes a new file: give <new.ckpt> another path than <old.ckpt>")
    part = dst + ".part"
    n = 0
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(part, "w", zipfile.ZIP_STORED, allowZip64=True) as zout:
        for info in zin.infolist():
            out = zipfile.ZipInfo(info.filename, date_time=info.date_time)
            out.compress_type = zipfile.ZIP_STORED
            if info.filename.endswith("/data.pkl") or info.filename == "data.pkl":
                data, changed = _rewrite_globals(zin.read(info))
                zout.writestr(out, data)
                n += changed
            else:
                out.file_size = info.file_size
                with zin.open(info) as fi, zout.open(out, "w", force_zip64=info.file_size > 2**31) as fo:
                    shutil.copyfileobj(fi, fo, 64 << 20)
    os.replace(part, dst)
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", action="store_true", help="convert a checkpoint instead of a config")
    parser.add_argument("src", help="a config (or, with --ckpt, a checkpoint) written by older code")
    parser.add_argument("dst", help="where to write the converted file")
    args = parser.parse_args()
    if args.ckpt:
        n = convert_checkpoint(args.src, args.dst)
        print(f"{n} class reference(s) converted: {args.src} -> {args.dst}")
    else:
        from omegaconf import OmegaConf
        OmegaConf.save(convert(OmegaConf.load(args.src)), args.dst)


if __name__ == "__main__":
    main()
