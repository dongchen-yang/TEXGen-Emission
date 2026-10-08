"""Convert a config or a checkpoint written by an older version of this code to the current names.

Every published checkpoint carries the config it was trained with (<run>/configs/parsed.yaml), and
both name classes and keys the current code no longer has. Convert them before use:

    python scripts/convert_config.py <run>/configs/parsed.yaml <copy>/configs/parsed.yaml
    python scripts/convert_config.py --ckpt <run>/ckpts/<file>.ckpt <copy>/ckpts/<file>.ckpt

Inference reads the config beside the checkpoint (<run>/ckpts/../configs/parsed.yaml), so write the
results into a copy of the run directory and keep the originals as the record of the run. A
checkpoint is converted without loading it: only the class paths its pickle names are rewritten, and
every tensor is copied as stored.
"""
import argparse
import os
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


def convert_checkpoint(src, dst):
    """Copy a torch zip checkpoint, rewriting the GLOBAL class references of its data.pkl (torch.save
    pickles with protocol 2). Written to <dst>.part, then renamed."""
    part = dst + ".part"
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(part, "w", zipfile.ZIP_STORED, allowZip64=True) as zout:
        for info in zin.infolist():
            out = zipfile.ZipInfo(info.filename, date_time=info.date_time)
            out.compress_type = zipfile.ZIP_STORED
            if info.filename.endswith("/data.pkl") or info.filename == "data.pkl":
                data = zin.read(info)
                for (om, on), (nm, nn) in PICKLED.items():
                    data = data.replace(f"c{om}\n{on}\n".encode(), f"c{nm}\n{nn}\n".encode())
                    if om.encode() in data:
                        raise SystemExit(f"{src}: {om} is still referenced after the rewrite")
                zout.writestr(out, data)
            else:
                out.file_size = info.file_size
                with zin.open(info) as fi, zout.open(out, "w", force_zip64=info.file_size > 2**31) as fo:
                    shutil.copyfileobj(fi, fo, 64 << 20)
    os.replace(part, dst)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", action="store_true", help="convert a checkpoint instead of a config")
    parser.add_argument("src", help="a config (or, with --ckpt, a checkpoint) written by older code")
    parser.add_argument("dst", help="where to write the converted file")
    args = parser.parse_args()
    if args.ckpt:
        convert_checkpoint(args.src, args.dst)
    else:
        from omegaconf import OmegaConf
        OmegaConf.save(convert(OmegaConf.load(args.src)), args.dst)


if __name__ == "__main__":
    main()
