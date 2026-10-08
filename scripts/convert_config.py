"""Convert a config written by an older version of this code to the current names.

Every published checkpoint carries the config it was trained with (<run>/configs/parsed.yaml), and
that file names classes and keys the current code no longer has. Convert it before use:

    python scripts/convert_config.py <run>/configs/parsed.yaml <copy>/configs/parsed.yaml

Inference reads the config beside the checkpoint (<run>/ckpts/../configs/parsed.yaml), so write the
result into a copy of the run directory and keep the original as the record of the run.
"""
import argparse

from omegaconf import OmegaConf

# key -> {old value: new value}
RENAMED = {
    "data_cls": {"spuv.data.lightgen_uv.LightGenDataModule": "spuv.data.mesh_uv.ObjaverseDataModule"},
    "system.backbone_cls": {
        "spuv.models.sparse_networks.lightgen_pointuvnet.LightGenPointUVNet":
            "spuv.models.sparse_networks.texgen_network.PointUVNet",
    },
}
# keys the current code no longer has (alpha is always read)
REMOVED = ["data.use_alpha"]


def convert(cfg):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", help="a config written by older code")
    parser.add_argument("dst", help="where to write the converted config")
    args = parser.parse_args()
    OmegaConf.save(convert(OmegaConf.load(args.src)), args.dst)


if __name__ == "__main__":
    main()
