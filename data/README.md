# ImageNet-S919 validation

Official dataset: https://github.com/LUSSeg/ImageNet-S

`python scripts/download_data.py --num-samples 500` downloads four public
mirror parquet shards (~1.7 GB) into data/downloads/mirror and extracts only
the deterministic 500 samples. `--num-samples all` extracts all 12,419.
The mirror is explicitly unofficial; provenance, hashes, official split-ID
checks and spatial/target checks are saved in ImageNetS919/provenance.json.

Expected native layout:

```
data/ImageNetS919/
  validation/n01440764/ILSVRC2012_val_00024327.JPEG
  validation-segmentation/n01440764/ILSVRC2012_val_00024327.png
  provenance.json
```

If the public mirror becomes unavailable, obtain the ImageNet ILSVRC2012
validation images through https://image-net.org/download.php and follow
the official repository's validation preparation instructions. Official masks:
https://github.com/LUSSeg/ImageNet-S/releases/download/ImageNet-S/ImageNetS919-5f7f58ae1003d21da9409a8576bf7680.zip

Do not substitute ImageNet-S50/S300 or the older ImageNet segmentation set.
The official metadata source checkout is data/ImageNet-S-source, and the
small metadata needed to load data is preserved separately in data/metadata.
