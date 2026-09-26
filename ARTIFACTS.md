# Data and raw artifact restoration

Git contains research records and the small fitted selector/gate parameters. Large public datasets, pretrained weights and generated tensors remain in the original local workspace. Keep that workspace as the raw backup until a separate artifact store is explicitly arranged.

| Excluded material | Original project path | Restoration |
|---|---|---|
| COCO val2017 RGB and annotations | `data/coco/` | Official download logic in `scripts/prepare_coco.py`; retain frozen cohort IDs. |
| ImageNet-S / ImageNet images and archives | `data/ImageNetS919/`, `data/downloads/` | Follow `data/README.md` and acquisition scripts; access requirements may apply. |
| CLIP ViT-B/16 weights | `data/checkpoints/ViT-B-16.pt` | Existing `src/model.py` uses official CLIP download machinery. SHA256 in frozen method. |
| SAM ViT-B weights | `data/checkpoints/sam_vit_b_01ec64.pth` | Official URL recorded in `scripts/prepare_sam.py`; download the weight separately without overwriting existing study records. |
| Attribution maps and candidate arrays | `outputs/**/checkpoints/*.npz` and nested equivalents | Copy exact originals from the existing workspace, or reproduce in a separate checkout/output destination. |
| Older galleries and generated plots | `outputs/` PNG/PDF files other than current heldout examples | Copy from original workspace or regenerate reports after restoring their required inputs. |

Do not run acquisition/preparation scripts blindly against published outputs: some also create provenance records or manifests. Read their behavior first. Frozen inference deliberately refuses an existing run; preserve that protection. Reproducing a completed run requires a separate destination and comparison against retained metrics, not deleting the historical outputs.

Source integrity records contain original Windows paths and environment-specific package paths. `.gitattributes` disables newline conversion to preserve original file hashes. Missing excluded files or relocated package paths must be distinguished from actual source changes; never rewrite the historical integrity manifest to make a new machine appear identical.

The repository inventory is in `REPOSITORY_SNAPSHOT.json`. Git does not contain the paper PDF from the parent workspace. The research notes preserve the interpretation, equations, unresolved ambiguities and source references used in the experiments.
