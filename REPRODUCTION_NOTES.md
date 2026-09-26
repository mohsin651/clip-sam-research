# Reproduction decisions and limitations

Subsequent user-authorized source-informed refinements are documented separately
in REFINEMENT_PLAN.md and REFINEMENT_RESULTS.md. Everything below describes the
preserved original literal implementation unless explicitly stated otherwise.

Source: supplied `s44443-026-00779-3.pdf`, Li et al., 2026, DOI
10.1007/s44443-026-00779-3. Method equations were inspected in rendered PDF
pages 5–6, not reconstructed from the incomplete Markdown. Sections 4.1,
4.3, 4.4 and Tables 2/5 were read from the supplied paper.

## Mathematical ambiguities requiring author clarification

1. **Equation (11) is literally scalar.** It prints
   `M = ReLU(sum_i W_i sum_d G_cls[d] V_i[d])`.
   The outer spatial sum removes the patch axis. For localization we retain
   the individual spatial contributions and use
   `M_i = ReLU(W_i sum_d G_cls[d] V_i[d])`.
   This is an explicit interpretation, not an exact transcription of Eq. (11).
   Diagnostics save both signed patch contributions and the literal scalar
   `ReLU(sum_i signed_i)`; the scalar is not the sum of positive patch maps.
2. **Equation (4) versus Equation (6).** Eq. (4) sums spatial values i=1..N,
   excluding CLS, whereas Eq. (6) differentiates with respect to V_cls.
   Excluding V_cls from the actual final aggregation makes that gradient zero.
   We retain CLS in the forward attention as pretrained CLIP does, and exclude
   it only from the 196 spatial attribution positions.
3. **Full channels versus pretrained multi-head attention.** The paper states
   attention is computed across the full dimension without splitting heads.
   Default `attention_mode: full` therefore uses a single 768-dimensional
   attention operation in the final vision block, scaled by sqrt(768).
   Earlier blocks retain pretrained native attention. This changes the final
   forward function from native CLIP; weights are unchanged. `native` preserves
   CLIP's 12 heads and computes attribution cosines across the concatenated
   768 channels. It is a separately labeled sensitivity interpretation, not
   interchangeable with the default. No averaging of Q/K/V over heads is used.
4. **Ablation formulas are not fully specified.** Section 4.4 says the baseline
   uses the matching-score gradient at the attention output and Grad-Only
   substitutes G_cls without W. We use `ReLU(sum_d G[d] V_i[d])` for baseline
   and `ReLU(sum_d G_cls[d] V_i[d])` for Grad-Only. G differentiates the
   pre-output-projection attention aggregation, matching Eq. (4)'s coordinates.
   Both omit spatial weighting. The paper gives no explicit baseline formula.
5. **Testable implication:** in single-head final attention,
   `G_cls = attention[CLS,CLS] * G`, a positive scalar multiple. Consequently
   baseline and Grad-Only have identical scale-invariant localization metrics
   (up to roundoff). The unit tests verify this identity. It cannot reproduce
   the distinct first two rows of Table 5 under this interpretation. We do not
   alter the math to force the published trend. Native multi-head attention can
   yield different channel weights because each head has its own CLS weight.

## Model and tensor layout

OpenAI CLIP `ViT-B/16`, repository commit
`d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`. Checkpoint SHA256 is recorded in
environment.txt and every run. FP32, eval mode, all parameters frozen, no
optimizer/training. The score is cosine similarity, without logit scaling or
softmax over classes. Gradients are enabled at the detached final-block input;
this leaves all downstream gradients exact while avoiding earlier-block graphs.

CLIP block input: [197, B, 768]. Linear projected Q/K/V: [B,197,768]. Native
heads: [B,12,197,64]. Full-channel mode: [B,197,768], no split. Native head
outputs concatenate back to 768 channels before the learned output projection.
Q_cls, G_cls, G_Q: [B,768]; patch K/V: [B,196,768]; phi/R/W: [B,196].
Patch order is row-major, CLS at position zero, spatial map 14x14. Explicit
native attention is checked against the unmodified model on CUDA, and its
forward and input derivatives are checked against PyTorch attention in tests.

## Dataset and preprocessing

The Gao reference and 12,419/919 counts identify ImageNet-S919 validation,
not the distinct older ImageNet-Segmentation benchmark with fewer images.
Official source: https://github.com/LUSSeg/ImageNet-S . RGB masks decode as
R+256*G; 0 is background/other and 1000 is ignored. Foreground is the sample's
target segmentation class, not every nonzero pixel. All metrics exclude void.
`class_id` is the original zero-based ImageNet-1K class ID; `segmentation_class_id`
is the one-based sorted official 919 mask class. Text names come from the
ImageNet class-index mapping saved under data/metadata. The original class
synset comes from the SOURCE path in the official validation mapping, NOT the
destination folder, which can be reassigned. The five official merged synsets
are mapped to their merged mask class while preserving the original text label.

Images come from the public **unofficial** mirror
https://huggingface.co/datasets/braceletboy/imagenet-s . Its revision and shard
SHA256 values are saved. All mirror IDs must match the official split exactly;
selected dimensions and target labels are checked. Original ImageNet image
byte identity cannot be independently established from this mirror alone.
Its card's metadata says MIT but its prose refers to ImageNet terms; we make
no claim that the underlying ImageNet images are MIT licensed.

OpenAI preprocessing: RGB, bicubic shortest-side resize to 224, center crop
224x224, checkpoint mean/std normalization. Masks receive identical geometry
with nearest-neighbor interpolation. Evaluation is on the visible 224x224
crop, not distorted full-image geometry. The paper does not specify crop/mask
alignment. Original-sized images remain in data; saved `original.png` is
explicitly the model input crop. Patch maps use bilinear interpolation with
align_corners=False. This interpolation choice is unspecified in the paper.

## Prompts, masks, metrics, statistics

The localization prompt template is not specified. Configurable default:
`a photo of a {class_name}`. The segmentation threshold is also unspecified;
we min-max scale resized scores over valid pixels and use >=0.5. Threshold
is configurable. EPG uses raw nonnegative energy without min-max subtraction;
PG/AP likewise use raw scores. AP is sklearn precision-recall average precision.
IoU is foreground IoU per image, averaged across images, as requested; the
paper does not resolve foreground-only versus foreground/background mIoU.
AP, PAcc, EPG and PG are also macro-averaged over images.

Zero-energy or constant maps count as PG misses; EPG is zero for zero energy.
Constant-score AP follows sklearn. Empty target AP is zero; empty-union IoU
is one. Empty targets in original annotations or after cropping are retained,
not silently resampled. CSV columns `target_present_original` and
`target_present_crop` disclose both cases. The official split puts
ILSVRC2012_val_00030368 under Boston bull (mask ID 196), but the original
ImageNet class in its official SOURCE mapping is worm fence (mask ID 848),
which is present in the annotation. This discovered edge case is why target
semantics use the official source mapping rather than destination folders.
PG ties use the first valid row-major maximum. Degenerate maps stop the run
for investigation; they are never silently dropped or replaced.

Subset: sorted official image IDs, NumPy default_rng(42) permutation. 20 is
the prefix of 500; all variants share the same ordering. Existing subset files
must match. Bootstrap: 2,000 image-level resamples, seed 2026, percentile 95%
CI, sample SD. Paired differences align image IDs before resampling. Runtime
is shared forward/backward attribution time for all variants, CUDA synchronized;
it excludes rendering and must not be interpreted as independent variant cost.
Whole-run wall time includes rendering and statistics.

## Current environment

The supplied request described Linux, but this workspace is Windows. A managed
Python 3.11 and a root `.venv` were created without modifying system Python.
CUDA 12.8 PyTorch wheels work with the installed driver and RTX 4000 Ada 20 GB.
Exact installed packages are in requirements.txt; machine details in
environment.txt. Linux may require platform-appropriate CUDA wheel resolution.

This is a documented interpretation study, not a claim of exact reproduction.
In particular, author clarification is needed for the five mathematical points
above before any Table 5 discrepancy can establish implementation failure.

## Observed tensor audit

The first three real-image debug records verify
`G_cls = attention[CLS,CLS] * G` with zero maximum absolute discrepancy in
FP32. Their CLS self-attention weights are approximately 0.777–0.853. Around
96–97% of their fused spatial weights W are negative. Since Eq. (7) leaves
phi signed and Eq. (10) adds only ReLU(cos(G_Q,K)), the implementation retains
these negative W values. It does not insert ReLU(phi), softmax, min-max
rescaling of phi, or a fusion coefficient. The resulting patchwise final ReLU
can suppress positive channel contributions and retain negative ones multiplied
by negative W. This is an observed consequence of the documented interpretation,
not a numerical NaN/Inf failure. See outputs/logs/real_tensor_audit.json.
