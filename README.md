# Notarius
English speech to text deep learning model based on QuartzNet, with an inverted bottleneck architecture replacing the standard time-channel separable convolutions.

## Architecture

Notarius keeps the same macro-structure as QuartzNet (C1 → B1–B5 → C2 → C3 → C4 with block-level residuals) but replaces the core convolution module and makes several other changes.

### QuartzNet baseline

Each block runs R=5 `TSCConv` modules in sequence, where `TSCConv` is:

```
Depthwise Conv → Pointwise Conv → BN → (optional ReLU)
```

The block adds a 1×1 pointwise residual from input to output, applies ReLU after the add, and uses fixed channel widths (256 for B1–B2, 512 for B3–B5, 1024 for C3).

### Notarius changes

**1. Inverted Bottleneck module (`IBConv`) replaces `TSCConv`**

Each `IBConv` expands channels before the depthwise conv and compresses back after, following the MobileNetV2 inverted bottleneck pattern:

```
Pointwise expand (C → C×t) → BN → ReLU
Depthwise Conv               → BN → ReLU
Pointwise compress (C×t → C) → BN
```

The final compress step has **no ReLU** (linear bottleneck) — applying ReLU in the narrow compressed space destroys information.

**2. Per-module residuals added inside `IBConv`**

When input and output channels match and stride=1, each `IBConv` adds its own skip connection in addition to the existing block-level residual. QuartzNet only has block-level residuals.

**3. R=3 modules per block instead of R=5**

Each `IBConv` contains 3 convolutions (expand + depthwise + compress) versus `TSCConv`'s 2 (depthwise + pointwise). R=3 IB modules gives a similar total conv depth per block as R=5 QuartzNet modules.

**4. Configurable channel width via `C` parameter**

Fixed QuartzNet widths are replaced by a single `C` parameter. Block channels are `C` (B1–B2) and `2C` (B3–B5), with the C3 head at `4C`. Recommended configs:

| C | Params | Comparable to |
|---|--------|---------------|
| 172 | ~6.7M | QuartzNet 5×5 |
| 192 | ~8.2M | between 5×5 and 10×5 |
| 256 | ~14.1M | QuartzNet 10×5–15×5 |

**5. Expand factor `t=2` (default)**

QuartzNet expands channels only at the C3 head (256→1024, ×4). Notarius expands within every conv module at ratio `t` (default 2), keeping mid-channel count manageable.

## Save and reuse training progress

Training now writes checkpoints to `outputs/checkpoints`:

- `last.pt`: most recent epoch
- `best.pt`: best validation loss so far
- `epoch_XXX.pt`: periodic snapshot (default every 10 epochs)
- `final_model.pt`: final weights after training ends

Run training:

```bash
python model/train.py
```

Resume a stopped run from a checkpoint:

```python
from model.train import train_model

train_model(B=5, R=5, num_epochs=10, resume_from="last.pt")
```

Transcribe a real wav file using a saved checkpoint:

```bash
python model/transcribe.py --audio /path/to/audio.wav --checkpoint outputs/checkpoints/best.pt
```
