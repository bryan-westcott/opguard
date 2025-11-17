### 💭 Why this package?

Have you ever…

- wasted time juggling devices, dtypes, and model variants for every machine you run on?
- accidentally pulled remote weights you _swore_ were cached?
- forgotten to call `.eval()` or disable grads before a “quick test”?
- wished your favorite HF model already had your mixins and variant pre-baked?
- reluctantly restarted a Jupyter kernel just to reclaim VRAM after a stray exception?
- spent time fiddling with picky quantization parameters?
- fought with yet another inconsistent model loader signature (but still love HF)?
- watched production suddenly and perplexingly fail because a weight revision changed?
- burned time and money re-caching publicly available weights across environments?

Yeah — same.

---

### ⚙️ What it is

**OpGuard** provides a minimal, extensible layer for safe, deterministic inference.
It wraps all the gritty setup and teardown around your model so you can focus on logic, not leaks.

---

### 🧩 Composable context managers

Use them directly if you only need specific functionality:

| Category           | Convenience<br>Aggregate |                          Individual<br>Guards                          | Purpose                                                                                    |
| :----------------- | :----------------------: | :--------------------------------------------------------------------: | :----------------------------------------------------------------------------------------- |
| **Initialization** |       `init_guard`       | `device_guard`,<br>`dtype_guard`,<br>`variant_guard`,<br>`quant_guard` | Pick the best device, dtype, model variant, and quantization for your hardware             |
| **Loading**        |       `load_guard`       |    `local_guard`,<br>`eval_guard`,<br>`vram_guard`<br>`cache_guard`    | Enforce local-only loads, set train/eval mode, guard VRAM on load and safely cache exports |
| **Calling**        |       `call_guard`       |  `eval_guard`,<br>`autocast_guard`,<br>`grad_guard`,<br>`vram_guard`   | Handle mixed precision, no-grad inference, and predictable VRAM cleanup                    |
| **Cleanup**        |            -             |                              `free_guard`                              | Garbage-collect and clear Torch caches on release (included in OpGuard)                    |

---

### 🧰 `OpGuardBase`: all-in-one wrapper object

If you’d rather not wire these together yourself, subclass `OpGuardBase`.
It gives you **all of the above** in one clean abstraction:

- **Automatic precision fallback** — picks the best supported dtype/device/variant
- **Local-only caching** — no accidental network pulls in production
- **Automatic AMP/eval/no-grad mode** — automatic mixed precision, no-gradient and eval modes
- **Guarded execution** — detaches outputs and clears VRAM on every free
- **Predictable cleanup** — no zombie tensors, no Jupyter restarts, even on exceptions
- **VRAM-safe exception handling** — careful trace-scrubbing and garbage/cache handling avoids leaks on exceptions
- **Revision-aware exports** — protects you from silent upstream changes
- **Unified API** — works with Diffusers, Transformers, or your own models
- **Easy/flexible extensibility** — easy to extend without boilerplate or dogmatism

OpGuard also adds an abstract base class that is easily customizable with a few
lines of code (see domain-specific examples in `nlp.py`, `vae.py`, `sd.py`, etc.).
There are several approaches for specializing to a number of ML/AI problems beyond
simple inference, including:

- Diffusers mixins, see `sd.py`
- ControlNets (not callable alone), see `control.py`
- Inversion problems (uses grads), see `inversion.py`

The goal is flexibility without dogmatic use patterns while retaining
all the above protections. OpGuard avoids the need for boilerplate code,
allowing data scientists and developers to move quickly with confidence.

---

### 🚀 Minimal Example (Quick Start)

Note that we only write code for the parts that differ from other models.
All the model loading, device handling, revision enforcement, caching,
memory (VRAM) management are all handled automatically. Note also that
the example below shows how easy it is to build an atypical use case:
two calls to provide encode followed by decode.

```python
import torch
from opguard import OpGuardBase
from diffusers import AutoencoderTiny

class TinyVAE(OpGuardBase):
    NAME = "tiny-vae"
    MODEL_ID = "madebyollin/taesd"
    REVISION = "main"
    DETECTOR_TYPE = AutoencoderTiny

    def _load_processor(self) -> VaeImageProcessor:
        """Load the pre/post processor (detector for inference loaded automatically)"""
        return VaeImageProcessor(vae_scale_factor=8)

    def _preprocess(self, *, input_raw: PILImage) -> torch.FloatTensor:
        """Apply pre-processing (called automatically before _predict)"""
        return self._processor.preprocess(input_raw).to(self.device, self.dtype)  # (B,C,H,W)

    def _predict(self, *, input_proc):
        """Build a bespoke predictor that does encode-decode sequence (two calls)."""
        return self._detector.decode(
            self._detector.encode(input_proc.to(self.device, self.dtype)).latents
        ).sample

    def _postprocess(self, *, output_raw: torch.FloatTensor) -> PILImage:
        """Apply post-processing (called automatically after _predict."""
        return self._processor.postprocess(output_raw, output_type="pil")[0]

# Test code
with TinyVAE() as vae:
    x = torch.rand(1, 3, 512, 512, device=vae.device, dtype=vae.dtype) * 2 - 1
    y = vae(input_raw=x)
```

---

### ⚖️ Running on Google Colab

If `!pip freeze | grep torch` shows `torch==2.8.0+cu126`:

* `!pip3 install --no-deps xformers --index-url https://download.pytorch.org/whl/cu126`

Then, install `opguard` from git:

* `!pip install "opguard[test] @ git+https://bryan-westcott@github.com/bryan-westcott/opguard.git@main"`

Finally, run smoke tests:

* `from opguard.tests import smoke; smoke()`



### ⚖️ License & attribution

Apache 2.0
