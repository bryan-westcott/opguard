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
lines of code. See domain-specific examples in:

- `nlp.py`: auto-captioning using HF transformers
- `vae.py`: variational autoencoders
- `sd.py`: stable diffusion and SDXL using HF diffusers.

There are several approaches for specializing to a number of ML/AI problems beyond
simple inference, including:

- Specialized mixins for diffusion (pipeline components), see `sd.py`
- Heavy quantization for auto-captioning (typically large), see `nlp.py`
- ControlNets (not callable alone), see `control.py`
- Inversion problems (uses grads), see `inversion.py`
- Bi-directional models (VAEs have both and encoder and decoder), see `vae.py`

The goal is flexibility without dogmatic use patterns while retaining
all the above protections. OpGuard avoids the need for boilerplate code,
allowing data scientists and developers to move quickly with confidence.

---

### 🚀 Minimal Example (Quick Start)

Note that we only write code for the parts that differ from other models.
All the model loading, device handling, revision enforcement, caching,
memory (VRAM) management are all handled automatically.

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

# Run the VAE safely
with TinyVAE() as vae:
    x = torch.rand(1, 3, 512, 512, device=vae.device, dtype=vae.dtype) * 2 - 1
    y = vae(input_raw=x)
```

---

### ⚙️ Running on Google Colab

If `!pip freeze | grep torch` shows `torch==2.8.0+cu126`:

- `!pip3 install --no-deps xformers --index-url https://download.pytorch.org/whl/cu126`

Then, install `opguard` from git:

- `!pip install "opguard[test] @ git+https://bryan-westcott@github.com/bryan-westcott/opguard.git@main"`

Finally, run smoke tests:

- `from opguard.tests import smoke; smoke()`

### 🔍 Testing and profiling

- PyTest test are available in `tests/test.py`:
  - Smoke test (no GPU required): `pytest -m smoke`
  - Pass-through test (for profiling): `pytest -m trivial`
  - Slower tests (4GB VRAM): `pytest -m slow`
  - Larger tests (8GB+ VRAM): `pytest -m large`
  - Full coverage: `pytest -m full`

- Profiling with `torch`:
  - Overhead timing (no inference, no transformers/diffusers):
    - `python -m torch.utils.bottleneck src/opguard/tests/trivial.py`
  - Library timing inclusive (lightweight diffusers models):
    - `python -m torch.utils.bootleneck src/opguard/tests/smoke.py`
  - Inference timing inclusive (large diffusers and transformers models):
    - `python -m torch.utils.bottleneck src/opguard/tests/large.py`

### ❓ How does it work?

There are many features provided by this module, but two of the more difficult to get correct are the following:

#### VRAM Management

OpGuard implements a strict cleanup pipeline based on the behavior in `util.py` and `OpGuardBase`. It ensures that long-running Torch and Transformers/Diffusers sessions do not leak VRAM. This can occur when references to objects on GPU prevent proper freeing, such as detached outputs or exceptions and is especially problematic with notebooks (where the solution is often to restart the kernel and lose all intermediate work).

1. Sanitized exceptions
   Exceptions are scrubbed so that traceback frames and their local variables do not retain references to tensors or model objects. This prevents VRAM from being pinned after an error.

2. Deep output detachment
   All outputs are recursively detached and moved to CPU. This removes autograd graphs and guarantees that no GPU references survive past the call boundary.

3. Deterministic synchronization
   All CUDA devices resolved by device_guard are synchronized before cleanup. This ensures predictable behavior and prevents cleanup from racing with in-flight kernels.

4. Aggressive cleanup
   After synchronization, OpGuard runs garbage collection and clears CUDA
   memory using `torch.cuda.empty_cache()`. Cleanup happens even on exceptions
   because these steps execute in finally blocks. As overly aggressive garbage
   collection can slow processing, the garbage collection (and
   sync/cache-clearing) will by default only occurs on exception or free; for
   normal operation we rely on output detachment.

5. Lazy loading
   The base object supports deferred (lazy) loading until call and (optional) post-call freeing (with all the above protections) to allow multiple models to more effectively share VRAM.

This prevents the usual VRAM degradation that occurs in notebooks and services when stale tensors or tracebacks accumulate.

---

#### Model Caching

OpGuard uses a deterministic, signature-based caching system built from the helpers in `util.py`. This avoids accidental cache hits, remote pulls, and inconsistent model variants.

1. Variant and precision detection
   OpGuard inspects HuggingFace variants and maps them to explicit export variants such as `fp16`, `bf16`, `i8`, `fp4`, or `nf4`. Export directories encode these choices directly.

2. Precision-aware local exports
   When a model is converted or quantized, OpGuard writes it to a stable export directory with metadata describing its cache signature.

3. Deterministic cache keys
   Cache signatures are based on the loader function fingerprint, normalized arguments, model id, device, dtype, and variant. This ensures cached exports are refreshed when code or settings change.

4. Match or refresh
   If an export matches the expected signature, OpGuard loads it locally. If not, it reloads the model, rebuilds the export, and updates metadata.

5. Strict locality
   When local_files_only or only_load_export is enabled, OpGuard never performs remote pulls. All variant detection and loading occurs locally.

This produces reproducible, offline-safe, and revision-stable model loading across environments.

### ⚖️ License & attribution

Apache 2.0
