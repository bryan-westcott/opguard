


### Why this package?

Have you ever…

* wasted time juggling devices, dtypes, and model variants for every machine you run on?  
* accidentally pulled remote weights you *swore* were cached?  
* forgotten to call `.eval()` or disable grads before a “quick test”?  
* wished your favorite HF model already had your mixins and variant pre-baked?  
* reluctantly restarted a Jupyter kernel just to reclaim VRAM after a stray exception?  
* fought with yet another inconsistent model loader signature (but still love HF)?  
* watched production suddenly and perplexingly fail because a weight revision changed?  
* burned time and money re-caching publicly available weights across environments?  

Yeah — same.

---

### What it is

**OpGuard** provides a minimal, extensible layer for safe, deterministic inference.  
It wraps all the gritty setup and teardown around your model so you can focus on logic, not leaks.

---

### Composable context managers

Use them directly if you only need specific functionality:

| Category           | Convenience<br>Aggregate  | Individual<br>Guards | Purpose |
|:-------------------|:----------------:|:---:|:--------------|
| **Initialization** |  `init_guard` | `device_guard`,<br>`dtype_guard`,<br>`variant_guard` | Pick the best device, dtype, and model variant for your hardware |
| **Loading**        | `load_guard` | `local_guard`,<br>`eval_guard`,<br>`cache_guard` | Enforce local-only loads, set train/eval mode, and safely cache exports |
| **Calling**        | `call_guard` | `autocast_guard`,<br>`grad_guard`,<br>`vram_guard` | Handle mixed precision, no-grad inference, and predictable VRAM cleanup |
| **Cleanup**        | - |`free_guard` | Garbage-collect and clear Torch caches on release |

---

### `OpGuardBase`: all-in-one wrapper object

If you’d rather not wire these together yourself, subclass `OpGuardBase`.  
It gives you **all of the above** in one clean abstraction:

* device/dtype/variant resolution  
* automatic precision fallback (bf16 → fp16 → fp32)  
* local-only caching and export management  
* deterministic AMP / grad / VRAM cleanup  
* trace-scrubbing exception handling  

---

### Minimal Example 

```python
import torch
from opguard import OpGuardBase

class TinyVAE(OpGuardBase):
    NAME = "tiny-vae"
    MODEL_ID = "madebyollin/taesd"
    REVISION = "main"

    def _load_detector(self, **kw):
        from diffusers import AutoencoderTiny
        return AutoencoderTiny.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            local_files_only=self.local_files_only,
            revision=self.revision,
        ).to(self.device)

    def _predict(self, *, input_proc):
        return self._detector.decode(
            self._detector.encode(input_proc.to(self.device, self.dtype)).latents
        ).sample

with TinyVAE() as vae:
    x = torch.rand(1, 3, 512, 512, device=vae.device, dtype=vae.dtype) * 2 - 1
    y = vae(input_raw=x)
```

---

### What you get

* **Automatic precision fallback** — picks the best supported dtype  
* **Local-only caching** — no accidental network pulls in production  
* **Guarded execution** — detaches outputs and clears VRAM on every call  
* **Predictable cleanup** — no zombie tensors, no restarts  
* **Revision-aware exports** — protects you from silent upstream changes  
* **Unified API** — works with Diffusers, Transformers, or your own models  

---

### License & attribution

Apache 2.0, see LICENSE and NOTICE files.
