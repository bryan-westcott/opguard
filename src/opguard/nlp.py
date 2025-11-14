"""BLIP NLP captioners."""

# We do not care about LSP substitutability, OpGuardBase is not used directly
# mypy: disable-error-code=override

# ruff: noqa: N801  # need to specify variants of nubmered models (Blip2_XBit)

from typing import ClassVar

import torch
from PIL.Image import Image as PILImage
from torch import Tensor
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
    BlipForConditionalGeneration,
    BlipProcessor,
)
from transformers.image_processing_base import BatchFeature

from .base import OpGuardBase
from .util import Detector


class Blip(OpGuardBase):
    """BLIP1/2 captioner with optional conditional text prompt."""

    # Placeholder for processor
    PROCESSOR_TYPE: ClassVar[type[Detector] | None] = None

    def _load_processor(self) -> BlipProcessor:
        if not self.PROCESSOR_TYPE:
            message = "Must set PROCSSOR_TYPE Class Var"
            raise ValueError(message)
        return self.PROCESSOR_TYPE.from_pretrained(
            self.model_id,
            revision=self.REVISION,
            use_fast=True,
            device_map=self.device_map,
        )

    def _preprocess(self, *, input_raw: PILImage, text: str | None = None) -> BatchFeature:
        return self._processor(input_raw, text=text, return_tensors="pt").to(
            self.device,
            self.dtype,
        )

    def _predict(self, *, input_proc: BatchFeature) -> Tensor:
        return self._detector.generate(**input_proc, max_new_tokens=50)

    def _postprocess(self, *, output_raw: Tensor) -> str:
        return self._processor.decode(output_raw[0], skip_special_tokens=True).strip()

    def _caller(self, *, input_raw: PILImage, text: str | None = None) -> str:
        input_proc = self._preprocess(input_raw=input_raw, text=text)
        output_raw = self._predict(input_proc=input_proc)
        return self._postprocess(output_raw=output_raw)  # output_proc


class Blip1(Blip):
    """BLIP1 captioner with optional conditional text prompt."""

    NAME = "blip1"
    MODEL_ID = "Salesforce/blip-image-captioning-base"
    REVISION = "main"
    DETECTOR_TYPE = BlipForConditionalGeneration
    PROCESSOR_TYPE = BlipProcessor


class Blip2(Blip):
    """Blip2 captioner with optional conditional text prompt.

    Various precisions supported (in order of decreasing VRAM):     Blip2_32bit,
    Blip2_16bit, Blip2_8bit, Blip2_4bit
    """

    MODEL_ID = "Salesforce/blip2-opt-2.7b"
    REVISION = "main"
    DETECTOR_TYPE = Blip2ForConditionalGeneration
    DEFAULT_DEVICE_MAP = "auto"
    PROCESSOR_TYPE = Blip2Processor
    SKIP_TO_DEVICE = True
    SKIP_TO_DTYPE = True


class Blip2_32Bit(Blip2):
    """Blip2 captioner in full-precision float32 compute (~14GB).

    WARNING: full precision (float32), for less VRAM use:
         Blip2_16bit, Blip2_8bit, Blip2_4bit

    Weights size (in storage): 14 GB
    """

    NAME = "blip2-conditional-32bit"
    DTYPE_PREFERENCE = torch.float32


class Blip2_16Bit(Blip2):
    """Blip2 captioner in half-precision float16 compute (~7GB)."""

    NAME = "blip2-conditional-16bit"
    DTYPE_PREFERENCE = torch.bfloat16


class Blip2_8Bit(Blip2):
    """Blip2 captioner with 8-bit quantization."""

    NAME = "blip2-conditional-8bit"
    DTYPE_PREFERENCE = torch.bfloat16
    DEFAULT_QUANT_TYPE = "i8"


class Blip2_4Bit(Blip2):
    """Blip2 captioner with 4-bit quantization (under 4GB VRAM)."""

    NAME = "blip2-conditional-4bit"
    DTYPE_PREFERENCE = torch.bfloat16  # when using 8-bit bnb, must use float16 not bfloat16
    DEFAULT_QUANT_TYPE = "nf4"
    DEFAULT_QUANT_USE_DOUBLE: bool = True
