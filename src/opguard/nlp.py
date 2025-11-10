"""BLIP NLP captioners."""

# We do not care about LSP substitutability, OpGuardBase is not used directly
# mypy: disable-error-code=override

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


class Blip(OpGuardBase):
    """BLIP1/2 captioner with optional conditional text prompt."""

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

    def _load_processor(self) -> BlipProcessor:
        return BlipProcessor.from_pretrained(
            self.model_id,
            revision=self.REVISION,
            use_fast=True,
            device_map=self.device_map,
        )


class Blip2(Blip):
    """Blip2 captioner with optional conditional text prompt."""

    NAME = "blip2-conditional"
    MODEL_ID = "Salesforce/blip2-opt-2.7b"
    REVISION = "main"
    DETECTOR_TYPE = Blip2ForConditionalGeneration
    DTYPE_PREFERENCE = torch.float16  # when using 8-bit bnb, must use float16 not bfloat16
    DEFAULT_DEVICE_MAP = "auto"
    SKIP_TO_DEVICE = True
    SKIP_TO_DTYPE = True
    DEFAULT_QUANT_TYPE = "i8"
    FROM_PRETRAINED_SKIP_KWARGS = ("quantization_config",)  # use individual components

    def _load_processor(self) -> Blip2Processor:
        return Blip2Processor.from_pretrained(
            self.model_id,
            revision=self.REVISION,
            use_fast=True,
            device_map=self.device_map,
        )

    def _load_detector(self) -> Blip2ForConditionalGeneration:
        return super()._load_detector(
            load_in_8bit=getattr(self.quant_config, "load_in_8bit", False),  # pair with FROM_PRETRAINED_SKIP_KWARGS
        )
