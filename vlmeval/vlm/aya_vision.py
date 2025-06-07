import logging

import torch
from PIL import Image

from ..smp import *
from .base import BaseModel


class AyaVision(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path="CohereForAI/aya-vision-8b", **kwargs):
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except Exception as e:
            logging.critical("Please install the latest version of transformers.")
            raise e

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, device_map="cuda", torch_dtype=torch.float16
        ).eval()

        self.device = self.model.device
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

        self.system_prompt = kwargs.pop("system_prompt", None)

        default_kwargs = {"do_sample": True, "temperature": 0.3, "max_new_tokens": 512}
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs

    def message2pipeline(self, message):
        ret = []
        if hasattr(self, "system_prompt") and self.system_prompt is not None:
            ret = [
                dict(
                    role="system", content=[dict(type="text", text=self.system_prompt)]
                )
            ]
        content = []
        for m in message:
            if m["type"] == "text":
                content.append(dict(type="text", text=m["value"]))
            elif m["type"] == "image":
                content.append(dict(type="image", url=m["value"]))
        ret.append(dict(role="user", content=content))
        return ret

    def generate_inner(self, message, dataset=None):
        messages = self.message2pipeline(message)
        inputs = self.processor.apply_chat_template(
            messages,
            padding=True,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**inputs, **self.kwargs)
            generation = generation[0][input_len:]

        decoded = self.processor.tokenizer.decode(generation, skip_special_tokens=True)
        return decoded
