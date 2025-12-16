import json
import os
from typing import Any, Dict, List

import requests
from PIL import Image

from ..smp import encode_image_to_base64, get_logger, np
from .base import BaseAPI

APIBASES = {
    # Default to local vLLM-style OpenAI endpoint (matches eval_vllm.sh in SmolVLM repo).
    "LOCAL": "http://localhost:23333/v1/chat/completions",
}


class SmolVLMAPIWrapper(BaseAPI):
    """OpenAI-compatible API wrapper for SmolVLM served via vLLM.

    This mirrors the GPT/Kimi API wrappers: messages follow the OpenAI
    chat/completions schema, and images are sent as base64 ``image_url`` blocks.
    """

    is_api: bool = True

    def __init__(
        self,
        model: str = "smolvlm",
        retry: int = 5,
        key: str | None = None,
        verbose: bool = True,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        timeout: int = 120,
        api_base: str | None = None,
        max_tokens: int = 1024,
        img_size: int = -1,
        img_detail: str = "auto",
        **kwargs: Any,
    ):
        self.model = os.environ.get("SMOLVLM_API_MODEL", model)
        self.fail_msg = "Failed to obtain answer via SmolVLM API. "
        self.max_tokens = os.environ.get("SMOLVLM_API_MAX_TOKENS", max_tokens)
        self.temperature = temperature
        self.img_size = img_size
        self.img_detail = img_detail
        self.timeout = timeout

        # Prefer SmolVLM-specific env vars, then fall back to generic LMDeploy ones.
        if key is None:
            key = os.environ.get(
                "SMOLVLM_API_KEY", os.environ.get("SMOLVLM_API_KEY", None)
            )
        self.key = key

        if api_base is None:
            if "SMOLVLM_API_BASE" in os.environ and os.environ["SMOLVLM_API_BASE"]:
                api_base = os.environ["SMOLVLM_API_BASE"]
            elif "LMDEPLOY_API_BASE" in os.environ and os.environ["LMDEPLOY_API_BASE"]:
                api_base = os.environ["LMDEPLOY_API_BASE"]
            else:
                api_base = "LOCAL"

        if api_base in APIBASES:
            self.api_base = APIBASES[api_base]
        else:
            self.api_base = api_base

        self.logger = get_logger("SmolVLMAPI")
        self.logger.info(f"Using API Base: {self.api_base}; API Key: {self.key}")
        super().__init__(
            retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs
        )

    def prepare_itlist(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        assert np.all([isinstance(x, dict) for x in inputs])
        has_images = np.sum([x["type"] == "image" for x in inputs])
        if has_images:
            content_list = []
            for msg in inputs:
                if msg["type"] == "text":
                    content_list.append(dict(type="text", text=msg["value"]))
                elif msg["type"] == "image":
                    img = Image.open(msg["value"])
                    b64 = encode_image_to_base64(img, target_size=self.img_size)
                    img_struct = dict(url=f"data:image/jpeg;base64,{b64}")
                    if self.img_detail:
                        img_struct["detail"] = self.img_detail
                    content_list.append(dict(type="image_url", image_url=img_struct))
        else:
            assert all([x["type"] == "text" for x in inputs])
            text = "\n".join([x["value"] for x in inputs])
            content_list = [dict(type="text", text=text)]
        return content_list

    def prepare_inputs(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        input_msgs: List[Dict[str, Any]] = []
        if self.system_prompt is not None:
            input_msgs.append(dict(role="system", content=self.system_prompt))
        assert isinstance(inputs, list) and isinstance(inputs[0], dict)
        assert np.all(["type" in x for x in inputs]) or np.all(
            ["role" in x for x in inputs]
        ), inputs
        if "role" in inputs[0]:
            assert inputs[-1]["role"] == "user", inputs[-1]
            for item in inputs:
                input_msgs.append(
                    dict(
                        role=item["role"], content=self.prepare_itlist(item["content"])
                    )
                )
        else:
            input_msgs.append(dict(role="user", content=self.prepare_itlist(inputs)))
        return input_msgs

    def generate_inner(self, inputs: List[Dict[str, Any]], **kwargs: Any):
        input_msgs = self.prepare_inputs(inputs)
        temperature = kwargs.pop("temperature", self.temperature)
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)

        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"

        payload: Dict[str, Any] = dict(
            model=self.model,
            messages=input_msgs,
            n=1,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        response = requests.post(
            self.api_base,
            headers=headers,
            data=json.dumps(payload),
            timeout=self.timeout * 1.1,
        )
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
        answer = self.fail_msg
        try:
            resp_struct = json.loads(response.text)
            answer = resp_struct["choices"][0]["message"]["content"].strip()
        except Exception as err:
            if self.verbose:
                self.logger.error(f"{type(err)}: {err}")
                self.logger.error(
                    response.text if hasattr(response, "text") else response
                )

        return ret_code, answer, response


class SmolVLMAPI(SmolVLMAPIWrapper):
    """Thin alias to mirror other API wrappers."""

    def generate(self, message, dataset=None):
        return super(SmolVLMAPI, self).generate(message)
