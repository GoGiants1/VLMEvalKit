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

MMBENCH_DATASETS = {
    "MMBench_DEV_EN",
    "MMBench_TEST_EN",
    "MMBench_DEV_CN",
    "MMBench_TEST_CN",
    "MMBench",
    "MMBench_CN",
    "MMBench_DEV_EN_V11",
    "MMBench_DEV_CN_V11",
    "MMBench_TEST_EN_V11",
    "MMBench_TEST_CN_V11",
    "MMBench_V11",
    "MMBench_CN_V11",
    "CCBench",
}

MMMU_DATASETS = {"MMMU_DEV_VAL", "MMMU_TEST"}
MATHVISTA_DATASETS = {"MathVista_MINI"}
CHARTQA_DATASETS = {"ChartQA_TEST"}
DOCVQA_DATASETS = {"DocVQA_VAL", "DocVQA_TEST"}
TEXTVQA_DATASETS = {"TextVQA_VAL", "TextVQA_TEST"}

BRIEF_ANSWER_DATASETS = {
    "MME",
    "MMVet",
    "OCRVQA_TEST",
    "OCRVQA_TESTCORE",
    "InfoVQA_VAL",
    "InfoVQA_TEST",
    "OCRBench",
}

HALLUSION_DATASETS = {"HallusionBench"}

PUREMCQ_DATASETS = {
    "MMStar",
    "SEEDBench_IMG",
    "AI2D_TEST",
    "ScienceQA_VAL",
    "ScienceQA_TEST",
}

VIDEO_DATASETS = {
    "MMBench-Video",
    "MLVU",
    "MLVU_MCQ",
    "MLVU_OpenEnded",
    "TempCompass",
    "TempCompass_MCQ",
    "TempCompass_Captioning",
    "TempCompass_YorN",
    "MVBench",
    "MVBench_MP4",
    "Video-MME",
    "LongVideoBench",
}


def _concat_text_items(inputs: List[Dict[str, Any]]) -> str:
    return "".join(
        str(item["value"]).strip() for item in inputs if item.get("type") == "text"
    )


def _normalize_prompt_version(version: str | None, default: str = "v2") -> str:
    if version is None:
        return default
    normalized = str(version).strip().lower()
    if normalized in {"v1", "1"}:
        return "v1"
    if normalized in {"v2", "2"}:
        return "v2"
    return default


def _replace_first_n_occurrences(
    text: str, old: str, new: str, n: int
) -> tuple[str, int]:
    if n <= 0 or not old or old == new:
        return text, 0

    replaced = 0
    cursor = 0
    parts: list[str] = []
    while replaced < n:
        idx = text.find(old, cursor)
        if idx == -1:
            break
        parts.append(text[cursor:idx])
        parts.append(new)
        cursor = idx + len(old)
        replaced += 1
    parts.append(text[cursor:])
    return "".join(parts), replaced


def _build_smolvlm2_style_video_prompt(
    inputs: List[Dict[str, Any]],
    dataset: str,
    sampling_frames: int = 64,
    add_timestamps: bool = True,
) -> tuple[str, List[Dict[str, Any]]]:
    # Mirrors `SmolVLM2.build_prompt_video` in `vlmeval/vlm/smolvlm.py`.
    prompt_parts: list[str] = []
    image_blocks: list[list[Dict[str, Any]]] = []
    selected_images: list[Dict[str, Any]] = []

    system_message = next(
        (
            msg
            for msg in inputs
            if msg.get("type") == "text" and msg.get("role") == "system"
        ),
        None,
    )

    if system_message:
        prompt_parts.extend(
            [
                "<|im_start|>System:",
                str(system_message.get("value", "")),
                "<end_of_utterance>\n",
            ]
        )
    else:
        prompt_parts.extend(
            [
                "<|im_start|>System:",
                "pay attention to the video and answer the question",
                "<end_of_utterance>\n",
            ]
        )

    prompt_parts.extend(
        ["<|im_start|>User:", "Here are some frames sampled from a video:\n"]
    )

    text_messages: list[Dict[str, Any]] = []
    current_block: list[Dict[str, Any]] = []

    for msg in inputs:
        if msg.get("type") == "image":
            current_block.append(msg)
        else:
            if current_block:
                image_blocks.append(current_block)
                current_block = []
            if msg.get("role") != "system":
                text_messages.append(msg)

    if current_block:
        image_blocks.append(current_block)

    for block in image_blocks:
        if not block:
            continue
        if len(block) > sampling_frames:
            frame_indices = np.linspace(
                0, len(block) - 1, sampling_frames, dtype=int
            ).tolist()
            trimmed_block = [block[i] for i in frame_indices]
            block_timestamps = [f"{i // 60:02}:{i % 60:02}" for i in frame_indices]
        else:
            trimmed_block = block
            block_timestamps = [f"{i // 60:02}:{i % 60:02}" for i in range(len(block))]

        for img, ts in zip(trimmed_block, block_timestamps):
            ts_str = f"{ts}" if add_timestamps else ""
            prompt_parts.extend([f"Frame from {ts_str}:", "<image>"])
            selected_images.append(img)

        prompt_parts.append("\n")

    for msg in text_messages:
        prompt_parts.append(str(msg.get("value", "")).strip())

    prompt_parts.append("<end_of_utterance>")
    prompt_parts.append("\nAssistant:")

    prompt = " ".join(prompt_parts)

    if dataset in ["MLVU_MCQ", "MLVU_OpenEnded", "LongVideoBench"]:
        prompt = prompt.replace(
            "Options:",
            "respond ONLY with one of the multiple choice letter options (A/B/C/D):",
        )
    elif dataset in ["TempCompass_MCQ", "TempCompass_Captioning", "TempCompass_YorN"]:
        if dataset == "TempCompass_MCQ":
            prompt = prompt.replace("Options:", "Choices:")
            prompt = prompt.replace(
                "Please select the correct answer from the options above.",
                "Answer with the letter.",
            )
    elif dataset in ["MVBench", "MVBench_MP4"]:
        if "Options:" in prompt:
            prompt = prompt.replace(
                "Options:",
                "respond ONLY with one of the multiple choice letter options (A/B/C/D):",
            )
            prompt = prompt.replace("Best option:(", "Answer:")
    elif dataset in ["Video-MME"]:
        if "Options:" in prompt:
            prompt = prompt.replace("Options:", "Choices:")
            prompt = prompt.replace(
                "Please select the correct answer from the options above.",
                "Answer with the letter.",
            )
    elif dataset in ["MLVU", "MMBench-Video", "TempCompass"]:
        pass

    return prompt, selected_images


def apply_dataset_prompting(
    inputs: List[Dict[str, Any]],
    dataset: str | None,
    *,
    video_prompt_version: str | None = "v2",
) -> List[Dict[str, Any]]:
    """Apply dataset-specific prompt formatting.

    Mirrors the prompt formatting logic in `vlmeval/vlm/smolvlm.py` (SmolVLM & SmolVLM2).
    """

    if not dataset:
        return inputs

    dataset = str(dataset)
    if not inputs:
        return inputs

    # Keep original non-text items in-order (images/others), rebuild a single text item when needed.
    non_text_items: List[Dict[str, Any]] = [
        item.copy() for item in inputs if item.get("type") != "text"
    ]

    if dataset in MMBENCH_DATASETS:
        replace_mapping = {
            "\nOptions:": "\nChoices:",
            "Please select the correct answer from the options above.": "Answer with a letter.",
        }
        instruction = _concat_text_items(inputs)
        for k, v in replace_mapping.items():
            instruction = instruction.replace(k, v)
        instruction = instruction.strip()

        if instruction.startswith("Hint:"):
            try:
                hint, question = instruction.split("\nQuestion:", 1)
                question, choices = question.split("\nChoices:", 1)
                instruction = (
                    "Question:" + question + "\n" + hint + "\nChoices:" + choices
                )
            except ValueError:
                pass

        instruction += "\nAnswer:"
        return non_text_items + [dict(type="text", value=instruction)]

    if dataset in MMMU_DATASETS:
        replace_mapping = {
            "Question:": "",
            "Please select the correct answer from the options above.": "Answer with the letter.",
            "\nOptions:": "\nChoices:",
        }
        instruction = _concat_text_items(inputs)
        for k, v in replace_mapping.items():
            instruction = instruction.replace(k, v)
        instruction = instruction.strip()

        images = [item.copy() for item in inputs if item.get("type") == "image"]
        image_refs = "".join(f" <image {idx}> " for idx in range(1, len(images) + 1))
        instruction = f"{image_refs}{instruction}"
        if "A." in instruction and "B." in instruction:
            instruction += "\nAnswer:"

        out: List[Dict[str, Any]] = [dict(type="text", value="Question: ")]
        for idx, image_item in enumerate(images, start=1):
            out.append(dict(type="text", value=f"<image {idx}>:"))
            out.append(image_item)
            out.append(dict(type="text", value="\n"))
        out.append(dict(type="text", value=instruction))
        return out

    if dataset in MATHVISTA_DATASETS:
        replace_mapping = {
            "(A) ": "A. ",
            "(B) ": "B. ",
            "(C) ": "C. ",
            "(D) ": "D. ",
            "(E) ": "E. ",
            "(F) ": "F. ",
            "(G) ": "G. ",
            "(H) ": "H. ",
            "\nOptions:": "\nChoices:",
            "Hint: ": "",
        }
        instruction = _concat_text_items(inputs)
        for k, v in replace_mapping.items():
            instruction = instruction.replace(k, v)
        instruction = instruction.strip()
        if "A." in instruction and "B." in instruction:
            instruction += "\nAnswer:"
        return non_text_items + [dict(type="text", value=instruction)]

    if dataset in CHARTQA_DATASETS:
        instruction_prefix = (
            "For the question below, follow the following instructions:\n"
            "-The answer should contain as few words as possible.\n"
            "-Don’t paraphrase or reformat the text you see in the image.\n"
            "-Answer a binary question with Yes or No.\n"
            "-When asked to give a numerical value, provide a number like 2 instead of Two.\n"
            "-If the final answer has two or more items, provide it in the list format like [1, 2].\n"
            "-When asked to give a ratio, give out the decimal value like 0.25 instead of 1:4.\n"
            "-When asked to give a percentage, give out the whole value like 17 instead of decimal like 0.17%.\n"
            "-Don’t include any units in the answer.\n"
            "-Do not include any full stops at the end of the answer.\n"
            "-Try to include the full label from the graph when asked about an entity.\n"
            "Question: "
        )
        question = _concat_text_items(inputs)
        return non_text_items + [dict(type="text", value=instruction_prefix + question)]

    if dataset in DOCVQA_DATASETS:
        instruction_prefix = (
            "Give a short and terse answer to the following question. "
            "Do not paraphrase or reformat the text you see in the image. Do not include any full stops. "
            "Just give the answer without additional explanation. Question: "
        )
        question = _concat_text_items(inputs)
        return non_text_items + [dict(type="text", value=instruction_prefix + question)]

    if dataset in TEXTVQA_DATASETS:
        instruction_prefix = (
            "Answer the following question about the image using as few words as possible. "
            "Follow these additional instructions:\n"
            "-Always answer a binary question with Yes or No.\n"
            "-When asked what time it is, reply with the time seen in the image.\n"
            "-Do not put any full stops at the end of the answer.\n"
            "-Do not put quotation marks around the answer.\n"
            "-An answer with one or two words is favorable.\n"
            "-Do not apply common sense knowledge. The answer can be found in the image.\n"
            "Question: "
        )
        question = _concat_text_items(inputs)
        return non_text_items + [dict(type="text", value=instruction_prefix + question)]

    if dataset in BRIEF_ANSWER_DATASETS:
        instruction = _concat_text_items(inputs).strip()
        instruction += "\nGive a very brief answer."
        return non_text_items + [dict(type="text", value=instruction)]

    if dataset in HALLUSION_DATASETS:
        instruction = _concat_text_items(inputs).strip()
        instruction += "\nAnswer yes or no."
        return non_text_items + [dict(type="text", value=instruction)]

    if dataset in PUREMCQ_DATASETS:
        replace_mapping = {
            "\nOptions:": "\nChoices:",
            "Please select the correct answer from the options above.": "Answer with the letter.",
        }
        instruction = _concat_text_items(inputs)
        for k, v in replace_mapping.items():
            instruction = instruction.replace(k, v)
        instruction = instruction.strip()
        instruction += "\nAnswer:"
        return non_text_items + [dict(type="text", value=instruction)]

    if dataset in VIDEO_DATASETS:
        version = _normalize_prompt_version(video_prompt_version, default="v1")
        if version == "v1":
            instruction = _concat_text_items(inputs)
            if dataset in {"MLVU_MCQ", "MLVU_OpenEnded", "LongVideoBench"}:
                instruction = instruction.replace(
                    "Options:",
                    "respond ONLY with one of the multiple choice letter options (A/B/C/D):",
                )
            elif dataset == "TempCompass_MCQ":
                instruction = instruction.replace("Options:", "Choices:")
                instruction = instruction.replace(
                    "Please select the correct answer from the options above.",
                    "Answer with the letter.",
                )
            elif dataset in {"MVBench", "MVBench_MP4"} and "Options:" in instruction:
                instruction = instruction.replace(
                    "Options:",
                    "respond ONLY with one of the multiple choice letter options (A/B/C/D):",
                )
                instruction = instruction.replace("Best option:(", "Answer:")
            elif dataset == "Video-MME" and "Options:" in instruction:
                instruction = instruction.replace("Options:", "Choices:")
                instruction = instruction.replace(
                    "Please select the correct answer from the options above.",
                    "Answer with the letter.",
                )
            else:
                return inputs

            return non_text_items + [dict(type="text", value=instruction)]

        prompt, selected_images = _build_smolvlm2_style_video_prompt(inputs, dataset)
        if not selected_images:
            return inputs

        placeholder = "<SMOLVLM_API_IMAGE>"
        prompt_with_placeholders, replaced = _replace_first_n_occurrences(
            prompt, "<image>", placeholder, len(selected_images)
        )
        if replaced != len(selected_images):
            return inputs

        segments = prompt_with_placeholders.split(placeholder)
        if len(segments) != len(selected_images) + 1:
            return inputs

        out: List[Dict[str, Any]] = []
        for seg, image_item in zip(segments, selected_images):
            if seg:
                out.append(dict(type="text", value=seg))
            if image_item is not None:
                out.append(image_item.copy())
        last = segments[-1]
        if last:
            out.append(dict(type="text", value=last))
        return out

    return inputs


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
        video_prompt_version: str | None = "v1",
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
        env_prompt_version = os.environ.get(
            "SMOLVLM_API_PROMPT_VERSION"
        ) or os.environ.get("SMOLVLM_API_VIDEO_PROMPT_VERSION")
        self.video_prompt_version = _normalize_prompt_version(
            env_prompt_version
            if env_prompt_version is not None
            else video_prompt_version,
            default="v1",
        )
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
        dataset = kwargs.pop("dataset", None)
        video_prompt_version = kwargs.pop("video_prompt_version", None)
        prompt_version = kwargs.pop("prompt_version", None)
        if video_prompt_version is None and prompt_version is not None:
            video_prompt_version = prompt_version
        if video_prompt_version is None:
            video_prompt_version = self.video_prompt_version

        inputs = apply_dataset_prompting(
            inputs, dataset, video_prompt_version=video_prompt_version
        )
        input_msgs = self.prepare_inputs(inputs)
        temperature = kwargs.pop("temperature", self.temperature)
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)

        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"

        stop_token_ids = [0, 2, 49191]  # <endoftext>, <|im_end|>, <end_of_utterance>
        payload: Dict[str, Any] = dict(
            model=self.model,
            messages=input_msgs,
            n=1,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_token_ids=stop_token_ids,
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
        return super(SmolVLMAPI, self).generate(message, dataset=dataset)
