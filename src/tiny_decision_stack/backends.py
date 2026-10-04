from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ScoredChoice:
    choice: str
    confidence: float
    probabilities: dict[str, float]


class SemanticBackend(Protocol):
    def normalize(self, state: str, question: str, options: dict[str, str]) -> dict[str, Any]: ...

    def clarify(
        self,
        state: str,
        normalized: dict[str, Any],
        question: str,
        options: dict[str, str],
        probabilities: dict[str, float],
    ) -> dict[str, Any]: ...


class DecisionBackend(Protocol):
    def decide(self, state: str | dict[str, Any], question: str, options: dict[str, str]) -> ScoredChoice: ...


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("LFM did not return a JSON object")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("LFM output must be a JSON object")
    return value


class LocalLFMBackend:
    """Lazy local Transformers backend for LiquidAI/LFM2.5-350M."""

    def __init__(self, model_id: str = "LiquidAI/LFM2.5-350M", device: str | None = None):
        self.model_id = model_id
        self.device = device or None
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs: dict[str, Any] = {"torch_dtype": "auto"}
        kwargs["device_map"] = {"": self.device} if self.device else "auto"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)

    def _generate_json(self, prompt: str) -> dict[str, Any]:
        self._load()
        assert self._model is not None and self._tokenizer is not None
        messages = [
            {
                "role": "system",
                "content": "Return exactly one valid JSON object. Do not add markdown or commentary.",
            },
            {"role": "user", "content": prompt},
        ]
        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        output = self._model.generate(
            **inputs,
            max_new_tokens=384,
            do_sample=False,
            repetition_penalty=1.02,
        )
        generated = output[0][inputs["input_ids"].shape[-1] :]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        return _extract_json(text)

    def normalize(self, state: str, question: str, options: dict[str, str]) -> dict[str, Any]:
        return self._generate_json(
            "You are a semantic preprocessor for a separate decision model. Do NOT choose an option. "
            "Preserve only decision-relevant evidence and uncertainty.\n\n"
            f"QUESTION:\n{question}\n\n"
            f"OPTIONS:\n{json.dumps(options, ensure_ascii=False)}\n\n"
            f"RAW STATE:\n{state}\n\n"
            "Return this schema exactly: "
            '{"facts":[],"constraints":[],"risks":[],"evidence_for":{},'
            '"evidence_against":{},"missing_information":[]}.'
        )

    def clarify(
        self,
        state: str,
        normalized: dict[str, Any],
        question: str,
        options: dict[str, str],
        probabilities: dict[str, float],
    ) -> dict[str, Any]:
        return self._generate_json(
            "A separate calibrated decision model was uncertain. Re-express the evidence more explicitly, "
            "without selecting an answer and without inventing facts. Highlight conflicts and missing information.\n\n"
            f"QUESTION:\n{question}\n\n"
            f"OPTIONS:\n{json.dumps(options, ensure_ascii=False)}\n\n"
            f"RAW STATE:\n{state}\n\n"
            f"CURRENT STRUCTURED STATE:\n{json.dumps(normalized, ensure_ascii=False)}\n\n"
            f"CURRENT OPTION PROBABILITIES:\n{json.dumps(probabilities)}\n\n"
            "Return exactly the same schema: "
            '{"facts":[],"constraints":[],"risks":[],"evidence_for":{},'
            '"evidence_against":{},"missing_information":[]}.'
        )


class LocalOpenDeciderBackend:
    """Lazy local backend using the official opendecider Python package."""

    def __init__(self, model_id: str = "manjunathshiva/opendecider-nano", device: str | None = None):
        self.model_id = model_id
        self.device = device or None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from opendecider import load

        kwargs: dict[str, Any] = {}
        if self.device:
            kwargs["device"] = self.device
        self._model = load(self.model_id, **kwargs)

    def decide(self, state: str | dict[str, Any], question: str, options: dict[str, str]) -> ScoredChoice:
        self._load()
        result = self._model.system_one(
            state,
            {
                "decision": {
                    "type": "choice",
                    "instructions": question,
                    "criteria": options,
                }
            },
        )
        answer = result["answers"]["decision"]
        probabilities = {str(k): float(v) for k, v in answer["probabilities"].items()}
        return ScoredChoice(
            choice=str(answer["choice"]),
            confidence=float(answer["confidence"]),
            probabilities=probabilities,
        )
