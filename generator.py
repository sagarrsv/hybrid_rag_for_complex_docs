

"""
generator.py
===========================================================
Provider-swappable multimodal answer generator.

Features:
- LangChain model adapter: Gemini or Groq
- Pydantic structured output validation
- Page-level source citations
- Layout-heavy text + image generation (default)
- Optional image-only generation
- Gemini token counting for BOTH context variants
- Groq token-budget projections for BOTH variants

Retrieval and PDF extraction remain outside this module.
===========================================================
"""

import base64
import os
import time
from typing import Any, Dict, List, Literal
from transformers import AutoTokenizer

from pydantic import BaseModel, Field, ConfigDict

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
)

from config import (
    GENERATOR_MODEL_NAME,
    GENERATOR_TEMPERATURE,
    GENERATOR_MAX_TOKENS,
    PT_VISUAL,
)


# -----------------------------------------------------------
# 1. Structured answer schema
# -----------------------------------------------------------

class GeneratedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        description="Concise answer grounded in document evidence."
    )

    answerable: bool = Field(
        description="Whether the supplied pages support the answer."
    )

    citations: List[str] = Field(
        description="Citation IDs from the supplied pages, e.g. P1."
    )


# -----------------------------------------------------------
# 2. System prompt
# -----------------------------------------------------------

SYSTEM_PROMPT = """
You are a technical research-paper question-answering assistant.

Answer ONLY from the provided PDF page evidence.

Evidence rules:
1. Each source ID (P1, P2, P3, etc.) identifies ONE PDF page.
2. A layout-heavy page may contain BOTH extracted text and
   a rendered image of that SAME PDF page.
3. These are complementary representations, not two
   separate sources.
4. Use both representations when available.
5. Preserve important numbers, units, model names and
   experimental conditions.
6. Read equations, tables and plots carefully.
7. Never invent evidence or citations.
8. Donot always assume rank 1 is the best source/
9. If evidence is insufficient, set answerable=false.
10. Cite only the supplied source IDs.
11. Return a concise, self-contained answer.
"""


# -----------------------------------------------------------
# 3. LangChain model factory
# -----------------------------------------------------------

def create_model(
    provider: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
):
    """
    Create a LangChain chat model.

    The rest of the generator remains provider-independent.
    """

    if provider == "gemini":
        from langchain_google_genai import (
            ChatGoogleGenerativeAI,
        )

        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    raise ValueError(f"Unsupported provider: {provider}")


# -----------------------------------------------------------
# 4. Generator
# -----------------------------------------------------------

class MultimodalGenerator:
    def __init__(
        self,
        provider: str = "gemini",
        model_name: str = GENERATOR_MODEL_NAME,
        layout_text_enabled: bool = True,
    ):
        self.provider = provider
        self.model_name = model_name
        self.qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B")

        # True: layout-heavy pages use text + image.
        # False: layout-heavy pages use image only.
        self.layout_text_enabled = layout_text_enabled

        self.model = create_model(
            provider=self.provider,
            model_name=self.model_name,
            temperature=GENERATOR_TEMPERATURE,
            max_tokens=GENERATOR_MAX_TOKENS,
        )

        # Native JSON-schema output is supported by the
        # chosen LangChain integrations.
        # The specific model/endpoint must also support it.
        self.structured_model = self.model.with_structured_output(
            GeneratedAnswer,
            method="json_schema",
            include_raw=True,
        )
        actual_model = (
                self.model.model
                if self.provider == "gemini"
                else self.model.model_name
            )
        print("\n[Generator Verification]")
        print(f"Requested provider: {self.provider}")
        print(f"Requested model: {self.model_name}")
        print(f"Actual client class: {type(self.model).__name__}")
        print(f"Actual client model:  {actual_model}")

        # Used only for Gemini preflight token measurement.
        # This does not generate an answer.
        self._gemini_counter = None

        if provider == "gemini":
            from google import genai

            api_key = (
                os.getenv("GOOGLE_API_KEY")
                or os.getenv("GEMINI_API_KEY")
            )

            if not api_key:
                raise EnvironmentError(
                    "Set GOOGLE_API_KEY or GEMINI_API_KEY."
                )

            self._gemini_counter = genai.Client(
                api_key=api_key
            )

    # -------------------------------------------------------
    # 5. Build provider-independent multimodal messages
    # -------------------------------------------------------

    def _build_messages(
        self,
        query: str,
        context: Dict[str, Any],
        layout_text_enabled: bool,
    ):
        """
        Preserve retrieval order.

        Text-dense page:
            Header -> extracted text

        Layout-heavy page:
            Header -> optional extracted text -> image

        Images are encoded as standard LangChain
        base64 data-URL image blocks.
        """

        blocks = []

        for page in context["pages"]:
            citation_id = page["citation_id"]
            is_visual = page["page_type"] == PT_VISUAL

            header = (
                f"\n[Source {citation_id}]\n"
                f"Document: {page['doc_id']}\n"
                f"PDF page: {page['page_num']}\n"
                f"Page type: {page['page_type']}\n"
            )

            blocks.append({
                "type": "text",
                "text": header,
            })

            # Text-dense pages always include text.
            # Layout-heavy pages include text only if
            # the optional setting is enabled.
            include_text = (
                not is_visual or layout_text_enabled
            )

            if include_text and page["text"]:
                blocks.append({
                    "type": "text",
                    "text": (
                        f"Extracted text for {citation_id}:\n"
                        + page["text"]
                    ),
                })

            if is_visual:
                image_bytes = page.get("image_bytes")

                if image_bytes is None:
                    raise ValueError(
                        f"Missing image for layout-heavy "
                        f"source {citation_id}. "
                        "Build context with include_images=True."
                    )

                encoded_image = base64.b64encode(
                    image_bytes
                ).decode("ascii")

                blocks.append({
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            "data:image/png;base64,"
                            + encoded_image
                        )
                    },
                })

        blocks.append({
            "type": "text",
            "text": f"\nQuestion:\n{query}",
        })

        return [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=blocks),
        ]

    # -------------------------------------------------------
    # 6. Provider-independent approximate token projection
    # -------------------------------------------------------

    
    def _groq_projection(self,messages) -> Dict[str, int]:
        """
        Project Groq input tokens using:
            qwen tokenizer
            2,048 tokens per image

        IMPORTANT:
        This is a planning estimate, NOT Groq API usage.
        The text-token approximation may be inaccurate,
        particularly for equations and scientific notation.
        """

        text_parts = []
        image_count = 0

        for message in messages:
            content = message.content

            if isinstance(content, str):
                text_parts.append(content)
                continue

            for block in content:
                if block["type"] == "text":
                    text_parts.append(block["text"])

                elif block["type"] == "image_url":
                    image_count += 1


        full_text = "\n".join(text_parts)

        estimated_text_tokens = len(
            self.qwen_tokenizer.encode(full_text,add_special_tokens=False)
            )

        estimated_image_tokens = (
            image_count * 2048
        )

        return {
            "estimated_text_tokens": estimated_text_tokens,
            "estimated_image_tokens": estimated_image_tokens,
            "image_count": image_count,
            "estimated_input_tokens": (
                estimated_text_tokens
                + estimated_image_tokens
            ),
        }

    # -------------------------------------------------------
    # 7. Gemini native input-token measurement
    # -------------------------------------------------------

    def _count_gemini_tokens(self, messages) -> int:
        """
        Count Gemini input tokens without generating.

        Converts the same textual/image payload into native
        Google SDK parts.

        Note:
        Response-schema overhead and other API-specific
        accounting may cause slight differences from the
        generation request's reported usage.
        """

        from google.genai import types

        parts = []

        for message in messages:
            content = message.content

            if isinstance(content, str):
                parts.append(
                    types.Part.from_text(text=content)
                )
                continue

            for block in content:
                if block["type"] == "text":
                    parts.append(
                        types.Part.from_text(
                            text=block["text"]
                        )
                    )

                elif block["type"] == "image_url":
                    data_url = block["image_url"]["url"]

                    # Recover original PNG bytes from the
                    # LangChain data-URL representation.
                    image_bytes = base64.b64decode(
                        data_url.split(",", 1)[1]
                    )

                    parts.append(
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type="image/png",
                        )
                    )

        response = self._gemini_counter.models.count_tokens(
            model=self.model_name,
            contents=[
                types.Content(
                    role="user",
                    parts=parts,
                )
            ],
        )

        return int(response.total_tokens or 0)

    # -------------------------------------------------------
    # 8. Compare both context variants
    # -------------------------------------------------------

    def compare_token_modes(
        self,
        query: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calculate token usage for TWO possible contexts:

        text_image:
            Text-dense -> text
            Layout-heavy -> text + image

        image_only:
            Text-dense -> text
            Layout-heavy -> image only

        Does not call the generation endpoint.
        """

        results = {}

        for mode, include_layout_text in (
            ("text_image", True),
            ("image_only", False),
        ):
            messages = self._build_messages(
                query,
                context,
                layout_text_enabled=include_layout_text,
            )

            projection = self._groq_projection(
                messages
            )

            measured_gemini_tokens = None

            if self._gemini_counter is not None:
                measured_gemini_tokens = (
                    self._count_gemini_tokens(messages)
                )

            results[mode] = {
                "gemini_counted_input_tokens":
                    measured_gemini_tokens,
                "groq_projection": projection,
            }

        results["estimated_layout_text_saving_groq"] = (
            results["text_image"]["groq_projection"][
                "estimated_input_tokens"
            ]
            -
            results["image_only"]["groq_projection"][
                "estimated_input_tokens"
            ]
        )

        if self.provider == "gemini":
            results["gemini_input_token_difference"] = (
                results["text_image"][
                    "gemini_counted_input_tokens"
                ]
                -
                results["image_only"][
                    "gemini_counted_input_tokens"
                ]
            )

        return results

    # -------------------------------------------------------
    # 9. Actual answer generation
    # -------------------------------------------------------

    def generate(
        self,
        query: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate with the configured provider and
        record actual provider-reported token usage.

        Also calculate both token-budget alternatives.
        """

        if not context["pages"]:
            return {
                "answer": "Insufficient evidence.",
                "answerable": False,
                "citations": [],
                "usage": {},
                "token_comparison": {},
                "generation_latency": 0.0,
            }

        # Two budget comparisons, but only one generation.
        # token_comparison = self.compare_token_modes(
        #     query,
        #     context,
        # )
        token_comparison = {}

        messages = self._build_messages(
            query,
            context,
            layout_text_enabled=self.layout_text_enabled,
        )

        start = time.perf_counter()

        response = self.structured_model.invoke(
            messages
        )

        latency = time.perf_counter() - start

        # include_raw=True returns:
        # raw, parsed, parsing_error
        if response.get("parsing_error") is not None:
            raise ValueError(
                "Structured output parsing failed: "
                f"{response['parsing_error']}"
            )

        parsed = response["parsed"]

        if parsed is None:
            raise ValueError(
                "Model returned no parsed structured answer."
            )

        # Ensure the shared Pydantic schema is applied.
        answer = GeneratedAnswer.model_validate(parsed)

        # Check citations against actual source metadata.
        allowed = {
            c["citation_id"]: c
            for c in context["citations"]
        }

        if any(
            cid not in allowed
            for cid in answer.citations
        ):
            raise ValueError(
                f"Invalid citation IDs: {answer.citations}"
            )

        if answer.answerable and not answer.citations:
            raise ValueError(
                "Answerable output must cite at least one page."
            )

        resolved_citations = []

        if answer.answerable:
            resolved_citations = [
                allowed[cid]
                for cid in dict.fromkeys(answer.citations)
            ]

        # Actual model-reported token usage.
        raw = response["raw"]
        metadata = raw.usage_metadata or {}

        input_tokens = metadata.get(
            "input_tokens"
        )
        output_tokens = metadata.get(
            "output_tokens"
        )
        total_tokens = metadata.get(
            "total_tokens"
        )

        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

        return {
            "answer": answer.answer,
            "answerable": answer.answerable,
            "citations": resolved_citations,
            "usage": usage,
            "token_comparison": token_comparison,
            "generation_mode": (
                "text_image"
                if self.layout_text_enabled
                else "image_only"
            ),
            "generation_latency": latency,
        }
