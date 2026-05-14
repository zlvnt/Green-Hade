"""
Unified Query Processor for HADE

Combines routing, context resolution, reformulation, and escalation check
into a single LLM call for efficiency.

Dispatch by provider:
- Provider tool-strong (anthropic/google) → with_structured_output (native
  tool_use), reliability optimal di provider itu.
- Provider tool-marginal (minimax dll) → fallback ke JSON-mode prompting +
  manual parse via Pydantic. Lihat learn/tool-use-protocol-emission.md.

Prompt template sama untuk dua path — JSON spec di prompt redundant tapi
harmless di native mode.
"""

import json
import re
from typing import Any, Dict, Literal, Optional
from functools import lru_cache
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field, ValidationError

from app.core.model import extract_reply


PROVIDERS_WITH_NATIVE_STRUCTURED_OUTPUT = {"anthropic", "google"}


class UnifiedProcessorOutput(BaseModel):
    """Schema output. Dipake dua path: with_structured_output (native) +
    Pydantic validate hasil JSON parse (fallback)."""

    routing_decision: Literal["direct", "docs"] = Field(
        description="direct = langsung reply tanpa cari KB; docs = cari di KB dulu"
    )
    resolved_query: str = Field(description="Intent user yang dipahami")
    reformulated_query: str = Field(
        description="Query optimal untuk search KB (sama dengan resolved_query kalau direct)"
    )
    escalate: bool = Field(description="True jika harus diteruskan ke CS manusia")
    escalation_reason: str = Field(
        default="",
        description="Alasan eskalasi kalau escalate=true. String kosong kalau escalate=false."
    )
    reasoning: str = Field(description="Penjelasan singkat keputusan, max 20 kata")


class UnifiedProcessor:
    """
    Unified agent yang handle:
    1. Routing decision (direct/docs)
    2. Query reformulation (optimize for retrieval)
    3. Escalation check (needs human?)

    Single LLM call for efficiency.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        prompt_template_path: Optional[str] = None,
        use_native_structured: bool = False,
    ):
        """
        Args:
            llm: LangChain BaseChatModel (provider-agnostic, dari create_fast_llm).
            prompt_template_path: Path ke prompt template file.
            use_native_structured: True kalau provider support tool_use reliable
                (anthropic/google). False = fallback ke prompt+parse (minimax dll).
        """
        self.llm = llm
        self.use_native_structured = use_native_structured
        self.structured_llm = (
            llm.with_structured_output(UnifiedProcessorOutput)
            if use_native_structured
            else None
        )

        if prompt_template_path:
            self.prompt_template = self._load_prompt_template(prompt_template_path)
        else:
            self.prompt_template = self._get_default_prompt()

    def _load_prompt_template(self, template_path: str) -> str:
        """Load prompt template from file."""
        path = Path(template_path)
        if not path.exists():
            path = Path(__file__).parent.parent.parent / template_path

        if not path.exists():
            print(f"WARNING: Prompt template not found: {template_path}, using default")
            return self._get_default_prompt()

        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _get_default_prompt(self) -> str:
        """Default prompt template (fallback)."""
        return """Kamu adalah Query Processor untuk HADE, AI Customer Service Greenhouse (hidroponik Indonesia).

=== TUJUANMU ===
Analisis query user dan tentukan strategi respons:
- routing="direct" → langsung generate respons tanpa cari referensi
- routing="docs" → cari informasi di knowledge base dulu
- escalate=true → teruskan ke CS manusia

=== INPUT ===
Query: {query}
History: {history}

=== ANALISIS (3 STEP) ===

STEP 1 - ROUTING:
Tentukan routing berdasarkan intent user (gunakan history jika relevan).
- "direct": greeting, acknowledgment, terima kasih, chitchat ringan
- "docs": pertanyaan produk, kebijakan, prosedur, return/refund, garansi, komplain

STEP 2 - REFORMULATION (jika routing="docs"):
Optimalkan query untuk pencarian knowledge base.

STEP 3 - ESCALATION CHECK:
Escalate=true jika SALAH SATU:
- B2B inquiry: volume/bulk, supply rutin/kontinuitas, custom varian, nego harga, identify sebagai restoran/hotel/reseller
- Komplain serius: produk rusak/busuk, pengiriman hilang, tagihan salah
- User minta CS/manusia/owner langsung
- Di luar kapabilitas bot

=== OUTPUT FORMAT ===
Bales HANYA dengan JSON valid (tanpa markdown fence, tanpa teks lain), schema:
{{
  "routing_decision": "direct" | "docs",
  "resolved_query": "intent user yang dipahami",
  "reformulated_query": "query optimal untuk search KB (sama dengan resolved_query kalau direct)",
  "escalate": true | false,
  "escalation_reason": "alasan singkat kalau escalate=true, kosong kalau false",
  "reasoning": "penjelasan singkat keputusan, max 20 kata"
}}"""

    def process(self, query: str, history: str = "") -> Dict[str, Any]:
        """
        Process query lewat unified pipeline.

        Returns dict dengan keys:
            routing_decision, resolved_query, needs_reformulation,
            reformulated_query, escalate, escalation_reason, reasoning.
        """
        prompt = self.prompt_template.format(
            query=query,
            history=history or "Tidak ada history percakapan sebelumnya"
        )

        try:
            if self.use_native_structured:
                result_obj = self.structured_llm.invoke(prompt)
                if result_obj is None:
                    raise ValueError(
                        "structured_llm returned None — provider didn't emit tool_use"
                    )
            else:
                ai_msg = self.llm.invoke(prompt)
                text, reasoning = extract_reply(ai_msg)
                if reasoning:
                    print(f"🧠 UNIFIED REASONING:\n{reasoning}\n")
                payload = self._parse_json(text)
                result_obj = UnifiedProcessorOutput.model_validate(payload)

            result = result_obj.model_dump()

            # Backward-compat field
            result["needs_reformulation"] = (
                result["reformulated_query"] != result["resolved_query"]
            )

            return result

        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            print(f"ERROR: UnifiedProcessor parse failed: {type(e).__name__}: {e}")
            return self._fallback_response(query)
        except Exception as e:
            print(f"ERROR: UnifiedProcessor failed: {type(e).__name__}: {e}")
            return self._fallback_response(query)

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        """
        Extract JSON dari LLM output. Toleransi:
        - Markdown fence ```json ... ```
        - Teks pre/post-amble di luar JSON object
        """
        if not text:
            raise ValueError("Empty LLM response")

        # Strip markdown fence kalau ada
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            return json.loads(fence_match.group(1))

        # Cari JSON object pertama dengan brace matching
        start = text.find("{")
        if start == -1:
            raise ValueError(f"No JSON object found in response: {text[:200]}")

        # Naive brace counter (cukup buat output kita yang flat)
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i + 1])

        raise ValueError(f"Unbalanced JSON braces in response: {text[:200]}")

    def _fallback_response(self, query: str) -> Dict[str, Any]:
        """Fallback kalau LLM gagal — safe default ke RAG."""
        return {
            "routing_decision": "docs",
            "resolved_query": query,
            "needs_reformulation": False,
            "reformulated_query": query,
            "escalate": False,
            "escalation_reason": "",
            "reasoning": "Fallback response due to processing error"
        }


@lru_cache(maxsize=1)
def _get_unified_processor() -> UnifiedProcessor:
    """Singleton UnifiedProcessor."""
    from app.config import settings
    from app.core.model import create_fast_llm

    llm = create_fast_llm(temperature=settings.UNIFIED_PROCESSOR_TEMPERATURE)
    use_native = settings.MODEL_PROVIDER in PROVIDERS_WITH_NATIVE_STRUCTURED_OUTPUT
    return UnifiedProcessor(
        llm=llm,
        prompt_template_path=settings.UNIFIED_PROCESSOR_PROMPT_PATH,
        use_native_structured=use_native,
    )


def process_query(query: str, history: str = "") -> Dict[str, Any]:
    """Convenience: process query via singleton processor."""
    processor = _get_unified_processor()
    return processor.process(query, history)
