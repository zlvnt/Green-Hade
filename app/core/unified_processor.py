"""
Unified Query Processor for HADE

Combines routing, context resolution, reformulation, and escalation check
into a single LLM call for efficiency.

Provider-agnostic via app.core.model.create_fast_llm() — JSON output
dijaga via Pydantic schema (llm.with_structured_output).
"""

from typing import Any, Dict, Literal, Optional
from functools import lru_cache
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field


class UnifiedProcessorOutput(BaseModel):
    """Schema output yang dipaksa via with_structured_output."""

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
    ):
        """
        Args:
            llm: LangChain BaseChatModel (provider-agnostic, dari create_fast_llm).
            prompt_template_path: Path ke prompt template file.
        """
        self.llm = llm
        self.structured_llm = llm.with_structured_output(UnifiedProcessorOutput)

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

Output dalam format struktur sesuai schema."""

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
            result_obj: UnifiedProcessorOutput = self.structured_llm.invoke(prompt)
            result = result_obj.model_dump()

            # Backward-compat field
            result["needs_reformulation"] = (
                result["reformulated_query"] != result["resolved_query"]
            )

            return result

        except Exception as e:
            print(f"ERROR: UnifiedProcessor failed: {e}")
            return self._fallback_response(query)

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
    return UnifiedProcessor(
        llm=llm,
        prompt_template_path=settings.UNIFIED_PROCESSOR_PROMPT_PATH,
    )


def process_query(query: str, history: str = "") -> Dict[str, Any]:
    """Convenience: process query via singleton processor."""
    processor = _get_unified_processor()
    return processor.process(query, history)
