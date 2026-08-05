import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from routers import chat  # noqa: E402
from schemas import ChatRequest  # noqa: E402
from services.rag import RetrievedChunk  # noqa: E402


class ChatGuardrailTests(unittest.TestCase):
    def test_money_notation_is_normalized_to_won(self):
        self.assertEqual(chat._money_values("30만원"), {300000})
        self.assertEqual(chat._money_values("300,000원"), {300000})
        self.assertEqual(chat._money_values("30~45만원"), {300000, 450000})

    def test_equivalent_money_notation_is_allowed(self):
        summary = "- 예상 비용: 300,000원 ~ 450,000원"
        answer = "예상 범위는 30만~45만원입니다."
        self.assertFalse(chat._violates_price_guardrail(answer, summary))

    def test_fabricated_money_is_rejected(self):
        summary = "- 예상 비용: 300,000원 ~ 450,000원"
        answer = "예상 비용은 500,000원입니다."
        self.assertTrue(chat._violates_price_guardrail(answer, summary))

    def test_rag_fallback_does_not_expose_unverified_price_rows(self):
        chunk = RetrievedChunk(
            content=(
                "## 교환 판단\n고정부가 깨지면 교환을 검토합니다.\n"
                "| severe | 어셈블리 교환 | 300,000~600,000원 |"
            ),
            title="사이드미러 손상",
            section="교환 판단",
            source="ai/docs/example.md",
        )
        answer = chat._fallback_answer([chunk], has_price=False)
        self.assertIn("고정부가 깨지면", answer)
        self.assertNotIn("300,000", answer)
        self.assertNotIn("600,000", answer)

    def test_price_question_uses_exact_diagnosis_range_without_llm(self):
        summary = (
            "- 부위: 전방 범퍼\n"
            "- 손상 종류: 균열\n"
            "- 수리 방식: 범퍼 교환\n"
            "- 예상 비용: 300,000원 ~ 450,000원"
        )
        answer = chat._rule_based_price_answer("예상 비용은 얼마인가요?", summary)

        self.assertIn("300,000원 ~ 450,000원", answer)
        self.assertIn("범퍼 교환", answer)
        self.assertNotIn("375,000", answer)

    def test_price_question_without_estimate_returns_no_price_answer(self):
        answer = chat._rule_based_price_answer(
            "교체 가격이 얼마인가요?",
            "- 부위: 루프\n- 손상 종류: 찌그러짐",
        )
        self.assertEqual(answer, chat.NO_PRICE_ANSWER)

    def test_non_price_question_still_uses_llm_path(self):
        answer = chat._rule_based_price_answer(
            "이 상태에서 운전해도 되나요?",
            "- 예상 비용: 300,000원 ~ 450,000원",
        )
        self.assertIsNone(answer)

    def test_price_chat_skips_rag_and_llm(self):
        payload = ChatRequest(
            session_id="test-session",
            message="예상 비용은 얼마인가요?",
            diagnosis_summary=(
                "- 부위: 전방 범퍼\n"
                "- 수리 방식: 범퍼 교환\n"
                "- 예상 비용: 300,000원 ~ 450,000원"
            ),
        )

        with (
            patch.object(chat.rag, "search") as rag_search,
            patch.object(chat.llm_client, "generate") as llm_generate,
        ):
            response = asyncio.run(chat.chat(payload))

        self.assertEqual(response.answer_mode, "rule_based")
        self.assertIn("300,000원 ~ 450,000원", response.answer)
        rag_search.assert_not_called()
        llm_generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
