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

    def test_natural_repair_cost_question_uses_price_path(self):
        answer = chat._rule_based_price_answer(
            "수리비는 어느 정도 나올까?",
            "- 예상 비용: 250,000원 ~ 350,000원",
        )

        self.assertIn("250,000원 ~ 350,000원", answer)

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

    def test_severe_hood_repair_question_uses_diagnosis_method(self):
        summary = (
            "- 부위: 본닛(후드)\n"
            "- 손상 종류: 찌그러짐\n"
            "- 심각도: 심각 (사용자 선택값)\n"
            "- 수리 방식: 후드 교환 + 도색\n"
            "- 예상 비용: 600,000원 ~ 1,000,000원"
        )
        answer = chat._rule_based_repair_answer(
            "이 정도면 보닛 교체해야 하나요?", summary
        )

        self.assertIn("본닛(후드)", answer)
        self.assertIn("후드 교환 + 도색", answer)
        self.assertIn("실물 점검", answer)
        self.assertEqual(chat._money_values(answer), set())

    def test_moderate_hood_is_not_misreported_as_exchange(self):
        answer = chat._rule_based_repair_answer(
            "보닛 교체가 필요한가요?",
            (
                "- 부위: 본닛(후드)\n"
                "- 심각도: 중간 (사용자 선택값)\n"
                "- 수리 방식: 판금 + 전체 도색"
            ),
        )

        self.assertIn("교환 대상으로 분류되지 않", answer)
        self.assertIn("판금 + 전체 도색", answer)

    def test_exchange_forbidden_method_is_not_misclassified(self):
        answer = chat._rule_based_repair_answer(
            "쿼터패널도 교체해야 하나요?",
            (
                "- 부위: 쿼터패널\n"
                "- 심각도: 심각 (사용자 선택값)\n"
                "- 수리 방식: 절단·용접 수리 (교환 불가 부위)"
            ),
        )

        self.assertIn("단순 교환 대상이 아니", answer)
        self.assertIn("절단·용접 수리", answer)

    def test_repair_question_without_method_keeps_llm_path(self):
        answer = chat._rule_based_repair_answer(
            "루프를 교체해야 하나요?",
            "- 부위: 루프\n- 심각도: 심각 (사용자 선택값)",
        )
        self.assertIsNone(answer)

    def test_unrelated_question_keeps_llm_path(self):
        answer = chat._rule_based_repair_answer(
            "이 상태에서 운전해도 되나요?",
            "- 부위: 본닛(후드)\n- 수리 방식: 후드 교환 + 도색",
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

    def test_korean_retry_uses_shorter_timeout_budget(self):
        payload = ChatRequest(
            session_id="test-session",
            message="수리할 때 주의할 점을 알려주세요.",
            diagnosis_summary="- 부위: 본닛(후드)\n- 손상 종류: 찌그러짐",
        )

        with (
            patch.object(chat.rag, "search", return_value=[]),
            patch.object(
                chat.llm_client,
                "generate",
                side_effect=["这是中文回答。", "한국어 답변입니다."],
            ) as llm_generate,
        ):
            response = asyncio.run(chat.chat(payload))

        self.assertTrue(response.used_llm)
        self.assertEqual(response.answer_mode, "llm")
        self.assertEqual(llm_generate.call_count, 2)
        self.assertEqual(
            llm_generate.call_args_list[1].kwargs["timeout"],
            chat.llm_client.RETRY_TIMEOUT,
        )

    def test_repair_decision_chat_skips_rag_and_llm(self):
        payload = ChatRequest(
            session_id="test-session",
            message="이 정도면 보닛 교체해야 하나요?",
            diagnosis_summary=(
                "- 부위: 본닛(후드)\n"
                "- 심각도: 심각 (사용자 선택값)\n"
                "- 수리 방식: 후드 교환 + 도색\n"
                "- 예상 비용: 600,000원 ~ 1,000,000원"
            ),
        )

        with (
            patch.object(chat.rag, "search") as rag_search,
            patch.object(chat.llm_client, "generate") as llm_generate,
        ):
            response = asyncio.run(chat.chat(payload))

        self.assertEqual(response.answer_mode, "rule_based")
        self.assertFalse(response.used_llm)
        self.assertFalse(response.rag_used)
        self.assertIn("후드 교환 + 도색", response.answer)
        self.assertNotIn("600,000원", response.answer)
        rag_search.assert_not_called()
        llm_generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
