import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from routers import chat  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()
