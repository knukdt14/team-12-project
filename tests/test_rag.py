import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from services import rag  # noqa: E402


class RAGTests(unittest.TestCase):
    def test_documents_are_available_without_vectorstore(self):
        status = rag.status()
        self.assertTrue(status["ready"])
        self.assertEqual(status["documents"], 12)
        self.assertGreater(status["chunks"], 12)

    def test_existing_vectorstore_is_reported_as_pending_before_lazy_load(self):
        with (
            patch.object(rag, "VECTORSTORE_DIR") as vectorstore_dir,
            patch.object(rag, "_vectorstore", None),
            patch.object(rag, "_vectorstore_attempted", False),
        ):
            vectorstore_dir.exists.return_value = True
            status = rag.status()

        self.assertEqual(status["mode"], "hybrid_pending")
        self.assertFalse(status["semantic_ready"])
        self.assertTrue(status["vectorstore_present"])

    def test_diagnosis_context_resolves_generic_follow_up(self):
        results = rag.search(
            "그럼 교체해야 하나요?",
            diagnosis_context=(
                "- 부위: 앞범퍼\n- 손상 종류: 균열\n- 심각도: 심각\n"
                "- 수리 방식: 범퍼 교환"
            ),
        )
        self.assertTrue(results)
        sources = {item.source for item in results}
        self.assertTrue(
            any("05_범퍼" in source or "04_균열" in source for source in sources),
            sources,
        )

    def test_unrelated_greeting_does_not_force_irrelevant_document(self):
        self.assertEqual(rag.search("안녕하세요"), [])

    def test_front_matter_is_metadata_not_answer_content(self):
        results = rag.search("앞범퍼가 깨졌는데 교체해야 하나요?")
        self.assertTrue(results)
        self.assertTrue(all("doc_id:" not in item.content for item in results))
        self.assertTrue(all(not item.content.startswith("---") for item in results))

    def test_natural_korean_inflections_retrieve_repair_docs(self):
        scratch = rag.search("차가 긁혔어요")
        scratched = rag.search("차를 긁었어요")
        door_ding = rag.search("차 문을 찍었어요")
        roof = rag.search("루프가 찌그러졌어요")
        pillar = rag.search("필러가 찌그러졌어요")
        side_step = rag.search("사이드스텝이 찌그러졌어요")
        self.assertTrue(any("03_스크래치" in item.source for item in scratch))
        self.assertTrue(any("03_스크래치" in item.source for item in scratched))
        self.assertTrue(any("02_덴트" in item.source for item in door_ding))
        self.assertTrue(any("02_덴트" in item.source for item in roof))
        self.assertTrue(any("02_덴트" in item.source for item in pillar))
        self.assertTrue(any("02_덴트" in item.source for item in side_step))

    def test_space_removal_does_not_create_false_surcharge_match(self):
        results = rag.search("펄 색상 수입차 할증이 있나요?")
        self.assertTrue(results)
        self.assertIn("10_특수도장", results[0].source)
        self.assertFalse(any("12_휠" in item.source for item in results))

    def test_short_crack_alias_does_not_match_money_words(self):
        special_paint = rag.search("특수색 할증 금액 알려줘")
        insurance = rag.search("보험금은 어떻게 처리하나요?")
        fee = rag.search("요금 알려줘")

        self.assertIn("10_특수도장", special_paint[0].source)
        self.assertIn("08_보험", insurance[0].source)
        self.assertFalse(any("04_균열" in item.source for item in special_paint))
        self.assertFalse(any("04_균열" in item.source for item in insurance))
        self.assertFalse(any("04_균열" in item.source for item in fee))

        actual_crack = rag.search("금요일에 앞유리에 금이 갔어요")
        self.assertTrue(
            any(
                "04_균열" in item.source or "06_유리" in item.source
                for item in actual_crack
            )
        )

    def test_query_intent_prioritizes_the_decision_or_pricing_section(self):
        dent = rag.search("후드 찌그러짐 severe 교환해야 하나요?")
        paint = rag.search("펄 도장 추가 비용 얼마야?")

        self.assertIn("02_덴트", dent[0].source)
        self.assertIn("선택 기준", dent[0].section)
        self.assertIn("10_특수도장", paint[0].source)
        self.assertIn("조정값", paint[0].section)

    def test_off_domain_query_does_not_invoke_semantic_search(self):
        with patch.object(rag, "_semantic_candidates") as semantic:
            self.assertEqual(rag.search("오늘 서울 날씨가 어때요?"), [])
        semantic.assert_not_called()


if __name__ == "__main__":
    unittest.main()
