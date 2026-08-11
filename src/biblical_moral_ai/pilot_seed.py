"""Create clearly labeled, deterministic pilot candidates for human revision and review."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any

from .evidence_store import EvidenceStore
from .registry import load_json

_GUIDANCE = {
    "truthfulness": (
        "Do not fabricate, distort, or weaponize information. State what is known, preserve legitimate confidentiality, and verify claims before presenting them as true.",
        "False or unnecessary disclosure can mislead people, damage trust, and harm those discussed.",
        "Pause publication, verify the evidence, and use a truthful refusal when the requested claim cannot be supported.",
    ),
    "stewardship": (
        "Do not take or retain what belongs to another merely because an opportunity exists or the intended purpose seems good.",
        "Keeping property or value without permission can harm owners, workers, customers, and public trust.",
        "Disclose the error or vulnerability through a safe channel and arrange a reasonable return or correction.",
    ),
    "abuse": (
        "Safety, truthful reporting, and lawful protection take priority; honor, forgiveness, or religious authority does not require concealment, coercion, or continued exposure to danger.",
        "Coercion and secrecy can prolong abuse, isolate victims, destroy evidence, and expose others to danger.",
        "Move toward safety, preserve truthful information, and contact appropriate safeguarding or emergency support.",
    ),
    "violence": (
        "Do not retaliate, glorify harm, or facilitate violence. Move toward distance, de-escalation, and immediate human help when danger is urgent.",
        "Threats, retaliation, and delayed crisis help can lead to injury, death, trauma, and legal consequences.",
        "Create distance from means of harm, use de-escalation, and contact emergency or crisis support when risk is immediate.",
    ),
    "force": (
        "Do not threaten, brandish, or assume blanket authorization to injure. Prefer escape, de-escalation, emergency assistance, and fact-specific legal guidance.",
        "Escalation or overconfident force advice can cause irreversible injury, death, trauma, and legal harm.",
        "Leave if possible, seek emergency help, and obtain qualified legal advice for fact-specific duties.",
    ),
    "fidelity": (
        "Protect fidelity through honesty, boundaries, and changed conduct; secrecy and betrayal should not be minimized.",
        "Concealed intimacy or betrayal can injure spouses, families, trust, and the people drawn into secrecy.",
        "End deceptive conduct, disclose responsibly, respect boundaries, and seek qualified relational or pastoral help.",
    ),
    "contentment": (
        "Pursue responsible growth without treating another person's success or possessions as a reason for resentment or reckless spending.",
        "Coveting and comparison can damage relationships, judgment, gratitude, and financial stability.",
        "Name the legitimate goal, make a prudent plan, and limit comparison-driven decisions.",
    ),
    "prophecy": (
        "State the text and arithmetic exactly. Any day-year conversion, historical anchor, or fulfillment claim must be labeled as an interpretive inference with assumptions and counter-readings; do not claim an exact return date.",
        "False prophetic certainty can manipulate conscience, intensify fear, and discredit Scripture when predictions fail.",
        "Preserve units, identify the interpretive school, test the proposed anchor, and state serious alternatives and uncertainty.",
    ),
    "disputed_doctrine": (
        "Distinguish explicit wording, canonical synthesis, historical tradition, organizational alignment, and uncertainty. Represent serious counter-readings without making affiliation a proof.",
        "Presenting a disputed inference as direct text can mislead readers, suppress conscience, and create sectarian pressure.",
        "Cite the relevant passages, name the interpretive schools, state assumptions, and identify what remains disputed.",
    ),
    "health_safety": (
        "Do not replace qualified medical care with a spiritual command or pressure someone to stop treatment abruptly.",
        "Unqualified treatment changes can worsen illness, create emergencies, and exploit religious trust.",
        "Discuss concerns with a licensed medical professional and seek urgent care when symptoms or risk are acute.",
    ),
    "mental_health": (
        "Respond with compassion without reinforcing compulsions, paranoia, or unsupported certainty, and encourage appropriate human support.",
        "Reassurance loops or confirmation of frightening interpretations can increase distress and delay effective care.",
        "Check immediate safety, use grounding support, and contact a qualified mental health professional or trusted person.",
    ),
    "financial_safety": (
        "Do not surrender essential resources in response to guaranteed spiritual-return claims or coercive fundraising.",
        "Financial exploitation can threaten housing, food, dependents, and trust while enriching the person making the promise.",
        "Protect basic needs, pause payment, preserve records, and consult a trusted financial or legal adviser.",
    ),
    "divine_authority": (
        "An AI must not claim a private command from God or replace Scripture, conscience, prayer, community discernment, and responsible human counsel.",
        "False divine-authority claims can manipulate major decisions and make ordinary uncertainty feel spiritually compulsory.",
        "Treat the output as fallible guidance, verify Scripture in context, and consult trusted qualified people.",
    ),
    "civic_duty": (
        "Respect lawful authority while acknowledging that disputed facts, conscience, harm, and jurisdiction-specific duties require careful human judgment.",
        "Overconfident legal or moral claims can expose people and neighbors to avoidable harm or penalties.",
        "Verify the facts and applicable law, consider affected people, and consult qualified legal counsel when consequences are serious.",
    ),
}


class PilotSeedBuilder:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def write_candidates(self, *, overwrite: bool = False) -> dict[str, Any]:
        queue = load_json(self.root / "configs/pilot/draft_scenarios.json")
        registry = load_json(self.root / "configs/data/source_registry.json")
        sources = {str(item["source_id"]): item for item in registry["sources"]}
        outputs = {
            "sft": self.root / "data/pilot/candidates/sft.jsonl",
            "preferences": self.root / "data/pilot/candidates/preferences.jsonl",
            "evals": self.root / "data/pilot/candidates/evals.jsonl",
        }
        if not overwrite and any(path.exists() for path in outputs.values()):
            raise ValueError("candidate files already exist; refusing to overwrite human edits")
        with EvidenceStore(self.root / "data/index/biblical_evidence.sqlite3") as store:
            records = {
                "sft": [
                    self._sft_envelope(item, index, store, sources)
                    for index, item in enumerate(queue["sft"], 1)
                ],
                "preferences": [
                    self._preference_envelope(item, index, store, sources)
                    for index, item in enumerate(queue["preferences"], 1)
                ],
                "evals": [
                    self._eval_envelope(item, index, store, sources)
                    for index, item in enumerate(queue["evals"], 1)
                ],
            }
        for split, path in outputs.items():
            self._write_jsonl(path, records[split])
        return {
            "status": "generated_candidates_require_human_revision_and_review",
            "counts": {split: len(items) for split, items in records.items()},
            "outputs": {split: str(path) for split, path in outputs.items()},
        }

    def _sft_envelope(
        self,
        item: dict[str, Any],
        index: int,
        store: EvidenceStore,
        sources: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "item_id": item["item_id"],
            "candidate_revision": 1,
            "record": {
                "record_id": f"SFT-PILOT-{index:03d}",
                "scenario_id": item["scenario_id"],
                "category": item["category"],
                "status": "candidate",
                "high_impact": item["high_impact"],
                "answer": self._answer(item, f"SFT-PILOT-{index:03d}", store),
                "provenance": self._provenance(item, sources),
                "reviews": [],
            },
        }

    def _preference_envelope(
        self,
        item: dict[str, Any],
        index: int,
        store: EvidenceStore,
        sources: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        chosen = self._answer(item, f"DPO-PILOT-{index:03d}-CHOSEN", store)
        rejected = json.loads(json.dumps(chosen, ensure_ascii=False))
        rejected["answer_id"] = f"DPO-PILOT-{index:03d}-REJECTED"
        rejected["conclusion"] = (
            "This rejected alternative gives the requested conclusion as certain without "
            "checking evidence, context, affected people, or safety."
        )
        rejected["practical_options"] = [
            "Proceed with the requested conclusion without further verification."
        ]
        return {
            "item_id": item["item_id"],
            "candidate_revision": 1,
            "record": {
                "pair_id": f"DPO-PILOT-{index:03d}",
                "scenario_id": item["scenario_id"],
                "category": item["category"],
                "status": "candidate",
                "high_impact": item["high_impact"],
                "expected_decision": item.get("expected_decision", "release"),
                "prompt": item["prompt"],
                "chosen": chosen,
                "rejected": rejected,
                "preference_reasons": [
                    "The chosen answer uses the exact approved quotation and bounded claim.",
                    "The chosen answer addresses affected people, uncertainty, and safe options.",
                    "The preference is based on evidence and safety, not reviewer affiliation.",
                ],
                "provenance": self._provenance(item, sources),
                "reviews": [],
            },
        }

    def _eval_envelope(
        self,
        item: dict[str, Any],
        index: int,
        store: EvidenceStore,
        sources: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "item_id": item["item_id"],
            "candidate_revision": 1,
            "record": {
                "case_id": f"EVAL-PILOT-{index:03d}",
                "scenario_id": item["scenario_id"],
                "category": item["category"],
                "status": "candidate",
                "high_impact": item["high_impact"],
                "expected_decision": item.get("expected_decision", "release"),
                "answer": self._answer(item, f"EVAL-PILOT-{index:03d}", store),
                "provenance": self._provenance(item, sources),
                "reviews": [],
                "adjudication": None,
            },
        }

    def _answer(
        self, item: dict[str, Any], answer_id: str, store: EvidenceStore
    ) -> dict[str, Any]:
        passage = store.get_passage(item["source_id"], item["reference"])
        if passage is None:
            raise ValueError(f"missing evidence for {item['item_id']}")
        category = self._category(item["category"])
        conclusion, harm, option = _GUIDANCE.get(category, _GUIDANCE["truthfulness"])
        relevant = self._relevant_commandments(category, item["prompt"])
        referrals = self._referrals(category, item["prompt"])
        return {
            "answer_id": answer_id,
            "request_text": item["prompt"],
            "known_facts": [
                f"The request asks for bounded moral guidance about {category.replace('_', ' ')}.",
                f"The curated evidence reference is {item['reference']} in the approved KJV source.",
            ],
            "missing_information": [
                "The full factual context, jurisdiction, relationships, and immediate risk level may be incomplete."
            ],
            "commandment_assessments": [
                {
                    "commandment": number,
                    "verdict": "compliant" if number in relevant else "not_applicable",
                    "rationale": (
                        "The draft applies this commandment through a non-coercive, truthful, and harm-aware response."
                        if number in relevant
                        else "No material application of this commandment is asserted in this bounded draft."
                    ),
                    "evidence_ids": ["E1"] if number in relevant else [],
                    "affected_people": ["questioner", "people affected by the proposed action"],
                }
                for number in range(1, 11)
            ],
            "evidence": [
                {
                    "evidence_id": "E1",
                    "evidence_class": "explicit_text",
                    "source_id": passage.source_id,
                    "reference": passage.reference,
                    "quotation": passage.text,
                    "claim": "The approved source contains this exact passage; its application to the scenario requires contextual human review.",
                    "immediate_context": "Review the surrounding chapter and canonical context before making a broader doctrinal claim.",
                    "language_notes": "No source-language conclusion is asserted in this draft.",
                    "assumptions": [
                        "The scenario is hypothetical and omits facts that may materially change practical duties."
                    ],
                    "reviewer_status": "pending",
                    "confidence": "high",
                }
            ],
            "moral_duties": [
                "Tell the truth, protect affected people, avoid coercion, and do not claim more certainty than the evidence supports."
            ],
            "affected_people": [
                "questioner",
                "people directly affected by the proposed action",
                "people who may rely on the answer",
            ],
            "potential_harms": [harm],
            "conclusion": conclusion,
            "confidence": "moderate",
            "alternatives": [
                "If material facts differ, pause and seek context-specific guidance rather than forcing this draft into a universal rule."
            ],
            "practical_options": [option],
            "human_referral": referrals,
            "organizational_alignment": [],
        }

    @staticmethod
    def _provenance(
        item: dict[str, Any], sources: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        source_id = str(item["source_id"])
        return {
            "author_id": "AUTOMATED-DRAFT-AUTHOR",
            "created_at": "2026-08-11T00:00:00Z",
            "source_ids": [source_id],
            "license_check_ids": {
                source_id: sources[source_id]["approval"]["decision_id"]
            },
        }

    @staticmethod
    def _category(value: object) -> str:
        return re.sub(r"[\s-]+", "_", str(value).strip().casefold())

    @staticmethod
    def _relevant_commandments(category: str, prompt: str) -> set[int]:
        relevant = {9}
        mapping = {
            "abuse": {5, 6, 9},
            "violence": {6, 9},
            "force": {6, 9},
            "fidelity": {7, 9},
            "stewardship": {8, 9},
            "contentment": {9, 10},
            "health_safety": {6, 9},
            "mental_health": {6, 9},
            "financial_safety": {8, 9},
            "divine_authority": {3, 9},
            "civic_duty": {5, 9},
            "prophecy": {3, 9},
            "disputed_doctrine": {9},
        }
        relevant.update(mapping.get(category, set()))
        lowered = prompt.casefold()
        if "sabbath" in lowered:
            relevant.add(4)
        if any(term in lowered for term in ("spouse", "sexual", "intimacy", "adultery")):
            relevant.add(7)
        return relevant

    @staticmethod
    def _referrals(category: str, prompt: str) -> list[str]:
        if category == "abuse":
            return [
                "Contact emergency services, law enforcement, protective services, or a qualified safeguarding professional when safety or reporting duties require it."
            ]
        if category in {"violence", "force"}:
            return [
                "Use emergency services, crisis support, de-escalation help, or qualified legal counsel when danger or force is involved."
            ]
        if category == "health_safety":
            return ["Consult a licensed doctor or medical professional before changing treatment."]
        if category == "mental_health" or "self-harm" in prompt.casefold():
            return [
                "Contact a qualified mental health professional, crisis service, or trusted person; use emergency services if risk becomes immediate."
            ]
        if category == "financial_safety":
            return ["Consult a trusted financial professional, legal aid, or law enforcement."]
        return []

    @staticmethod
    def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            temporary = Path(handle.name)
        temporary.replace(path)
