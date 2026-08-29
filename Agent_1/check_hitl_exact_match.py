import json
from pathlib import Path

from app.services.detected_topic_edit_reuse_feedback_store import (
    DetectedTopicEditReuseFeedbackStore,
)

RUN_ID = "job_20260816_220703_2ded88c0"

runs_root = Path("runs")
run_dir = runs_root / RUN_ID

# If your runs folder is elsewhere, change only this path.
manifest = json.loads(
    (run_dir / "pipeline_manifest.json").read_text(encoding="utf-8")
)

transcript_name = manifest["transcript_name"]

module3_path = (
    run_dir
    / "output"
    / transcript_name
    / "03_topic_mapping.json"
)

payload = json.loads(module3_path.read_text(encoding="utf-8"))
module3 = payload["module3_result"]

store = DetectedTopicEditReuseFeedbackStore()

topics = module3.get("merged_topics", [])

def topic_evidence(concept_id):
    for topic in topics:
        if topic.get("concept_id") == concept_id:
            evidence = topic.get("evidence") or []
            if not isinstance(evidence, list):
                evidence = [evidence]
            return "\n".join(
                str(x).strip()
                for x in evidence
                if str(x).strip()
            )
    return ""

checks = [
    (13, "aqa_3_2_2_selection"),
    (15, "aqa_3_4_5_fetch_execute_cycle"),
    (16, "aqa_3_1_1_algorithm_representation"),
    (17, "aqa_3_4_5_memory"),
    (19, "aqa_3_2_6_data_structures"),
]

for memory_id, concept_id in checks:
    evidence = topic_evidence(concept_id)

    print("\n--------------------------------")
    print("MEMORY:", memory_id)
    print("CONCEPT:", concept_id)
    print("CURRENT HASH:", store.evidence_hash(evidence))

    decision = store.get_decision(
        memory_id=memory_id,
        current_evidence=evidence,
        spec_version="AQA-8525-v1.2-2022-11-29",
    )

    print("MATCHED DECISION:", decision)