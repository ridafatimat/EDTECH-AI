from sqlalchemy import text

from app.db.session import session_scope

query = text("""
SELECT
    id,
    memory_id,
    pipeline_run_id,
    source_concept_id,
    decision,
    reviewer_reason,
    spec_version,
    reviewed_by,
    reviewed_at
FROM detected_topic_edit_reuse_feedback
ORDER BY reviewed_at DESC, id DESC
LIMIT 30
""")

with session_scope() as session:
    rows = session.execute(query).mappings().all()

for row in rows:
    print("\n", dict(row))