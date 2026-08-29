from app.services.detected_topic_edit_reuse_feedback_store import (
    DetectedTopicEditReuseFeedbackStore,
)

store = DetectedTopicEditReuseFeedbackStore()

for memory_id in [13, 15, 16, 17, 18, 19]:
    print("\nMEMORY", memory_id)
    print(store.memory_snapshot(memory_id))