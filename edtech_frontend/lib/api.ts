const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

async function apiError(
  response: Response,
  fallback: string
): Promise<Error> {
  let message = fallback

  try {
    const body = await response.json()

    if (body?.detail) {
      message =
        typeof body.detail === 'string'
          ? body.detail
          : JSON.stringify(body.detail)
    }
  } catch {
    // Keep fallback.
  }

  return new Error(message)
}

async function jsonRequest<T>(
  url: string,
  options?: RequestInit,
  fallback = 'Request failed.'
): Promise<T> {
  const response = await fetch(url, {
    cache: 'no-store',
    ...options,
  })

  if (!response.ok) {
    throw await apiError(response, fallback)
  }

  return response.json()
}


export type RunSnapshot = {
  run_id?: string
  transcript_name?: string
  state?: string
  human_gate?: string | null
  human_action_required?: boolean
  manifest_status?: string
  module1_complete?: boolean
  module2_complete?: boolean
  module3_complete?: boolean
  topic_review_count?: number
  pending_topic_review_count?: number
  approved_topic_count?: number
  allowed_tools?: string[]
  reason?: string
  [key: string]: unknown
}


export type CreateRunResponse = {
  success: boolean
  run_id: string
  filename: string
  run_dir: string
  final_state: string
  human_action_required: boolean
  interrupt_count: number
  snapshot: RunSnapshot
}


export type RunProgressResponse = {
  success: boolean
  run_id: string
  transcript_name: string
  percent: number
  stage: string
  message: string
  background_status:
    | 'queued'
    | 'running'
    | 'waiting_for_human'
    | 'complete'
    | 'failed'
    | string
  error: string | null
  module1_ready: boolean
  module2_ready: boolean
  module3_ready: boolean
  human_action_required: boolean
  human_gate: string | null
  workflow_state: string | null
  eta_seconds: number | null
  eta_label: string | null
  eta_basis: string | null
  eta_sample_count: number
}


export type PreprocessingResponse = {
  success: boolean
  run_id: string
  transcript_name: string
  cleaned_transcript: string
  cleaned_word_count: number
  deterministic_stats: Record<string, unknown>
  technical_stats: Record<string, unknown>
  unresolved_issue_count: number
}


export type SemanticChunk = {
  chunk_id: number
  text?: string
  word_count?: number
  sentence_count?: number
  boundary_reason?: string
  boundary_similarity?: number | null
  boundary_transition_strength?: number | null
  start_sentence?: number
  end_sentence?: number
  overlap_word_count?: number
  [key: string]: unknown
}


export type SemanticResponse = {
  success: boolean
  run_id: string
  transcript_name: string
  chunk_count: number
  chunks: SemanticChunk[]
}


export type SyllabusOption = {
  concept_id: string
  label: string
  official_reference: string
  chapter_reference?: string
  official_title?: string
  domain?: string
  paper?: string
}


export type TopicItem = {
  topic_index: number
  effective_index?: number
  concept_id?: string
  topic?: string
  detected_topic?: string
  role?: string
  topic_role?: string
  official_reference?: string
  official_title?: string
  chapter_reference?: string
  domain?: string
  paper?: string
  confidence?: number | null
  ranking_score?: number | null
  source_chunk_ids?: number[]
  evidence?: unknown[]
  human_edited?: boolean
  human_added_topic?: boolean
  [key: string]: unknown
}


export type TopicReviewItem = {
  id?: number
  original_topic?: string
  rough_topic?: string
  proposed_decision?: string
  proposed_mapped_concept_id?: string
  confidence?: number | null
  reason?: string
  evidence_text?: string
  source_chunk_ids?: number[]
  qdrant_candidates?: Array<{
    concept_id?: string
    label?: string
    official_reference?: string
    score?: number
    [key: string]: unknown
  }>
  status?: string
  review_status?: string
  [key: string]: unknown
}



export type HistoricalMemoryCandidate = {
  memory_id: number
  runtime_status: 'decision_required' | 'historical_applied' | string
  runtime_reason?: string
  edit_action: string
  source_concept_id?: string
  source_topic?: string
  source_role?: string
  target_concept_id?: string
  target_topic?: string
  target_role?: string
  reviewer_reason?: string
  stored_evidence?: string
  historical_outcome: string
  context_diagnostic?: string
  current_evidence?: string
  spec_version?: string
  saved_decision?: 'approve_reuse' | 'reject_reuse' | string | null
  saved_reason?: string | null
  fresh_topic?: TopicItem | null
}

export type HistoricalMemoryReviewItem = {
  review_key: string
  topic_label: string
  status: 'decision_required' | 'historical_applied' | string
  memory_ids: number[]
  memories: HistoricalMemoryCandidate[]
  fresh_topic?: TopicItem | null
  saved_decision?: 'approve_reuse' | 'reject_reuse' | string | null
}

export type HistoricalMemoryReviewInput = {
  decision: 'use_historical' | 'keep_fresh'
  memory_ids: number[]
  selected_memory_id?: number
  reason: string
}

export type TopicsResponse = {
  success: boolean
  run_id: string
  transcript_name: string
  topics: TopicItem[]
  topic_count: number
  pending_reviews: TopicReviewItem[]
  pending_review_count: number
  resolved_reviews: TopicReviewItem[]
  resolved_review_count: number
  orphaned_reviews: TopicReviewItem[]
  orphaned_review_count: number
  review_status_authority?: string
  review_db_available?: boolean
  review_reconciliation_error?: string | null
  llm_results: Record<string, unknown>[]
  runtime: Record<string, unknown>
  historical_memory_reviews: HistoricalMemoryReviewItem[]
  historical_memory_review_count: number
  historical_memory_pending_count: number
  historical_memory_error?: string | null
  spec_version?: string | null
  approved_topics: TopicItem[]
  approved_topic_file_count: number
  agent2_handoff_ready: boolean
  snapshot: RunSnapshot
  syllabus_options: SyllabusOption[]
  syllabus_error?: string | null
  human_write_result?: unknown
}


export type MappingReviewInput = {
  action: 'approve' | 'reject' | 'correct'
  corrected_decision?: 'mapped' | 'out_of_syllabus'
  corrected_mapped_concept_id?: string
  reason?: string
  review_notes?: string
}


export type TopicEditInput = {
  action:
    | 'change_role'
    | 'replace_topic'
    | 'remove_topic'
    | 'add_topic'
  reason: string
  topic_index?: number
  source_concept_id?: string
  target_concept_id?: string
  target_role?: 'primary' | 'supporting'
  source_chunk_ids?: number[]
}



export type DashboardRun = {
  run_id: string
  transcript_name: string
  created_at: string
  status: string
  stage_index: number
  module1_complete: boolean
  module2_complete: boolean
  module3_complete: boolean
  topics_identified: number
  topics_approved: number
  assessment_questions: number
  assessment_status: string
  assessment_mode: string | null
  human_action_required: boolean
  logged_to_runs: boolean
}

export type DashboardResponse = {
  success: boolean
  metrics: {
    transcripts_processed: number
    topics_identified: number
    topics_approved: number
    assessment_questions: number
  }
  latest_run: DashboardRun | null
  recent_runs: DashboardRun[]
  logging: {
    enabled: boolean
    runs_root: string
    run_count: number
  }
}


export async function getDashboard(): Promise<DashboardResponse> {
  return jsonRequest<DashboardResponse>(
    `${API_URL}/api/dashboard`,
    undefined,
    'Could not load dashboard data.'
  )
}

export async function createRun(
  file: File
): Promise<CreateRunResponse> {
  const formData = new FormData()
  formData.append('file', file)

  return jsonRequest<CreateRunResponse>(
    `${API_URL}/api/runs`,
    {
      method: 'POST',
      body: formData,
    },
    'Transcript processing could not be started.'
  )
}


export async function getRunProgress(
  runId: string
): Promise<RunProgressResponse> {
  return jsonRequest<RunProgressResponse>(
    `${API_URL}/api/runs/${encodeURIComponent(runId)}/progress`,
    undefined,
    'Could not read Agent 1 progress.'
  )
}


export async function getPreprocessing(
  runId: string
): Promise<PreprocessingResponse> {
  return jsonRequest<PreprocessingResponse>(
    `${API_URL}/api/runs/${encodeURIComponent(runId)}/preprocessing`,
    undefined,
    'Could not load the cleaned transcript.'
  )
}


export async function getSemantic(
  runId: string
): Promise<SemanticResponse> {
  return jsonRequest<SemanticResponse>(
    `${API_URL}/api/runs/${encodeURIComponent(runId)}/semantic`,
    undefined,
    'Could not load semantic chunks.'
  )
}


export async function getTopics(
  runId: string
): Promise<TopicsResponse> {
  return jsonRequest<TopicsResponse>(
    `${API_URL}/api/runs/${encodeURIComponent(runId)}/topics`,
    undefined,
    'Could not load Agent 1 topic mapping.'
  )
}


export async function submitMappingReview(
  runId: string,
  reviewId: number,
  input: MappingReviewInput
): Promise<TopicsResponse> {
  return jsonRequest<TopicsResponse>(
    `${API_URL}/api/runs/${encodeURIComponent(runId)}/mapping-reviews/${reviewId}`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(input),
    },
    'Could not save the mapping review.'
  )
}



export async function submitHistoricalMemoryReview(
  runId: string,
  input: HistoricalMemoryReviewInput
): Promise<TopicsResponse> {
  return jsonRequest<TopicsResponse>(
    `${API_URL}/api/runs/${encodeURIComponent(runId)}/memory-reviews`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(input),
    },
    'Could not save the historical HITL decision.'
  )
}

export async function submitTopicEdit(
  runId: string,
  input: TopicEditInput
): Promise<TopicsResponse> {
  return jsonRequest<TopicsResponse>(
    `${API_URL}/api/runs/${encodeURIComponent(runId)}/topic-edits`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(input),
    },
    'Could not save the topic edit.'
  )
}


export async function approveTopicsForAgent2(
  runId: string,
  topicIndexes: number[]
): Promise<TopicsResponse> {
  return jsonRequest<TopicsResponse>(
    `${API_URL}/api/runs/${encodeURIComponent(runId)}/approve-topics`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        topic_indexes: topicIndexes,
      }),
    },
    'Could not approve topics for Agent 2.'
  )
}


export type AssessmentMode =
  | 'retrieve_hybrid'
  | 'complete_quiz'

export type AssessmentModelOption = {
  key: string
  display_name: string
  provider?: string | null
  model_id?: string | null
  context_window_tokens?: number | null
  hard_max_output_tokens?: number | null
}

export type AssessmentNotebookOption = {
  key: 'plan_a' | 'plan_b' | 'plan_c' | string
  label: string
  strategy?: string
  available: boolean
  resolved_path?: string | null
}

export type AssessmentConfigResponse = {
  success: boolean
  run_id: string
  approved_topics: TopicItem[]
  topic_count: number
  primary_topic_count: number
  supporting_topic_count: number
  actual_chunk_evidence_available: boolean
  models: AssessmentModelOption[]
  notebook_options: AssessmentNotebookOption[]
}

export type AssessmentStartInput = {
  mode: AssessmentMode
  paper: 'Any' | 'Paper 1' | 'Paper 2'
  number_of_questions: number
  target_total_marks: number
  minimum_question_marks: number
  maximum_question_marks: number
  minimum_primary_questions: number
  minimum_supporting_questions: number
  cover_all_approved_topics: boolean
  include_code_questions: boolean
  include_visual_questions: boolean
  programming_language: 'Automatic' | 'Python'
  model_key: string
  quiz_plan: 'plan_a' | 'plan_b' | 'plan_c'
  special_instructions: string
}

export type UserRegenerationAttemptInfo = {
  user_regeneration_attempts_used: number
  user_regeneration_attempts_remaining: number
  max_user_regeneration_attempts: number
}

export type AssessmentQuestion = {
  question_id: string
  generated_question_id?: string
  plan_index?: number
  source: 'official' | 'ai_generated' | string
  topic: string
  official_reference: string
  role: string
  marks: number
  paper: string
  question_number: string
  question_text: string
  context: string
  marking_guidance: unknown[] | string[]
  visual_paths: string[]
  visual_type: string
  visual_spec: Record<string, unknown>
  semantic_score?: number | null
  user_regeneration_attempts_used?: number
  user_regeneration_attempts_remaining?: number
  max_user_regeneration_attempts?: number
  retrieval_feedback?: {
    decision: 'relevant' | 'not_relevant' | string
    reason?: string
    memory_status?: string
    memory_eligible?: boolean
    memory_error?: string
  } | null
}

export type OfficialAssessmentResult = {
  questions: AssessmentQuestion[]
  question_count: number
  selected_marks: number
  release_status: string
  pdf_paths: string[]
  package_path?: string
}

export type GeneratedAssessmentResult = {
  quiz_mode: 'complete_quiz' | 'fill_shortfall' | string
  assessment_type: string
  human_review_state: string
  generated_quality_accepted: boolean
  release_ready: boolean
  candidate_questions: AssessmentQuestion[]
  accepted_questions: AssessmentQuestion[]
  candidate_count: number
  candidate_marks: number
  accepted_count: number
  accepted_marks: number
  // Existing Notebook 06 / internal retry counters (kept for compatibility).
  regeneration_attempts_used: number
  max_regeneration_attempts: number
  internal_regeneration_attempts_used?: number
  internal_max_regeneration_attempts?: number

  // New USER-triggered regeneration budget: max 2 per generated question.
  max_user_regeneration_attempts_per_question: number
  user_regeneration_attempts_by_plan_index: Record<string, UserRegenerationAttemptInfo>
  whole_quiz_user_regeneration_available: boolean
  whole_quiz_user_regeneration_attempts_remaining: number
  whole_quiz_user_regeneration_blocked_plan_indexes: number[]
  user_regeneration_disclaimer: string

  pdf_paths: string[]
  validation: Record<string, unknown>
}

export type AssessmentShortfall = {
  sufficient: boolean
  selected_questions: number
  requested_questions: number
  selected_marks: number
  target_marks: number
  missing_questions: number
  missing_marks: number
}

export type AssessmentStatusResponse = {
  success: boolean
  run_id: string
  status: 'idle' | 'queued' | 'running' | 'complete' | 'failed' | string
  stage: string
  progress: number
  progress_source?: string | null
  progress_detail?: string | null
  message: string
  mode?: AssessmentMode | null
  request?: Partial<AssessmentStartInput> | null
  shortfall?: AssessmentShortfall | null
  error?: string | null
  started_at_utc?: string
  stage_started_at_utc?: string
  completed_at_utc?: string
  updated_at_utc?: string
  eta_seconds?: number | null
  eta_label?: string | null
  eta_total_label?: string | null
  eta_basis?: string | null
  eta_sample_count?: number
  eta_low_seconds?: number | null
  eta_high_seconds?: number | null
  eta_total_low_seconds?: number | null
  eta_total_high_seconds?: number | null
  eta_process?: string | null
  eta_source?: string | null
  official?: OfficialAssessmentResult | null
  generated?: GeneratedAssessmentResult | null
  human_write_result?: unknown
  question_review_result?: {
    question_id: string
    plan_index: number
    action: string
    quiz_mode: string
    question_changed?: boolean | null
    user_regeneration_attempt_before?: UserRegenerationAttemptInfo | null
    user_regeneration_attempt_after?: UserRegenerationAttemptInfo | null
    regeneration_commit_gate?: unknown
  }
  quiz_review_result?: {
    quiz_mode: string
    decision: string
    affected_plan_indexes: number[]
    changed_plan_indexes: number[]
    user_regeneration_attempts_before: Record<string, UserRegenerationAttemptInfo> | Record<number, UserRegenerationAttemptInfo>
    user_regeneration_attempts_after: Record<string, UserRegenerationAttemptInfo> | Record<number, UserRegenerationAttemptInfo>
  }
}

export type Agent2EtaProcess =
  | 'official_retrieval'
  | 'complete_quiz_generation'
  | 'shortfall_generation'
  | 'question_regeneration'

export type Agent2EtaResponse = {
  success: boolean
  run_id: string
  process: Agent2EtaProcess
  eta_seconds?: number | null
  eta_label?: string | null
  eta_total_label?: string | null
  eta_basis?: string | null
  eta_sample_count?: number
  eta_low_seconds?: number | null
  eta_high_seconds?: number | null
  eta_total_low_seconds?: number | null
  eta_total_high_seconds?: number | null
  eta_process?: string | null
  eta_source?: string | null
}

export type RetrievalFeedbackInput = {
  decision: 'relevant' | 'not_relevant'
  reason: string
}

export type QuestionMarkingGuidanceItem = {
  marks: number
  criterion: string
}

export type GeneratedQuestionReviewInput = {
  quiz_mode: 'complete_quiz' | 'fill_shortfall'
  question_id: string
  plan_index: number
  action:
    | 'approve'
    | 'edit_question'
    | 'edit_marking_guidance'
    | 'regenerate'
    | 'reject'
  reason?: string
  question_text?: string
  marking_guidance?: QuestionMarkingGuidanceItem[]
}

export type QuizReviewInput = {
  quiz_mode: 'complete_quiz' | 'fill_shortfall'
  decision: 'approve' | 'regenerate' | 'reject'
  reason: string
}

export async function getAssessmentConfig(
  runId: string
): Promise<AssessmentConfigResponse> {
  return jsonRequest<AssessmentConfigResponse>(
    `${API_URL}/api/runs/${encodeURIComponent(runId)}/assessment/config`,
    undefined,
    'Could not load Agent 2 assessment configuration.'
  )
}

export async function startAssessment(
  runId: string,
  input: AssessmentStartInput
): Promise<AssessmentStatusResponse> {
  return jsonRequest<AssessmentStatusResponse>(
    `${API_URL}/api/runs/${encodeURIComponent(runId)}/assessment/start`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(input),
    },
    'Could not start Agent 2 assessment.'
  )
}

export async function getAssessmentStatus(
  runId: string
): Promise<AssessmentStatusResponse> {
  return jsonRequest<AssessmentStatusResponse>(
    `${API_URL}/api/runs/${encodeURIComponent(runId)}/assessment/status`,
    undefined,
    'Could not read Agent 2 assessment status.'
  )
}

export async function getAgent2Eta(
  runId: string,
  process: Agent2EtaProcess
): Promise<Agent2EtaResponse> {
  return jsonRequest<Agent2EtaResponse>(
    `${API_URL}/api/runs/${encodeURIComponent(runId)}/assessment/eta?process=${encodeURIComponent(process)}`,
    undefined,
    'Could not estimate Agent 2 processing time.'
  )
}

export async function submitRetrievalFeedback(
  runId: string,
  questionId: string,
  input: RetrievalFeedbackInput
): Promise<AssessmentStatusResponse> {
  return jsonRequest<AssessmentStatusResponse>(
    `${API_URL}/api/runs/${encodeURIComponent(runId)}/retrieval-feedback/${encodeURIComponent(questionId)}`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(input),
    },
    'Could not save retrieval feedback.'
  )
}


export async function submitGeneratedQuestionReview(
  runId: string,
  input: GeneratedQuestionReviewInput
): Promise<AssessmentStatusResponse> {
  return jsonRequest<AssessmentStatusResponse>(
    `${API_URL}/api/runs/${encodeURIComponent(runId)}/question-review`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(input),
    },
    'Could not save the question-level human review.'
  )
}


export async function submitQuizReview(
  runId: string,
  input: QuizReviewInput
): Promise<AssessmentStatusResponse> {
  return jsonRequest<AssessmentStatusResponse>(
    `${API_URL}/api/runs/${encodeURIComponent(runId)}/quiz-review`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(input),
    },
    'Could not save the Agent 2 human review.'
  )
}

export function assessmentAssetUrl(
  runId: string,
  path: string
): string {
  return `${API_URL}/api/runs/${encodeURIComponent(runId)}/asset?path=${encodeURIComponent(path)}`
}