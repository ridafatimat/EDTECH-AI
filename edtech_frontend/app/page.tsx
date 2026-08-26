'use client'

import { useEffect, useMemo, useState } from 'react'
import {
  ArrowRight,
  BookOpen,
  Check,
  CheckCircle2,
  Clock3,
  ChevronLeft,
  ChevronRight,
  ClipboardCheck,
  FileAudio,
  FileText,
  Home as HomeIcon,
  Info,
  History,
  Layers3,
  Menu,
  Network,
  Plus,
  Search,
  Sparkles,
  Upload,
  X,
  XCircle,
  Zap,
} from 'lucide-react'

import {
  approveTopicsForAgent2,
  assessmentAssetUrl,
  createRun,
  getDashboard,
  getAssessmentConfig,
  getAssessmentStatus,
  getAgent2Eta,
  getPreprocessing,
  getRunProgress,
  getSemantic,
  getTopics,
  startAssessment,
  submitMappingReview,
  submitHistoricalMemoryReview,
  submitGeneratedQuestionReview,
  submitQuizReview,
  submitRetrievalFeedback,
  submitTopicEdit,
  type Agent2EtaResponse,
  type AssessmentConfigResponse,
  type AssessmentMode,
  type AssessmentQuestion,
  type AssessmentStartInput,
  type AssessmentStatusResponse,
  type CreateRunResponse,
  type DashboardResponse,
  type DashboardRun,
  type PreprocessingResponse,
  type RunProgressResponse,
  type SemanticResponse,
  type HistoricalMemoryReviewItem,
  type SyllabusOption,
  type TopicItem,
  type TopicReviewItem,
  type TopicsResponse,
} from '@/lib/api'

const nav = [
  { label: 'Home', icon: HomeIcon, page: 'home' },
  { label: 'Dashboard', icon: Layers3, page: 'dashboard' },
  { label: 'Transcript', icon: FileText, page: 'transcript' },
  { label: 'Preprocessing', icon: Zap, page: 'preprocessing' },
  { label: 'Semantic Analysis', icon: Network, page: 'semantic' },
  { label: 'Topic Mapping', icon: BookOpen, page: 'topics' },
  { label: 'Assessment', icon: ClipboardCheck, page: 'assessment' },
]


function Badge({ children, tone = 'neutral' }: { children: React.ReactNode; tone?: 'neutral' | 'success' | 'warning' | 'info' | 'teal' }) {
  return <span className={`badge badge-${tone}`}>{children}</span>
}

function Stepper({ current }: { current: number }) {
  const steps = ['Transcript', 'Preprocessing', 'Semantic Analysis', 'Topic Mapping', 'Assessment']
  return <div className="stepper">{steps.map((step, index) => <div className="step-wrap" key={step}><div className={`step ${index < current ? 'done' : ''} ${index === current ? 'current' : ''}`}><span>{index < current ? <Check size={14} /> : index + 1}</span>{step}</div>{index < steps.length - 1 && <ChevronRight size={15} className="step-arrow" />}</div>)}</div>
}

function Sidebar({ page, setPage, open, setOpen }: { page: string; setPage: (page: string) => void; open: boolean; setOpen: (v: boolean) => void }) {
  return <>
    <button className="mobile-menu" onClick={() => setOpen(!open)} aria-label="Toggle navigation"><Menu /></button>
    <aside className={`sidebar ${open ? 'sidebar-expanded' : 'sidebar-collapsed'} ${open ? 'sidebar-open' : ''}`}>
      <div className="brand"><div className="brand-mark"><Sparkles size={18} /></div><div className="brand-copy"><strong>EDTech</strong><small>Lesson intelligence</small></div><button className="sidebar-toggle" onClick={() => setOpen(!open)} aria-label={open ? 'Collapse sidebar' : 'Expand sidebar'} title={open ? 'Collapse sidebar' : 'Expand sidebar'}>{open ? <ChevronLeft /> : <ChevronRight />}</button></div>
      <div className="nav-group"><p>WORKSPACE</p>{nav.slice(0, 2).map(item => <NavItem key={item.page} item={item} page={page} setPage={setPage} setOpen={setOpen} />)}</div>
      <div className="nav-group"><p>AGENT 1 · LESSON</p>{nav.slice(2, 6).map(item => <NavItem key={item.page} item={item} page={page} setPage={setPage} setOpen={setOpen} />)}</div>
      <div className="nav-group"><p>AGENT 2 · ASSESSMENT</p><NavItem item={nav[6]} page={page} setPage={setPage} setOpen={setOpen} /></div>
      <div className="sidebar-footer"><div className="scope-dot" /><div><strong>AQA GCSE</strong><small>Computer Science</small></div></div>
    </aside>
  </>
}
function NavItem({ item, page, setPage, setOpen }: any) { const Icon = item.icon; return <button className={`nav-item ${page === item.page ? 'active' : ''}`} onClick={() => { setPage(item.page); setOpen(false) }} title={item.label} aria-label={item.label}><Icon size={17} /><span>{item.label}</span></button> }

function Header({ eyebrow, title, description }: { eyebrow: string; title: string; description?: string }) { return <header className="page-header"><div><p className="eyebrow">{eyebrow}</p><h1>{title}</h1>{description && <p className="lead">{description}</p>}</div><div className="header-context"><span className="status-pip" />AQA GCSE Computer Science</div></header> }
function Button({ children, onClick, variant = 'primary', disabled = false }: any) { return <button disabled={disabled} onClick={onClick} className={`button button-${variant}`}>{children}</button> }
function Card({ children, className = '' }: { children: React.ReactNode; className?: string }) { return <section className={`card ${className}`}>{children}</section> }

function Home({ go }: { go: (p: string) => void }) { return <div className="landing-page"><header className="landing-nav"><div className="landing-brand"><span className="brand-mark"><Sparkles size={18} /></span><strong>EDTech</strong></div><div className="landing-nav-right"><Button variant="ghost" onClick={() => go('dashboard')}>Dashboard <ArrowRight size={16} /></Button></div></header><section className="landing-hero"><div className="hero-copy-column"><p className="landing-eyebrow">AI-ASSISTED LESSON INTELLIGENCE</p><h1 className="landing-title">From lesson transcript to <em>assessment-ready</em> learning.</h1><p className="landing-description">Analyse what was actually taught, map lesson evidence to the AQA GCSE Computer Science syllabus, and turn approved concepts into relevant assessment material.</p><div className="landing-actions"><Button onClick={() => go('transcript')}>Analyse a Transcript <ArrowRight size={17} /></Button><button className="text-link" onClick={() => go('dashboard')}>View Workflow <ArrowRight size={15} /></button></div><div className="trust-points"><span><Check size={14} /> Evidence-based</span><span><Check size={14} /> Human reviewed</span><span><Check size={14} /> AQA aligned</span></div></div><div className="preview-wrap"><div className="product-preview"><div className="preview-top"><div><p className="preview-label">EDTECH / LESSON ANALYSIS</p><h2>Lesson Analysis</h2></div><Badge tone="info">Review ready</Badge></div><div className="preview-workflow"><span className="active">Transcript</span><ChevronRight /><span className="active">Semantic</span><ChevronRight /><span>Topic Mapping</span><ChevronRight /><span>Assessment</span></div><div className="preview-topic"><div className="preview-topic-head"><div><p className="preview-label">PRIMARY TOPIC</p><h3>Bubble Sort</h3></div><Badge tone="success">AQA Mapped</Badge></div><p className="preview-evidence">“The teacher demonstrates how adjacent values are compared during each pass…”</p><div className="preview-meta"><span>Confidence <strong>94%</strong></span><span>3.1.3 Searching and Sorting</span></div></div><div className="preview-row"><div><p className="preview-label">SUPPORTING TOPIC</p><strong>Binary Search</strong></div><Badge tone="info">AQA Mapped</Badge></div></div><div className="review-card"><div className="review-card-title"><Info size={15} /><strong>System Review</strong></div><div className="review-grid"><span>Initially detected</span><strong>Selection</strong><span>Decision</span><b>Removed</b><span>Why</span><p>An if-statement appeared in the lesson, but Selection itself was not explicitly taught.</p></div></div><div className="ready-card"><CheckCircle2 size={15} /><span>Topic Review Ready</span></div></div></section><section className="landing-lower"><div className="pipeline-strip">{['Transcript', 'Semantic Analysis', 'AQA Mapping', 'Human Review', 'Assessment'].map((step, i) => <div className="landing-step" key={step}><span>{String(i + 1).padStart(2, '0')}</span><strong>{step}</strong>{i < 4 && <ChevronRight />}</div>)}</div><div className="landing-agents"><Card><div className="landing-agent-head"><span className="agent-number">01</span><p className="eyebrow">AGENT 1</p></div><h2>Transcript Intelligence</h2><p>Understand what was actually taught.</p><div className="agent-list"><span>Transcript preprocessing</span><span>Semantic lesson analysis</span><span>Evidence-backed AQA mapping</span><span>Human topic review</span></div></Card><Card><div className="landing-agent-head"><span className="agent-number">02</span><p className="eyebrow">AGENT 2</p></div><h2>Assessment Intelligence</h2><p>Turn approved lesson content into assessment material.</p><div className="agent-list"><span>Official question retrieval</span><span>Mark scheme retrieval</span><span>Quiz generation</span><span>Diagram-supported review</span></div></Card></div></section></div> }

function Dashboard({
  go,
  activeRun,
}: {
  go: (p: string) => void
  activeRun: CreateRunResponse | null
}) {
  const [data, setData] =
    useState<DashboardResponse | null>(null)

  const [loading, setLoading] = useState(true)
  const [error, setError] =
    useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    let timer: number | null = null

    const load = async () => {
      try {
        const value = await getDashboard()

        if (cancelled) return

        setData(value)
        setError(null)
      } catch (err) {
        if (cancelled) return

        setError(
          err instanceof Error
            ? err.message
            : 'Could not load dashboard.'
        )
      } finally {
        if (!cancelled) {
          setLoading(false)

          // Keep the KPI cards fresh while runs change.
          timer = window.setTimeout(load, 5000)
        }
      }
    }

    load()

    return () => {
      cancelled = true

      if (timer !== null) {
        window.clearTimeout(timer)
      }
    }
  }, [])

  const latest = data?.latest_run ?? null

  const metrics = [
    {
      label: 'Transcripts Processed',
      value: data?.metrics.transcripts_processed ?? 0,
      sub: 'Persisted lesson runs',
    },
    {
      label: 'Topics Identified',
      value: data?.metrics.topics_identified ?? 0,
      sub: 'Effective Agent 1 topics',
    },
    {
      label: 'Topics Approved',
      value: data?.metrics.topics_approved ?? 0,
      sub: 'Approved for Agent 2',
    },
    {
      label: 'Assessment Questions',
      value: data?.metrics.assessment_questions ?? 0,
      sub: 'Retrieved / generated',
    },
  ]

  return (
    <>
      <Header
        eyebrow="Workspace overview"
        title="Dashboard"
        description="Live values from persisted EDTech lesson runs and assessment activity."
      />

      <div className="dashboard-actions">
        <div
          style={{
            display: 'grid',
            gap: '0.2rem',
          }}
        >
          <span className="muted">
            {activeRun?.run_id
              ? `Current run: ${
                  activeRun.snapshot?.transcript_name
                  || activeRun.filename
                }`
              : latest
                ? `Latest lesson: ${latest.transcript_name}`
                : 'No lesson run yet'}
          </span>

          {data?.logging.enabled && (
            <small className="muted">
              Run logging active · {data.logging.run_count} saved run
              {data.logging.run_count === 1 ? '' : 's'}
            </small>
          )}
        </div>

        <Button onClick={() => go('transcript')}>
          Process New Transcript
          <ArrowRight size={17} />
        </Button>
      </div>

      {error && (
        <div className="guidance">
          <XCircle size={19} />
          <div>
            <strong>Dashboard could not refresh</strong>
            <p>{error}</p>
          </div>
        </div>
      )}

      <div className="metric-grid">
        {metrics.map((metric) => (
          <Card
            key={metric.label}
            className="metric"
          >
            <p>{metric.label}</p>
            <strong>
              {loading && !data
                ? '…'
                : metric.value.toLocaleString()}
            </strong>
            <small>{metric.sub}</small>
          </Card>
        ))}
      </div>
    </>
  )
}


function Transcript({
  go,
  onRunCreated,
  onProcessingStarted,
}: {
  go: (p: string) => void
  onRunCreated: (run: CreateRunResponse) => void
  onProcessingStarted: () => void
}) {
  const [file, setFile] = useState<File | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [previewText, setPreviewText] = useState('')
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [previewError, setPreviewError] = useState<string | null>(null)

  const chooseFile = () => {
    document.getElementById('transcript-file-input')?.click()
  }

  useEffect(() => {
    let cancelled = false
    let objectUrl: string | null = null

    const buildPreview = async () => {
      setPreviewText('')
      setPreviewError(null)
      setPreviewUrl(null)

      if (!file) return

      setPreviewLoading(true)

      const extension =
        file.name.split('.').pop()?.toLowerCase() || ''

      try {
        if (extension === 'pdf') {
          objectUrl = URL.createObjectURL(file)

          if (!cancelled) {
            setPreviewUrl(objectUrl)
          }

          return
        }

        if (extension === 'txt') {
          const rawText = await file.text()

          if (!cancelled) {
            setPreviewText(rawText)
          }

          return
        }

        if (extension === 'docx') {
          const mammoth = await import('mammoth')
          const arrayBuffer = await file.arrayBuffer()

          const result = await mammoth.extractRawText({
            arrayBuffer,
          })

          if (!cancelled) {
            setPreviewText(result.value || '')
          }

          return
        }

        if (!cancelled) {
          setPreviewError(
            'Preview is not available for this file type.'
          )
        }
      } catch (err) {
        if (!cancelled) {
          console.error('Transcript preview failed:', err)

          setPreviewError(
            'The transcript was selected successfully, but its preview could not be generated.'
          )
        }
      } finally {
        if (!cancelled) {
          setPreviewLoading(false)
        }
      }
    }

    buildPreview()

    return () => {
      cancelled = true

      if (objectUrl) {
        URL.revokeObjectURL(objectUrl)
      }
    }
  }, [file])

  const clearFile = () => {
    setFile(null)
    setError(null)
    setPreviewText('')
    setPreviewUrl(null)
    setPreviewError(null)

    const input = document.getElementById(
      'transcript-file-input'
    ) as HTMLInputElement | null

    if (input) input.value = ''
  }

  const processTranscript = async () => {
    if (!file || isProcessing) return

    setError(null)
    setIsProcessing(true)
    onProcessingStarted()
    go('preprocessing')

    try {
      const result = await createRun(file)
      onRunCreated(result)
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : 'Transcript processing could not be started.'

      console.error('Transcript processing failed:', err)
      setError(message)
      go('transcript')
      window.alert(message)
    } finally {
      setIsProcessing(false)
    }
  }

  const extension =
    file?.name.split('.').pop()?.toUpperCase() || ''

  const previewLimit = 12000
  const displayedPreview =
    previewText.length > previewLimit
      ? `${previewText.slice(0, previewLimit)}\n\n… Preview shortened for display.`
      : previewText

  return (
    <>
      <Header
        eyebrow="Agent 1 / Lesson"
        title="Upload Lesson Transcript"
        description="Upload the transcript from the lesson you want EDTech to analyse."
      />

      <Stepper current={0} />

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: file
            ? 'minmax(0, 0.9fr) minmax(0, 1.1fr)'
            : 'minmax(0, 1fr)',
          gap: '1.15rem',
          alignItems: 'stretch',
        }}
      >
        <Card className="upload-card">
          <input
            id="transcript-file-input"
            type="file"
            accept=".pdf,.docx,.txt"
            hidden
            onChange={(event) => {
              const selected =
                event.target.files?.[0] ?? null

              setFile(selected)
              setError(null)
            }}
          />

          <div
            className="dropzone"
            onClick={chooseFile}
            role="button"
            tabIndex={0}
            onKeyDown={(event) => {
              if (
                event.key === 'Enter'
                || event.key === ' '
              ) {
                event.preventDefault()
                chooseFile()
              }
            }}
            style={{
              minHeight: file ? '280px' : undefined,
            }}
          >
            <div className="upload-icon">
              <Upload />
            </div>

            <h2>
              {file
                ? file.name
                : 'Drop your transcript here'}
            </h2>

            <p>
              {file
                ? `${extension} · ${(
                    file.size / 1024
                  ).toFixed(1)} KB · Ready to process`
                : 'or click to browse from your device'}
            </p>

            {file && (
              <Badge tone="teal">
                File selected
              </Badge>
            )}
          </div>

          {file && (
            <div className="file-row">
              <FileAudio size={19} />

              <div>
                <strong>{file.name}</strong>
                <small>
                  {extension} ·{' '}
                  {(file.size / 1024).toFixed(1)} KB
                </small>
              </div>

              <button
                type="button"
                onClick={(event) => {
                  event.stopPropagation()
                  clearFile()
                }}
                aria-label="Remove file"
              >
                <X size={18} />
              </button>
            </div>
          )}

          {error && (
            <div className="guidance">
              <XCircle size={19} />
              <div>
                <strong>
                  Transcript processing failed
                </strong>
                <p>{error}</p>
              </div>
            </div>
          )}

          <div className="upload-footer">
            <div>
              <strong>Supported formats</strong>
              <span>
                PDF, DOCX and TXT transcript files
              </span>
            </div>

            <Button
              onClick={processTranscript}
              disabled={!file || isProcessing}
            >
              {isProcessing
                ? 'Starting Agent 1...'
                : 'Process Transcript'}

              {!isProcessing && (
                <ArrowRight size={17} />
              )}
            </Button>
          </div>
        </Card>

        {file && (
          <Card>
            <div
              className="card-heading"
              style={{
                marginBottom: '0.85rem',
              }}
            >
              <div>
                <p className="eyebrow">
                  TRANSCRIPT PREVIEW
                </p>

                <h2
                  style={{
                    marginBottom: '0.2rem',
                  }}
                >
                  {file.name}
                </h2>

                <span className="muted">
                  Preview before processing
                </span>
              </div>

              <Badge tone="info">
                {extension}
              </Badge>
            </div>

            <div
              style={{
                minHeight: '470px',
                height: '470px',
                borderRadius: '16px',
                overflow: 'hidden',
                background: 'rgba(43, 35, 59, 0.42)',
                border:
                  '1px solid rgba(255,255,255,0.15)',
                boxShadow:
                  'inset 0 1px 0 rgba(255,255,255,0.04)',
              }}
            >
              {previewLoading ? (
                <div
                  style={{
                    height: '100%',
                    display: 'grid',
                    placeItems: 'center',
                    textAlign: 'center',
                    padding: '2rem',
                  }}
                >
                  <div>
                    <div
                      className="spinner"
                      style={{
                        margin: '0 auto 1rem',
                      }}
                    />
                    <strong>
                      Preparing preview...
                    </strong>
                  </div>
                </div>
              ) : previewError ? (
                <div
                  style={{
                    height: '100%',
                    display: 'grid',
                    placeItems: 'center',
                    padding: '2rem',
                    textAlign: 'center',
                  }}
                >
                  <div>
                    <FileText
                      size={34}
                      style={{
                        marginBottom: '0.8rem',
                        opacity: 0.75,
                      }}
                    />
                    <strong
                      style={{
                        display: 'block',
                        marginBottom: '0.4rem',
                      }}
                    >
                      Preview unavailable
                    </strong>
                    <span className="muted">
                      {previewError}
                    </span>
                  </div>
                </div>
              ) : previewUrl ? (
                <iframe
                  src={previewUrl}
                  title={`${file.name} preview`}
                  style={{
                    width: '100%',
                    height: '100%',
                    border: 0,
                    background: '#ffffff',
                  }}
                />
              ) : (
                <div
                  style={{
                    height: '100%',
                    overflowY: 'auto',
                    padding: '1.15rem 1.25rem',
                  }}
                >
                  <pre
                    style={{
                      margin: 0,
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word',
                      fontFamily: 'inherit',
                      fontSize: '0.9rem',
                      lineHeight: 1.7,
                      color:
                        'rgba(255,255,255,0.92)',
                    }}
                  >
                    {displayedPreview
                      || 'No readable transcript text was found in this file.'}
                  </pre>
                </div>
              )}
            </div>

            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                gap: '1rem',
                marginTop: '0.8rem',
                fontSize: '0.78rem',
              }}
            >
              <span className="muted">
                {extension} ·{' '}
                {(file.size / 1024).toFixed(1)} KB
              </span>

              <span className="muted">
                Original upload preview
              </span>
            </div>
          </Card>
        )}
      </div>
    </>
  )
}

function Processing({
  kind = 'preprocessing',
  progress,
}: {
  kind?: 'preprocessing' | 'semantic' | 'topics'
  progress?: RunProgressResponse | null
}) {
  const isSemantic = kind === 'semantic'
  const isTopics = kind === 'topics'

  const currentStep = isTopics ? 3 : isSemantic ? 2 : 1
  const percent = Math.max(
    0,
    Math.min(100, progress?.percent ?? 0)
  )

  const title = isTopics
    ? 'Topic Mapping'
    : isSemantic
      ? 'Semantic Analysis'
      : 'Preprocessing'

  const statusTitle = isTopics
    ? 'Mapping lesson concepts...'
    : isSemantic
      ? 'Creating semantic chunks...'
      : 'Cleaning transcript...'

  const description = isTopics
    ? 'Matching evidence-backed lesson concepts to the official AQA syllabus.'
    : isSemantic
      ? 'Finding meaningful boundaries in the cleaned lesson text.'
      : 'Preparing the transcript for lesson-level analysis.'

  const detail =
    progress?.background_status === 'queued'
      ? 'Agent 1 run has been created and is starting the preprocessing pipeline...'
      : progress?.message || (
          isTopics
            ? 'EDTech is identifying official syllabus concepts and checking the human-review gate.'
            : isSemantic
              ? 'EDTech is analysing lesson structure and detecting semantic boundaries.'
              : 'EDTech is removing transcript noise and preparing cleaned lesson text.'
        )

  const workflowState = String(
    progress?.workflow_state || ''
  ).trim().toUpperCase()

  const toolFailed =
    progress?.background_status === 'failed'
    || workflowState === 'TOOL_FAILED'

  const failureTitle = isTopics
    ? 'Topic Mapping Failed'
    : isSemantic
      ? 'Semantic Analysis Failed'
      : 'Transcript Preprocessing Failed'

  if (toolFailed) {
    return (
      <>
        <Header
          eyebrow={`Agent 1 / ${title}`}
          title={title}
          description={description}
        />

        <Stepper current={currentStep} />

        <div
          className="guidance"
          style={{
            width: 'fit-content',
            maxWidth: 'min(92%, 760px)',
            margin: '1.5rem auto 0',
            padding: '1.05rem 1.3rem',
            alignItems: 'center',
          }}
        >
          <XCircle size={20} />

          <div>
            <strong>{failureTitle}</strong>

            <p>
              {progress?.error
                || 'A backend tool failed before this stage could complete. Check the required backend services and try again.'}
            </p>

            {workflowState && (
              <small
                style={{
                  display: 'block',
                  marginTop: '0.35rem',
                  opacity: 0.8,
                }}
              >
                Workflow state: {workflowState}
              </small>
            )}
          </div>
        </div>
      </>
    )
  }

  return (
    <>
      <Header
        eyebrow={`Agent 1 / ${title}`}
        title={title}
        description={description}
      />

      <Stepper current={currentStep} />

      <Card className="processing-card">
        <div className="spinner" />

        <div
          style={{
            fontSize: '2.2rem',
            fontWeight: 800,
            lineHeight: 1,
            marginTop: '0.75rem',
          }}
        >
          {percent}%
        </div>

        <div
          aria-label={`Processing progress ${percent}%`}
          style={{
            width: 'min(520px, 92%)',
            height: '10px',
            borderRadius: '999px',
            background: 'rgba(255,255,255,0.18)',
            overflow: 'hidden',
            margin: '1rem auto 1.2rem',
          }}
        >
          <div
            style={{
              width: `${percent}%`,
              height: '100%',
              borderRadius: '999px',
              background: 'currentColor',
              transition: 'width 300ms ease',
            }}
          />
        </div>

        <p className="eyebrow">SYSTEM STATUS</p>

        <h2>{statusTitle}</h2>

        <p className="lead">{detail}</p>

        {progress?.eta_label && percent < 100 && (
          <div
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.75rem',
              margin: '0.15rem auto 0.95rem',
              padding: '0.7rem 0.9rem',
              borderRadius: '14px',
              background: 'rgba(255,255,255,0.09)',
              border: '1px solid rgba(255,255,255,0.14)',
              boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.06)',
              textAlign: 'left',
            }}
          >
            <div
              style={{
                width: '34px',
                height: '34px',
                flex: '0 0 auto',
                display: 'grid',
                placeItems: 'center',
                borderRadius: '10px',
                background: 'rgba(255,255,255,0.10)',
                border: '1px solid rgba(255,255,255,0.12)',
              }}
            >
              <Clock3 size={17} />
            </div>

            <div
              style={{
                display: 'grid',
                gap: '0.12rem',
              }}
            >
              <strong
                style={{
                  fontSize: '0.95rem',
                  lineHeight: 1.25,
                }}
              >
                {progress.eta_label}
              </strong>

              {progress.eta_basis && (
                <span
                  style={{
                    fontSize: '0.76rem',
                    opacity: 0.72,
                  }}
                >
                  {progress.eta_basis}
                </span>
              )}
            </div>
          </div>
        )}

        {progress?.error && (
          <div className="guidance" style={{ marginTop: '1rem' }}>
            <XCircle size={19} />
            <div>
              <strong>Agent 1 stopped with an error</strong>
              <p>{progress.error}</p>
            </div>
          </div>
        )}

        <div className="process-notes">
          <span>
            <CheckCircle2 size={16} />
            Real backend stage progress
          </span>

          <span>
            <Info size={16} />
            Results appear automatically when ready
          </span>
        </div>
      </Card>
    </>
  )
}

function Preprocessed({
  go,
  data,
}: {
  go: (p: string) => void
  data: PreprocessingResponse
}) {
  return (
    <>
      <Header
        eyebrow="Agent 1 / Preprocessing"
        title="Cleaned Transcript"
        description="Review the prepared lesson text before semantic analysis."
      />

      <Stepper current={1} />

      <div className="success-banner">
        <CheckCircle2 size={20} />
        <div>
          <strong>
            Transcript preprocessing completed successfully
          </strong>
          <span>
            Real Module 1 output loaded for {data.transcript_name}.
          </span>
        </div>
      </div>

      <Card>
        <div className="card-heading">
          <div>
            <p className="eyebrow">Prepared content</p>
            <h2>Cleaned Transcript</h2>
          </div>
          <Badge tone="success">Completed</Badge>
        </div>

        <div className="transcript-box">
          {data.cleaned_transcript}
        </div>

        <div className="card-footer">
          <span className="muted">
            {data.cleaned_word_count.toLocaleString()} cleaned words ·
            Real backend output
          </span>

          <Button onClick={() => go('semantic')}>
            Continue to Semantic Analysis
            <ArrowRight size={17} />
          </Button>
        </div>
      </Card>
    </>
  )
}

function formatBoundary(value: unknown) {
  const text = String(value || 'semantic boundary')
    .replaceAll('_', ' ')
    .trim()

  return text
    ? text.charAt(0).toUpperCase() + text.slice(1)
    : 'Semantic boundary'
}

function SemanticDone({
  go,
  data,
}: {
  go: (p: string) => void
  data: SemanticResponse
}) {
  return (
    <>
      <Header
        eyebrow="Agent 1 / Semantic analysis"
        title="Semantic Lesson Chunks"
        description="Meaningful sections identified from the real cleaned transcript."
      />

      <Stepper current={2} />

      <div className="success-banner">
        <CheckCircle2 size={20} />
        <div>
          <strong>Semantic chunking completed successfully</strong>
          <span>
            {data.chunk_count} real semantic chunk
            {data.chunk_count === 1 ? '' : 's'} loaded from Module 2.
          </span>
        </div>
      </div>

      <div className="chunk-list">
        {data.chunks.map((chunk, index) => (
          <Card key={`${chunk.chunk_id}-${index}`}>
            <div className="chunk-top">
              <span className="chunk-number">
                {String(index + 1).padStart(2, '0')}
              </span>

              <strong>Chunk {chunk.chunk_id}</strong>

              <Badge tone="teal">
                {formatBoundary(chunk.boundary_reason)}
              </Badge>
            </div>

            <p>{String(chunk.text || '')}</p>

            <div className="chunk-footer">
              {typeof chunk.word_count === 'number'
                ? `${chunk.word_count} words · `
                : ''}
              Boundary: {formatBoundary(chunk.boundary_reason)}
              {typeof chunk.boundary_similarity === 'number'
                ? ` · Similarity ${chunk.boundary_similarity.toFixed(3)}`
                : ''}
            </div>
          </Card>
        ))}
      </div>

      <Button onClick={() => go('topics')}>
        Continue to Topic Mapping
        <ArrowRight size={17} />
      </Button>
    </>
  )
}


function topicName(topic: TopicItem) {
  return String(
    topic.topic
    || topic.detected_topic
    || topic.concept_id
    || 'Unnamed topic'
  )
}

function confidenceLabel(value: unknown) {
  const numeric = Number(value)

  if (!Number.isFinite(numeric)) return '—'

  const asPercent =
    numeric <= 1
      ? numeric * 100
      : numeric

  return `${Math.round(asPercent)}%`
}

function optionLabel(option: SyllabusOption) {
  return `${option.label} (${option.official_reference})`
}

function MappingReviewCard({
  runId,
  item,
  syllabusOptions,
  onUpdated,
}: {
  runId: string
  item: TopicReviewItem
  syllabusOptions: SyllabusOption[]
  onUpdated: (data: TopicsResponse) => void
}) {
  const [mode, setMode] =
    useState<'idle' | 'correct'>('idle')
  const [decision, setDecision] =
    useState<'mapped' | 'out_of_syllabus'>('mapped')
  const [conceptId, setConceptId] = useState('')
  const [reason, setReason] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const reviewId = Number(item.id)
  const candidates = Array.isArray(item.qdrant_candidates)
    ? item.qdrant_candidates
    : []

  const candidateOptions: SyllabusOption[] = candidates
    .filter((candidate) => candidate?.concept_id)
    .map((candidate) => ({
      concept_id: String(candidate.concept_id),
      label: String(
        candidate.label
        || candidate.concept_id
      ),
      official_reference: String(
        candidate.official_reference || ''
      ),
    }))

  const options =
    candidateOptions.length > 0
      ? candidateOptions
      : syllabusOptions

  const saveSimple = async (
    action: 'approve' | 'reject'
  ) => {
    if (!Number.isFinite(reviewId)) return

    setSaving(true)
    setError(null)

    try {
      const updated = await submitMappingReview(
        runId,
        reviewId,
        {
          action,
          reason: reason.trim(),
        }
      )

      onUpdated(updated)
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Could not save review.'
      )
    } finally {
      setSaving(false)
    }
  }

  const saveCorrection = async () => {
    if (!Number.isFinite(reviewId)) return

    if (!reason.trim()) {
      setError('A written correction reason is required.')
      return
    }

    if (decision === 'mapped' && !conceptId) {
      setError('Select the correct official AQA topic.')
      return
    }

    setSaving(true)
    setError(null)

    try {
      const updated = await submitMappingReview(
        runId,
        reviewId,
        {
          action: 'correct',
          corrected_decision: decision,
          corrected_mapped_concept_id:
            decision === 'mapped'
              ? conceptId
              : undefined,
          reason: reason.trim(),
        }
      )

      onUpdated(updated)
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Could not save correction.'
      )
    } finally {
      setSaving(false)
    }
  }

  return (
    <Card>
      <div className="topic-head">
        <div>
          <p className="eyebrow">
            MAPPING REVIEW · #{Number.isFinite(reviewId) ? reviewId : '—'}
          </p>
          <h2>
            {String(
              item.original_topic
              || item.rough_topic
              || 'Detected rough topic'
            )}
          </h2>

          <p className="muted">
            Proposed concept:{' '}
            {String(
              item.proposed_mapped_concept_id
              || 'Out of syllabus / unresolved'
            )}
          </p>
        </div>

        <Badge tone="warning">Human review required</Badge>
      </div>

      {item.evidence_text && (
        <div className="evidence">
          <p className="eyebrow">Lesson evidence</p>
          <p>“{String(item.evidence_text)}”</p>
        </div>
      )}

      <div className="reason-box">
        <div className="reason-title">
          <Info size={16} />
          <strong>System Decision Explanation</strong>
        </div>

        <div className="reason-grid">
          <div>
            <span>Confidence</span>
            <strong>{confidenceLabel(item.confidence)}</strong>
          </div>

          <div>
            <span>Source chunks</span>
            <strong>
              {Array.isArray(item.source_chunk_ids)
                ? item.source_chunk_ids.join(', ')
                : '—'}
            </strong>
          </div>

          <div className="reason-why">
            <span>Why</span>
            <p>
              {String(
                item.reason
                || 'This mapping reached the configured human-review boundary.'
              )}
            </p>
          </div>
        </div>
      </div>

      {error && (
        <div className="guidance" style={{ marginTop: '1rem' }}>
          <XCircle size={18} />
          <div>
            <strong>Could not save review</strong>
            <p>{error}</p>
          </div>
        </div>
      )}

      <div className="topic-actions">
        <Button
          variant="secondary"
          disabled={saving}
          onClick={() => saveSimple('approve')}
        >
          Approve Mapping
        </Button>

        <Button
          variant="danger"
          disabled={saving}
          onClick={() => saveSimple('reject')}
        >
          Reject
        </Button>

        <Button
          variant="ghost"
          disabled={saving}
          onClick={() =>
            setMode(mode === 'correct' ? 'idle' : 'correct')
          }
        >
          Correct Mapping
        </Button>
      </div>

      {mode === 'correct' && (
        <div className="feedback-form">
          <p className="eyebrow">Correct mapping</p>

          <select
            aria-label="Correction decision"
            value={decision}
            onChange={(event) =>
              setDecision(
                event.target.value as
                  | 'mapped'
                  | 'out_of_syllabus'
              )
            }
          >
            <option value="mapped">Official AQA topic</option>
            <option value="out_of_syllabus">
              Out of syllabus
            </option>
          </select>

          {decision === 'mapped' && (
            <select
              aria-label="Correct official AQA topic"
              value={conceptId}
              onChange={(event) =>
                setConceptId(event.target.value)
              }
            >
              <option value="">
                Select correct official topic
              </option>

              {options.map((option) => (
                <option
                  value={option.concept_id}
                  key={option.concept_id}
                >
                  {optionLabel(option)}
                </option>
              ))}
            </select>
          )}

          <textarea
            value={reason}
            onChange={(event) =>
              setReason(event.target.value)
            }
            placeholder="Why is the system suggestion wrong? A written reason is required."
          />

          <div className="form-actions">
            <Button
              onClick={saveCorrection}
              disabled={
                saving
                || !reason.trim()
                || (
                  decision === 'mapped'
                  && !conceptId
                )
              }
            >
              {saving
                ? 'Saving...'
                : 'Save Correction'}
            </Button>

            <Button
              variant="ghost"
              onClick={() => {
                setMode('idle')
                setError(null)
              }}
            >
              Cancel
            </Button>
          </div>
        </div>
      )}
    </Card>
  )
}


function TopicEditForm({
  runId,
  topic,
  action,
  syllabusOptions,
  onUpdated,
  onCancel,
}: {
  runId: string
  topic: TopicItem
  action: 'change_role' | 'replace_topic' | 'remove_topic'
  syllabusOptions: SyllabusOption[]
  onUpdated: (data: TopicsResponse) => void
  onCancel: () => void
}) {
  const currentRole =
    String(topic.role || topic.topic_role || 'supporting')
      .toLowerCase() === 'primary'
      ? 'primary'
      : 'supporting'

  const [targetRole, setTargetRole] =
    useState<'primary' | 'supporting'>(
      currentRole === 'primary'
        ? 'supporting'
        : 'primary'
    )
  const [targetConceptId, setTargetConceptId] =
    useState('')
  const [reason, setReason] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const save = async () => {
    if (!reason.trim()) {
      setError('A written reason is required.')
      return
    }

    if (
      action === 'replace_topic'
      && !targetConceptId
    ) {
      setError('Select the replacement topic.')
      return
    }

    setSaving(true)
    setError(null)

    try {
      const updated = await submitTopicEdit(
        runId,
        {
          action,
          reason: reason.trim(),
          topic_index: Number(topic.effective_index ?? topic.topic_index),
          source_concept_id: String(
            topic.concept_id || ''
          ) || undefined,
          target_concept_id:
            action === 'replace_topic'
              ? targetConceptId
              : undefined,
          target_role:
            action === 'change_role'
              ? targetRole
              : action === 'replace_topic'
                ? currentRole
                : undefined,
        }
      )

      onUpdated(updated)
      onCancel()
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Could not save topic edit.'
      )
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="feedback-form">
      <p className="eyebrow">
        {action === 'change_role'
          ? 'Change role'
          : action === 'replace_topic'
            ? 'Replace topic'
            : 'Remove topic'}
      </p>

      {action === 'change_role' && (
        <select
          aria-label="New topic role"
          value={targetRole}
          onChange={(event) =>
            setTargetRole(
              event.target.value as
                | 'primary'
                | 'supporting'
            )
          }
        >
          <option value="primary">Primary</option>
          <option value="supporting">Supporting</option>
        </select>
      )}

      {action === 'replace_topic' && (
        <select
          aria-label="Replacement topic"
          value={targetConceptId}
          onChange={(event) =>
            setTargetConceptId(event.target.value)
          }
        >
          <option value="">
            Select replacement official topic
          </option>

          {syllabusOptions
            .filter(
              (option) =>
                option.concept_id !== topic.concept_id
            )
            .map((option) => (
              <option
                value={option.concept_id}
                key={option.concept_id}
              >
                {optionLabel(option)}
              </option>
            ))}
        </select>
      )}

      <textarea
        value={reason}
        onChange={(event) =>
          setReason(event.target.value)
        }
        placeholder="Explain why this change is correct based on the lesson evidence. A reason is required."
      />

      {error && (
        <p style={{ margin: 0 }}>
          <strong>{error}</strong>
        </p>
      )}

      <div className="form-actions">
        <Button
          onClick={save}
          variant={
            action === 'remove_topic'
              ? 'danger'
              : 'primary'
          }
          disabled={
            saving
            || !reason.trim()
            || (
              action === 'replace_topic'
              && !targetConceptId
            )
          }
        >
          {saving ? 'Saving...' : 'Apply Decision'}
        </Button>

        <Button
          variant="ghost"
          onClick={onCancel}
        >
          Cancel
        </Button>
      </div>
    </div>
  )
}


function AddTopicForm({
  runId,
  syllabusOptions,
  chunks,
  onUpdated,
  onCancel,
}: {
  runId: string
  syllabusOptions: SyllabusOption[]
  chunks: SemanticResponse['chunks']
  onUpdated: (data: TopicsResponse) => void
  onCancel: () => void
}) {
  const [conceptId, setConceptId] = useState('')
  const [role, setRole] =
    useState<'primary' | 'supporting'>('supporting')
  const [selectedChunks, setSelectedChunks] =
    useState<number[]>([])
  const [reason, setReason] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const toggleChunk = (chunkId: number) => {
    setSelectedChunks((current) =>
      current.includes(chunkId)
        ? current.filter((value) => value !== chunkId)
        : [...current, chunkId]
    )
  }

  const save = async () => {
    if (!conceptId) {
      setError('Select the missing official topic.')
      return
    }

    if (selectedChunks.length === 0) {
      setError(
        'Select at least one semantic chunk as lesson evidence.'
      )
      return
    }

    if (!reason.trim()) {
      setError('A written reason is required.')
      return
    }

    setSaving(true)
    setError(null)

    try {
      const updated = await submitTopicEdit(
        runId,
        {
          action: 'add_topic',
          reason: reason.trim(),
          target_concept_id: conceptId,
          target_role: role,
          source_chunk_ids: selectedChunks,
        }
      )

      onUpdated(updated)
      onCancel()
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Could not add topic.'
      )
    } finally {
      setSaving(false)
    }
  }

  return (
    <Card>
      <div className="card-heading">
        <div>
          <p className="eyebrow">HUMAN CORRECTION</p>
          <h2>Add Missing Topic</h2>
        </div>
        <Badge tone="warning">Reason required</Badge>
      </div>

      <div className="feedback-form">
        <select
          aria-label="Missing official topic"
          value={conceptId}
          onChange={(event) =>
            setConceptId(event.target.value)
          }
        >
          <option value="">
            Select official AQA topic
          </option>

          {syllabusOptions.map((option) => (
            <option
              value={option.concept_id}
              key={option.concept_id}
            >
              {optionLabel(option)}
            </option>
          ))}
        </select>

        <select
          aria-label="Added topic role"
          value={role}
          onChange={(event) =>
            setRole(
              event.target.value as
                | 'primary'
                | 'supporting'
            )
          }
        >
          <option value="primary">Primary</option>
          <option value="supporting">Supporting</option>
        </select>

        <div>
          <p className="eyebrow" style={{ marginBottom: '0.6rem' }}>
            Select lesson evidence chunks
          </p>

          <div
            style={{
              display: 'grid',
              gap: '0.55rem',
              maxHeight: '220px',
              overflowY: 'auto',
            }}
          >
            {chunks.map((chunk) => {
              const chunkId = Number(chunk.chunk_id)

              return (
                <label
                  className="checkline"
                  key={chunkId}
                >
                  <input
                    type="checkbox"
                    checked={selectedChunks.includes(chunkId)}
                    onChange={() => toggleChunk(chunkId)}
                  />
                  Chunk {chunkId} ·{' '}
                  {String(chunk.text || '').slice(0, 120)}
                  {String(chunk.text || '').length > 120
                    ? '…'
                    : ''}
                </label>
              )
            })}
          </div>
        </div>

        <textarea
          value={reason}
          onChange={(event) =>
            setReason(event.target.value)
          }
          placeholder="Explain why this topic was genuinely taught and should be added."
        />

        {error && (
          <p style={{ margin: 0 }}>
            <strong>{error}</strong>
          </p>
        )}

        <div className="form-actions">
          <Button
            onClick={save}
            disabled={
              saving
              || !conceptId
              || selectedChunks.length === 0
              || !reason.trim()
            }
          >
            {saving ? 'Saving...' : 'Add Topic'}
          </Button>

          <Button
            variant="ghost"
            onClick={onCancel}
          >
            Cancel
          </Button>
        </div>
      </div>
    </Card>
  )
}


function HistoricalMemoryReviewCard({
  runId,
  item,
  onUpdated,
}: {
  runId: string
  item: HistoricalMemoryReviewItem
  onUpdated: (data: TopicsResponse) => void
}) {
  const firstMemory = item.memories[0]

  const [selectedMemoryId, setSelectedMemoryId] =
    useState<number>(
      firstMemory?.memory_id
      ?? item.memory_ids[0]
      ?? 0
    )

  const [reason, setReason] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] =
    useState<string | null>(null)

  const selectedMemory =
    item.memories.find(
      (memory) =>
        Number(memory.memory_id)
        === Number(selectedMemoryId)
    )
    || firstMemory

  const fresh =
    item.fresh_topic
    || selectedMemory?.fresh_topic
    || null

  const freshRole = String(
    fresh?.role
    || fresh?.topic_role
    || selectedMemory?.source_role
    || 'supporting'
  ).toLowerCase()

  const currentDecision =
    selectedMemory?.saved_decision
    || item.saved_decision
    || null

  const decisionRequired =
    item.status === 'decision_required'

  const isAddTopic =
    selectedMemory?.edit_action === 'add_topic'

  const addTopicName =
    String(
      selectedMemory?.target_topic
      || item.topic_label
      || 'this topic'
    ).trim()

  const displayTopicLabel =
    isAddTopic
      ? addTopicName
      : item.topic_label

  const historicalButtonLabel =
    isAddTopic
      ? `Add ${addTopicName}`
      : 'Use Previous Human Decision'

  const freshButtonLabel =
    isAddTopic
      ? `Keep Result Without ${addTopicName}`
      : 'Keep Fresh Result'

  const friendlyWhy = isAddTopic
    ? (
        decisionRequired
          ? `A previous reviewer added ${addTopicName} in a similar lesson, but EDTech is not confident enough to add it automatically here. Confirm whether it was actually taught in this lesson.`
          : `A previous reviewer added ${addTopicName} in a closely matching lesson. You can still keep the fresh result without it if the current lesson does not teach this topic.`
      )
    : (
        decisionRequired
          ? 'This lesson is similar to a previous correction, but EDTech is not confident enough to apply that correction automatically. Please choose the result that best matches what was actually taught in this lesson.'
          : 'This lesson closely matched a previous reviewer correction, so EDTech reused it automatically. You can still override that choice if the fresh lesson evidence should take priority.'
      )

  const saveDecision = async (
    decision:
      | 'use_historical'
      | 'keep_fresh'
  ) => {
    if (!reason.trim()) {
      setError(
        'Add a short reason so EDTech can learn from this decision.'
      )
      return
    }

    setSaving(true)
    setError(null)

    try {
      const updated =
        await submitHistoricalMemoryReview(
          runId,
          {
            decision,
            memory_ids: item.memory_ids,
            selected_memory_id:
              decision === 'use_historical'
                ? Number(selectedMemoryId)
                : undefined,
            reason: reason.trim(),
          }
        )

      onUpdated(updated)
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Could not save the historical HITL decision.'
      )
    } finally {
      setSaving(false)
    }
  }

  return (
    <Card>
      <div
        style={{
          borderRadius: '18px',
          overflow: 'hidden',
          border:
            '1px solid rgba(255,220,153,0.30)',
          background:
            'linear-gradient(135deg, rgba(255,208,122,0.075), rgba(255,255,255,0.035))',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'flex-start',
            justifyContent: 'space-between',
            gap: '1rem',
            padding: '1rem 1.05rem',
            borderBottom:
              '1px solid rgba(255,255,255,0.11)',
          }}
        >
          <div
            style={{
              display: 'flex',
              gap: '0.8rem',
              alignItems: 'flex-start',
            }}
          >
            <div
              style={{
                width: '40px',
                height: '40px',
                flex: '0 0 auto',
                display: 'grid',
                placeItems: 'center',
                borderRadius: '12px',
                background:
                  'rgba(255,206,117,0.14)',
                border:
                  '1px solid rgba(255,221,160,0.20)',
              }}
            >
              <History size={19} />
            </div>

            <div>
              <p
                className="eyebrow"
                style={{
                  marginBottom: '0.22rem',
                }}
              >
                EDTECH MEMORY CHECK
              </p>

              <h2
                style={{
                  margin: 0,
                  fontSize: '1.15rem',
                }}
              >
                {isAddTopic
                  ? `Possible missing topic: ${addTopicName}`
                  : `Previous correction found for ${displayTopicLabel}`}
              </h2>

              <p
                className="muted"
                style={{
                  margin: '0.28rem 0 0',
                  lineHeight: 1.5,
                }}
              >
                {isAddTopic
                  ? `EDTech previously learned that ${addTopicName} can be missed in a similar lesson. Decide whether it belongs in this lesson's final topic list.`
                  : 'Choose whether this lesson should use the fresh Module 3 result or the previous human correction.'}
              </p>
            </div>
          </div>

          <Badge
            tone={
              decisionRequired
                ? 'warning'
                : 'info'
            }
          >
            {decisionRequired
              ? 'Decision required'
              : 'Memory applied'}
          </Badge>
        </div>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns:
              'repeat(2, minmax(0, 1fr))',
            alignItems: 'start',
            gap: '0.8rem',
            padding: '0.95rem 1rem',
          }}
        >
          <div
            style={{
              borderRadius: '14px',
              padding: '0.9rem',
              background:
                'rgba(255,255,255,0.065)',
              border:
                '1px solid rgba(255,255,255,0.12)',
            }}
          >
            <p
              className="eyebrow"
              style={{
                marginBottom: '0.38rem',
              }}
            >
              FRESH RESULT
            </p>

            <strong
              style={{
                display: 'block',
                fontSize: '1rem',
              }}
            >
              {isAddTopic
                ? `${addTopicName} was not detected`
                : fresh
                  ? `Keep ${topicName(fresh)}`
                  : `Keep current ${displayTopicLabel} result`}
            </strong>

            {isAddTopic && (
              <p
                className="muted"
                style={{
                  margin: '0.45rem 0 0',
                  lineHeight: 1.5,
                  fontSize: '0.86rem',
                }}
              >
                Module 3 did not include {addTopicName} in the
                current lesson result.
              </p>
            )}

            <div
              style={{
                display: 'flex',
                flexWrap: 'wrap',
                gap: '0.4rem',
                marginTop: '0.55rem',
              }}
            >
              {fresh?.official_reference && (
                <span
                  style={{
                    padding: '0.3rem 0.5rem',
                    borderRadius: '999px',
                    background:
                      'rgba(255,255,255,0.09)',
                    border:
                      '1px solid rgba(255,255,255,0.13)',
                    fontSize: '0.76rem',
                  }}
                >
                  AQA {fresh.official_reference}
                </span>
              )}

              {!isAddTopic && (
                <Badge
                  tone={
                    freshRole === 'primary'
                      ? 'teal'
                      : 'warning'
                  }
                >
                  {freshRole === 'primary'
                    ? 'Primary'
                    : 'Supporting'}
                </Badge>
              )}

              {fresh && (
                <span
                  style={{
                    padding: '0.3rem 0.5rem',
                    borderRadius: '999px',
                    background:
                      'rgba(255,255,255,0.09)',
                    border:
                      '1px solid rgba(255,255,255,0.13)',
                    fontSize: '0.76rem',
                  }}
                >
                  {confidenceLabel(
                    fresh.confidence
                  )}
                </span>
              )}
            </div>
          </div>

          <div
            style={{
              borderRadius: '14px',
              padding: '0.9rem',
              background:
                'rgba(255,194,94,0.085)',
              border:
                '1px solid rgba(255,219,159,0.17)',
            }}
          >
            <p
              className="eyebrow"
              style={{
                marginBottom: '0.38rem',
              }}
            >
              PREVIOUS HUMAN DECISION
            </p>

            <strong
              style={{
                display: 'block',
                fontSize: '1rem',
              }}
            >
              {selectedMemory?.historical_outcome
                || 'Use previous correction'}
            </strong>

            {isAddTopic && selectedMemory?.target_role && (
              <div
                style={{
                  display: 'flex',
                  gap: '0.4rem',
                  marginTop: '0.5rem',
                }}
              >
                <Badge
                  tone={
                    String(selectedMemory.target_role).toLowerCase()
                      === 'primary'
                      ? 'teal'
                      : 'warning'
                  }
                >
                  {String(selectedMemory.target_role).toLowerCase()
                    === 'primary'
                    ? 'Primary'
                    : 'Supporting'}
                </Badge>
              </div>
            )}

            {selectedMemory?.reviewer_reason && (
              <p
                style={{
                  margin: '0.55rem 0 0',
                  lineHeight: 1.55,
                  fontSize: '0.88rem',
                }}
              >
                {selectedMemory.reviewer_reason}
              </p>
            )}
          </div>
        </div>

        {item.memories.length > 1 && (
          <div
            style={{
              padding: '0 1rem 0.8rem',
            }}
          >
            <label
              style={{
                display: 'grid',
                gap: '0.35rem',
              }}
            >
              <strong
                style={{
                  fontSize: '0.88rem',
                }}
              >
                Previous correction to compare
              </strong>

              <select
                value={selectedMemoryId}
                onChange={(event) =>
                  setSelectedMemoryId(
                    Number(event.target.value)
                  )
                }
                style={{
                  width: '100%',
                  padding: '0.68rem',
                  borderRadius: '10px',
                  border:
                    '1px solid rgba(255,255,255,0.18)',
                  background: '#8878b8',
                  color: '#fff',
                }}
              >
                {item.memories.map(
                  (memory) => (
                    <option
                      key={memory.memory_id}
                      value={memory.memory_id}
                    >
                      {memory.historical_outcome}
                    </option>
                  )
                )}
              </select>
            </label>
          </div>
        )}

        <div
          style={{
            margin: '0 1rem 0.8rem',
            padding: '0.78rem 0.85rem',
            borderRadius: '12px',
            background:
              'rgba(255,255,255,0.05)',
            border:
              '1px solid rgba(255,255,255,0.09)',
          }}
        >
          <div
            style={{
              display: 'flex',
              gap: '0.55rem',
              alignItems: 'flex-start',
            }}
          >
            <Info
              size={16}
              style={{
                marginTop: '0.12rem',
                flex: '0 0 auto',
              }}
            />

            <div>
              <strong>
                Why EDTech is asking
              </strong>

              <p
                className="muted"
                style={{
                  margin: '0.2rem 0 0',
                  lineHeight: 1.5,
                }}
              >
                {friendlyWhy}
              </p>
            </div>
          </div>

          {(selectedMemory?.context_diagnostic
            || selectedMemory?.runtime_reason) && (
            <details
              style={{
                marginTop: '0.6rem',
                fontSize: '0.78rem',
              }}
            >
              <summary
                style={{
                  cursor: 'pointer',
                  opacity: 0.78,
                  fontWeight: 700,
                }}
              >
                Technical details
              </summary>

              <p
                className="muted"
                style={{
                  margin: '0.45rem 0 0',
                  lineHeight: 1.5,
                  wordBreak: 'break-word',
                }}
              >
                {selectedMemory.context_diagnostic
                  || selectedMemory.runtime_reason}
              </p>
            </details>
          )}
        </div>

        {currentDecision && (
          <div
            className="success-banner"
            style={{
              margin: '0 auto 0.8rem',
            }}
          >
            <CheckCircle2 size={17} />

            <div>
              <strong>
                Decision saved
              </strong>
              <span>
                {currentDecision
                  === 'approve_reuse'
                  ? 'Using the previous human correction for this lesson.'
                  : 'Keeping the fresh Module 3 result for this lesson.'}
              </span>
            </div>
          </div>
        )}

        <div
          style={{
            padding: '0 1rem 1rem',
          }}
        >
          <label
            style={{
              display: 'grid',
              gap: '0.4rem',
            }}
          >
            <strong
              style={{
                fontSize: '0.9rem',
              }}
            >
              Why are you choosing this?
            </strong>

            <textarea
              value={reason}
              onChange={(event) =>
                setReason(
                  event.target.value
                )
              }
              placeholder={
                isAddTopic
                  ? `Briefly explain why ${addTopicName} should be added or left out of this lesson.`
                  : 'A short reason helps EDTech make better reuse decisions next time.'
              }
              style={{
                width: '100%',
                minHeight: '72px',
                resize: 'vertical',
                padding: '0.75rem',
                borderRadius: '11px',
                border:
                  '1px solid rgba(255,255,255,0.17)',
                background:
                  'rgba(255,255,255,0.075)',
                color: '#fff',
                font: 'inherit',
              }}
            />
          </label>

          {error && (
            <div
              className="guidance"
              style={{
                marginTop: '0.7rem',
              }}
            >
              <XCircle size={17} />
              <div>
                <strong>
                  Could not save decision
                </strong>
                <p>{error}</p>
              </div>
            </div>
          )}

          <div
            style={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: '0.6rem',
              marginTop: '0.75rem',
            }}
          >
            <Button
              disabled={
                saving
                || !reason.trim()
                || !selectedMemory
              }
              onClick={() =>
                saveDecision(
                  'use_historical'
                )
              }
            >
              {saving
                ? 'Saving...'
                : historicalButtonLabel}
              {!saving && (
                <History size={16} />
              )}
            </Button>

            <Button
              variant="secondary"
              disabled={
                saving
                || !reason.trim()
              }
              onClick={() =>
                saveDecision(
                  'keep_fresh'
                )
              }
            >
              {freshButtonLabel}
            </Button>
          </div>
        </div>
      </div>
    </Card>
  )
}


function Topics({
  go,
  data,
  semanticData,
  onUpdated,
}: {
  go: (p: string) => void
  data: TopicsResponse
  semanticData: SemanticResponse | null
  onUpdated: (data: TopicsResponse) => void
}) {
  const [editState, setEditState] = useState<{
    index: number
    action: 'change_role' | 'replace_topic' | 'remove_topic'
  } | null>(null)

  const [showAdd, setShowAdd] = useState(false)
  const [selected, setSelected] = useState<number[]>([])
  const [approving, setApproving] = useState(false)
  const [approvalError, setApprovalError] =
    useState<string | null>(null)

  useEffect(() => {
    setSelected(
      data.topics.map((topic) =>
        Number(topic.topic_index)
      )
    )
  }, [data.topics])

  const memoryPending =
    data.historical_memory_pending_count > 0

  const pending =
    data.pending_review_count > 0
    || memoryPending

  const toggleSelected = (index: number) => {
    setSelected((current) =>
      current.includes(index)
        ? current.filter((value) => value !== index)
        : [...current, index]
    )
  }

  const approveSelected = async () => {
    if (selected.length === 0) {
      setApprovalError(
        'Select at least one topic for Agent 2.'
      )
      return
    }

    setApproving(true)
    setApprovalError(null)

    try {
      const updated = await approveTopicsForAgent2(
        data.run_id,
        selected
      )

      onUpdated(updated)
      go('assessment')
    } catch (err) {
      setApprovalError(
        err instanceof Error
          ? err.message
          : 'Could not approve topics.'
      )
    } finally {
      setApproving(false)
    }
  }

  return (
    <>
      <Header
        eyebrow="Agent 1 / Topic mapping"
        title="Syllabus Topic Mapping"
        description="Review the real Module 3 output, resolve mandatory HITL items, and approve the final lesson topics."
      />

      <Stepper current={3} />

      <div className="guidance">
        <Info size={19} />
        <div>
          <strong>Human-in-the-loop is authoritative</strong>
          <p>
            Corrections are saved through the existing Agent 1
            PostgreSQL / contextual memory path. Every final-topic
            edit requires a written reason.
          </p>
        </div>
      </div>

      {data.historical_memory_error && (
        <div className="guidance">
          <XCircle size={19} />
          <div>
            <strong>
              Historical HITL memory unavailable
            </strong>
            <p>
              {data.historical_memory_error}
            </p>
          </div>
        </div>
      )}

      {data.historical_memory_reviews.length > 0 && (
        <section
          style={{
            display: 'grid',
            gap: '0.85rem',
            marginTop: '1rem',
          }}
        >
          <div className="card-heading">
            <div>
              <p className="eyebrow">
                HISTORICAL HITL MEMORY
              </p>

              <h2>
                EDTech remembered previous human corrections
              </h2>

              <p
                className="muted"
                style={{
                  margin:
                    '0.35rem 0 0',
                }}
              >
                EDTech has found previous human corrections
                that may be relevant to this lesson. Review only
                the uncertain ones before final topic approval.
              </p>
            </div>

            <Badge
              tone={
                memoryPending
                  ? 'warning'
                  : 'info'
              }
            >
              {memoryPending
                ? `${data.historical_memory_pending_count} decision${
                    data.historical_memory_pending_count === 1
                      ? ''
                      : 's'
                  } required`
                : 'Memory checked'}
            </Badge>
          </div>

          {data.historical_memory_reviews.map(
            (item) => (
              <HistoricalMemoryReviewCard
                key={item.review_key}
                runId={data.run_id}
                item={item}
                onUpdated={onUpdated}
              />
            )
          )}
        </section>
      )}

      {data.syllabus_error && (
        <div className="guidance">
          <Info size={19} />
          <div>
            <strong>Syllabus dropdown warning</strong>
            <p>{data.syllabus_error}</p>
          </div>
        </div>
      )}

      {data.orphaned_review_count > 0 && (
        <div className="guidance">
          <XCircle size={19} />
          <div>
            <strong>Review integrity issue</strong>
            <p>
              {data.orphaned_review_count} historical review
              record(s) could not be reconciled with PostgreSQL.
              Agent 1 remains fail-closed.
            </p>
          </div>
        </div>
      )}

      {data.pending_review_count > 0 && (
        <>
          <div
            className="guidance"
            style={{ marginTop: '1rem' }}
          >
            <Info size={19} />
            <div>
              <strong>
                {data.pending_review_count} mapping review
                {data.pending_review_count === 1 ? '' : 's'} required
              </strong>
              <p>
                Resolve the remaining AQA mapping decision
                {data.pending_review_count === 1 ? '' : 's'}
                {' '}before final approval.
              </p>
            </div>
          </div>

          <div className="topic-list">
            {data.pending_reviews.map((item, index) => (
              <MappingReviewCard
                key={`${item.id ?? 'review'}-${index}`}
                runId={data.run_id}
                item={item}
                syllabusOptions={data.syllabus_options}
                onUpdated={onUpdated}
              />
            ))}
          </div>
        </>
      )}

      <div
        className="card-heading"
        style={{ marginTop: '1.5rem' }}
      >
        <div>
          <p className="eyebrow">EFFECTIVE TOPIC LIST</p>
          <h2>
            {data.topic_count} Official Topic
            {data.topic_count === 1 ? '' : 's'}
          </h2>
        </div>

        <Badge tone={pending ? 'warning' : 'success'}>
          {memoryPending
            ? 'Historical memory review pending'
            : data.pending_review_count > 0
              ? 'Mapping review pending'
              : 'Mapping resolved'}
        </Badge>
      </div>

      {data.topic_count === 0 ? (
        <Card>
          <p>
            No official AQA topics are currently retained for
            this lesson.
          </p>
        </Card>
      ) : (
        <div className="topic-list">
          {data.topics.map((topic) => {
            const index = Number(topic.topic_index)
            const role = String(
              topic.role
              || topic.topic_role
              || 'supporting'
            ).toLowerCase()

            const isEditing =
              editState?.index === index

            return (
              <Card key={`${topic.concept_id}-${index}`}>
                <div className="topic-head">
                  <div>
                    <h2>{topicName(topic)}</h2>

                    <p className="muted">
                      AQA Mapping ·{' '}
                      {String(
                        topic.official_reference || 'No reference'
                      )}
                      {topic.paper
                        ? ` · ${String(topic.paper)}`
                        : ''}
                    </p>
                  </div>

                  <div className="topic-badges">
                    <Badge
                      tone={
                        role === 'primary'
                          ? 'teal'
                          : 'warning'
                      }
                    >
                      {role === 'primary'
                        ? 'Primary'
                        : 'Supporting'}
                    </Badge>

                    {topic.human_edited && (
                      <Badge tone="info">Human edited</Badge>
                    )}
                  </div>
                </div>

                <div className="reason-box">
                  <div className="reason-title">
                    <Info size={16} />
                    <strong>Current Module 3 Decision</strong>
                  </div>

                  <div className="reason-grid">
                    <div>
                      <span>Official reference</span>
                      <strong>
                        {String(
                          topic.official_reference || '—'
                        )}
                      </strong>
                    </div>

                    <div>
                      <span>Confidence</span>
                      <strong>
                        {confidenceLabel(topic.confidence)}
                      </strong>
                    </div>

                    <div className="reason-why">
                      <span>Source chunks</span>
                      <p>
                        {Array.isArray(topic.source_chunk_ids)
                          ? topic.source_chunk_ids.join(', ')
                          : '—'}
                      </p>
                    </div>
                  </div>
                </div>

                <div className="topic-actions">
                  <Button
                    variant="ghost"
                    disabled={pending}
                    onClick={() =>
                      setEditState({
                        index,
                        action: 'change_role',
                      })
                    }
                  >
                    Change Role
                  </Button>

                  <Button
                    variant="ghost"
                    disabled={pending}
                    onClick={() =>
                      setEditState({
                        index,
                        action: 'replace_topic',
                      })
                    }
                  >
                    Replace Topic
                  </Button>

                  <Button
                    variant="danger"
                    disabled={pending}
                    onClick={() =>
                      setEditState({
                        index,
                        action: 'remove_topic',
                      })
                    }
                  >
                    Remove Topic
                  </Button>
                </div>

                {isEditing && editState && (
                  <TopicEditForm
                    runId={data.run_id}
                    topic={topic}
                    action={editState.action}
                    syllabusOptions={data.syllabus_options}
                    onUpdated={onUpdated}
                    onCancel={() => setEditState(null)}
                  />
                )}
              </Card>
            )
          })}
        </div>
      )}

      {!pending && !showAdd && (
        <Card className="add-topic">
          <Plus size={18} />
          <div>
            <strong>Add Missing Topic</strong>
            <p>
              Use this only when an official concept is genuinely
              explained, demonstrated, or practised in the lesson.
            </p>
          </div>

          <Button
            variant="secondary"
            onClick={() => setShowAdd(true)}
          >
            Add Topic
          </Button>
        </Card>
      )}

      {!pending && showAdd && semanticData && (
        <AddTopicForm
          runId={data.run_id}
          syllabusOptions={data.syllabus_options}
          chunks={semanticData.chunks}
          onUpdated={onUpdated}
          onCancel={() => setShowAdd(false)}
        />
      )}

      {data.topic_count > 0 && (
        <Card>
          <div className="card-heading">
            <div>
              <p className="eyebrow">
                FINAL AGENT 1 APPROVAL
              </p>
              <h2>Topics to send to Agent 2</h2>
            </div>

            <Badge
              tone={
                data.agent2_handoff_ready
                  ? 'success'
                  : pending
                    ? 'warning'
                    : 'info'
              }
            >
              {data.agent2_handoff_ready
                ? 'Approved'
                : pending
                  ? 'Locked until reviews are resolved'
                  : 'Ready for approval'}
            </Badge>
          </div>

          <p className="muted">
            This is the final topic list Agent 2 will be allowed
            to use for retrieval or quiz generation.
          </p>

          {pending && (
            <div
              style={{
                display: 'flex',
                gap: '0.65rem',
                alignItems: 'flex-start',
                marginTop: '0.9rem',
                padding: '0.75rem 0.85rem',
                borderRadius: '12px',
                background:
                  'rgba(255,194,94,0.09)',
                border:
                  '1px solid rgba(255,219,159,0.16)',
              }}
            >
              <Info
                size={17}
                style={{
                  marginTop: '0.1rem',
                  flex: '0 0 auto',
                }}
              />
              <div>
                <strong>
                  Finish the review above first
                </strong>
                <p
                  className="muted"
                  style={{
                    margin: '0.2rem 0 0',
                  }}
                >
                  Once all historical-memory and mapping decisions
                  are resolved, this list will unlock automatically.
                </p>
              </div>
            </div>
          )}

          <div
            style={{
              display: 'grid',
              gap: '0.85rem',
              margin: '1.25rem 0 1.5rem',
            }}
          >
            {data.topics.map((topic, topicPosition) => {
              const index = Number(topic.topic_index)
              const isSelected = selected.includes(index)
              const role = String(
                topic.role
                || topic.topic_role
                || 'supporting'
              ).toLowerCase()

              return (
                <label
                  key={`approve-${index}`}
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'auto 1fr auto',
                    alignItems: 'center',
                    gap: '1rem',
                    padding: '1rem 1.1rem',
                    borderRadius: '14px',
                    border: isSelected
                      ? '1px solid rgba(255,255,255,0.72)'
                      : '1px solid rgba(255,255,255,0.24)',
                    background: isSelected
                      ? 'rgba(255,255,255,0.15)'
                      : 'rgba(35,28,50,0.14)',
                    boxShadow: isSelected
                      ? '0 8px 22px rgba(29,22,45,0.10)'
                      : 'none',
                    cursor: pending
                      ? 'not-allowed'
                      : 'pointer',
                    opacity: pending ? 0.62 : 1,
                    transition:
                      'background 160ms ease, border-color 160ms ease, transform 160ms ease',
                  }}
                >
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.8rem',
                    }}
                  >
                    <input
                      type="checkbox"
                      checked={isSelected}
                      disabled={pending}
                      onChange={() => toggleSelected(index)}
                      style={{
                        width: '19px',
                        height: '19px',
                        margin: 0,
                        cursor: 'pointer',
                        accentColor: '#2a2338',
                      }}
                    />

                    <span
                      style={{
                        width: '32px',
                        height: '32px',
                        borderRadius: '10px',
                        display: 'grid',
                        placeItems: 'center',
                        fontSize: '0.8rem',
                        fontWeight: 800,
                        background: 'rgba(255,255,255,0.18)',
                        border: '1px solid rgba(255,255,255,0.22)',
                      }}
                    >
                      {String(topicPosition + 1).padStart(2, '0')}
                    </span>
                  </div>

                  <div
                    style={{
                      display: 'grid',
                      gap: '0.32rem',
                      minWidth: 0,
                    }}
                  >
                    <strong
                      style={{
                        fontSize: '1rem',
                        lineHeight: 1.3,
                      }}
                    >
                      {topicName(topic)}
                    </strong>

                    <span
                      className="muted"
                      style={{
                        fontSize: '0.86rem',
                      }}
                    >
                      Final Agent 1 syllabus concept
                    </span>
                  </div>

                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'flex-end',
                      gap: '0.55rem',
                      flexWrap: 'wrap',
                    }}
                  >
                    <span
                      style={{
                        padding: '0.38rem 0.65rem',
                        borderRadius: '999px',
                        background: 'rgba(255,255,255,0.18)',
                        border: '1px solid rgba(255,255,255,0.24)',
                        fontSize: '0.8rem',
                        fontWeight: 750,
                        whiteSpace: 'nowrap',
                      }}
                    >
                      AQA {String(topic.official_reference || '—')}
                    </span>

                    <Badge
                      tone={
                        role === 'primary'
                          ? 'teal'
                          : 'warning'
                      }
                    >
                      {role === 'primary'
                        ? 'Primary'
                        : 'Supporting'}
                    </Badge>

                    {isSelected && (
                      <span
                        style={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '0.3rem',
                          fontSize: '0.8rem',
                          fontWeight: 800,
                          whiteSpace: 'nowrap',
                        }}
                      >
                        <Check size={15} />
                        Selected
                      </span>
                    )}
                  </div>
                </label>
              )
            })}
          </div>

          {approvalError && (
            <div className="guidance">
              <XCircle size={18} />
              <div>
                <strong>Approval could not be saved</strong>
                <p>{approvalError}</p>
              </div>
            </div>
          )}

          <div className="card-footer">
            <span className="muted">
              {pending ? (
                <>
                  <strong>{data.topic_count}</strong> final topic
                  {data.topic_count === 1 ? '' : 's'} waiting for review completion
                </>
              ) : (
                <>
                  <strong>{selected.length}</strong> of {data.topic_count} topics selected for Agent 2
                </>
              )}
            </span>

            <Button
              onClick={approveSelected}
              disabled={
                approving
                || pending
                || selected.length === 0
              }
            >
              {approving
                ? 'Saving Approval...'
                : 'Approve Selected Topics'}
              {!approving && <Check size={17} />}
            </Button>
          </div>
        </Card>
      )}

      {data.agent2_handoff_ready && (
        <>
          <div className="success-banner">
            <CheckCircle2 size={20} />
            <div>
              <strong>Agent 1 human review completed</strong>
              <span>
                The approved topic handoff is ready for Agent 2.
              </span>
            </div>
          </div>

          <Button onClick={() => go('assessment')}>
            Continue to Assessment
            <ArrowRight size={17} />
          </Button>
        </>
      )}
    </>
  )
}

function markingGuidanceLines(value: unknown): string[] {
  if (!value) return []
  if (typeof value === 'string') {
    return value.trim() ? [value.trim()] : []
  }
  if (Array.isArray(value)) {
    return value
      .map((item) => {
        if (typeof item === 'string') return item
        if (item && typeof item === 'object') {
          const record = item as Record<string, unknown>
          const parts = [
            record.marks ? `[${String(record.marks)}]` : '',
            record.criterion,
            record.text,
            record.marking_point,
            record.guidance,
          ]
            .filter(Boolean)
            .map(String)
          return parts.length ? parts.join(' ') : JSON.stringify(item)
        }
        return String(item ?? '')
      })
      .filter(Boolean)
  }
  if (typeof value === 'object') {
    return [JSON.stringify(value, null, 2)]
  }
  return [String(value)]
}

function QuestionVisuals({
  runId,
  question,
}: {
  runId: string
  question: AssessmentQuestion
}) {
  const paths = Array.isArray(question.visual_paths)
    ? question.visual_paths.filter(Boolean)
    : []

  const visualSpec = question.visual_spec || {}

  const visualCode = String(
    visualSpec.code
    || visualSpec.mermaid
    || visualSpec.content
    || visualSpec.source
    || ''
  ).trim()

  if (!paths.length && !visualCode) return null

  return (
    <section
      style={{
        marginTop: '1rem',
        borderRadius: '18px',
        border: '1px solid rgba(255,255,255,0.18)',
        background: 'rgba(255,255,255,0.07)',
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '1rem',
          padding: '0.9rem 1rem',
          background: 'rgba(255,255,255,0.055)',
          borderBottom: '1px solid rgba(255,255,255,0.12)',
        }}
      >
        <div>
          <p className="eyebrow" style={{ margin: 0, opacity: 0.78 }}>
            VISUAL / DIAGRAM
          </p>
          <strong style={{ display: 'block', marginTop: '0.2rem' }}>
            Question visual
          </strong>
        </div>

        <Badge tone="info">
          {question.visual_type || 'Visual'}
        </Badge>
      </div>

      <div
        style={{
          display: 'grid',
          gap: '0.85rem',
          padding: '1rem',
        }}
      >
        {paths.map((path, index) => (
          <div
            key={`${question.question_id}-visual-${index}`}
            style={{
              background: '#eee9fb',
              borderRadius: '14px',
              padding: '1rem',
              overflow: 'hidden',
              boxShadow: '0 12px 30px rgba(25,18,42,0.14)',
            }}
          >
            <img
              src={assessmentAssetUrl(runId, path)}
              alt={`Visual for question ${question.question_number || question.question_id}`}
              style={{
                display: 'block',
                width: '100%',
                maxHeight: '540px',
                objectFit: 'contain',
                borderRadius: '9px',
              }}
            />
          </div>
        ))}

        {!paths.length && visualCode && (
          <pre
            style={{
              margin: 0,
              padding: '1rem',
              borderRadius: '12px',
              overflowX: 'auto',
              background: 'rgba(29,23,42,0.5)',
              border: '1px solid rgba(255,255,255,0.12)',
              whiteSpace: 'pre-wrap',
              font: 'inherit',
            }}
          >
            {visualCode}
          </pre>
        )}
      </div>
    </section>
  )
}

function RetrievalHITL({
  runId,
  question,
  onUpdated,
}: {
  runId: string
  question: AssessmentQuestion
  onUpdated: (status: AssessmentStatusResponse) => void
}) {
  const saved = question.retrieval_feedback
  const [decision, setDecision] = useState<'relevant' | 'not_relevant'>(
    saved?.decision === 'not_relevant'
      ? 'not_relevant'
      : 'relevant'
  )
  const [reason, setReason] = useState(saved?.reason || '')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setDecision(
      question.retrieval_feedback?.decision === 'not_relevant'
        ? 'not_relevant'
        : 'relevant'
    )
    setReason(question.retrieval_feedback?.reason || '')
  }, [question.retrieval_feedback?.decision, question.retrieval_feedback?.reason])

  const save = async () => {
    if (decision === 'not_relevant' && !reason.trim()) {
      setError('Please add a reason for Not Relevant feedback.')
      return
    }
    setSaving(true)
    setError(null)
    try {
      const updated = await submitRetrievalFeedback(
        runId,
        question.question_id,
        {
          decision,
          reason: reason.trim(),
        }
      )
      onUpdated(updated)
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Could not save retrieval feedback.'
      )
    } finally {
      setSaving(false)
    }
  }

  return (
    <div
      style={{
        marginTop: '1.15rem',
        padding: '1rem',
        borderRadius: '14px',
        background: 'rgba(31,25,45,0.22)',
        border: '1px solid rgba(255,255,255,0.2)',
        display: 'grid',
        gap: '0.8rem',
      }}
    >
      <div>
        <p className="eyebrow" style={{ marginBottom: '0.25rem' }}>
          RETRIEVAL HITL · SELF-IMPROVING MEMORY
        </p>
        <strong>Does this official question match what was taught?</strong>
        <p className="muted" style={{ marginTop: '0.3rem' }}>
          Feedback is saved with the lesson context. Compatible future retrievals can learn from it.
        </p>
      </div>

      {saved?.decision && (
        <div className="success-banner" style={{ margin: 0 }}>
          <CheckCircle2 size={17} />
          <div>
            <strong>
              Saved: {saved.decision === 'relevant' ? 'Relevant' : 'Not Relevant'}
            </strong>
            <span>
              Memory status: {saved.memory_status || 'saved'}
              {saved.memory_eligible ? ' · indexed in retrieval memory' : ''}
            </span>
          </div>
        </div>
      )}

      {saved?.memory_error && (
        <div className="guidance">
          <Info size={17} />
          <div>
            <strong>Feedback saved, memory indexing needs attention</strong>
            <p>{saved.memory_error}</p>
          </div>
        </div>
      )}

      <div className="rating-actions" style={{ justifyContent: 'flex-start' }}>
        <Button
          variant={decision === 'relevant' ? 'primary' : 'secondary'}
          disabled={saving}
          onClick={() => setDecision('relevant')}
        >
          Relevant
        </Button>
        <Button
          variant={decision === 'not_relevant' ? 'danger' : 'secondary'}
          disabled={saving}
          onClick={() => setDecision('not_relevant')}
        >
          Not Relevant
        </Button>
      </div>

      <textarea
        aria-label="Retrieval feedback reason"
        value={reason}
        onChange={(event) => setReason(event.target.value)}
        placeholder={
          decision === 'not_relevant'
            ? 'Reason is required — explain what does not match the lesson.'
            : 'Optional reason / note'
        }
        style={{
          width: '100%',
          minHeight: '82px',
          resize: 'vertical',
          borderRadius: '10px',
          border: '1px solid rgba(255,255,255,0.24)',
          background: 'rgba(255,255,255,0.12)',
          color: 'inherit',
          padding: '0.75rem',
          font: 'inherit',
        }}
      />

      {error && (
        <div className="guidance">
          <XCircle size={17} />
          <div><strong>Could not save feedback</strong><p>{error}</p></div>
        </div>
      )}

      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
        <Button onClick={save} disabled={saving}>
          {saving ? 'Saving...' : 'Save Retrieval Feedback'}
          {!saving && <Check size={16} />}
        </Button>
      </div>
    </div>
  )
}

function formatHistoricalEtaRange(
  lowSeconds: number | null | undefined,
  highSeconds: number | null | undefined,
  elapsedSeconds = 0
) {
  if (
    typeof lowSeconds !== 'number'
    || typeof highSeconds !== 'number'
  ) {
    return null
  }

  const low = Math.max(0, lowSeconds - elapsedSeconds)
  const high = Math.max(low, highSeconds - elapsedSeconds)

  if (high < 20) return '≈ finishing soon'

  if (high < 60) {
    const lowRounded = Math.max(10, Math.round(low / 10) * 10)
    const highRounded = Math.max(lowRounded + 10, Math.round(high / 10) * 10)
    return `≈ ${lowRounded}–${highRounded} sec remaining`
  }

  const lowMinutes = Math.max(1, Math.floor(low / 60))
  const highMinutes = Math.max(lowMinutes, Math.ceil(high / 60))

  return lowMinutes === highMinutes
    ? `≈ ${highMinutes} min remaining`
    : `≈ ${lowMinutes}–${highMinutes} min remaining`
}

function formatElapsed(totalSeconds: number) {
  const seconds = Math.max(0, Math.floor(totalSeconds))
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  const remainder = seconds % 60
  return `${minutes}m ${String(remainder).padStart(2, '0')}s`
}

function useProcessElapsed(
  startedAtUtc: string | undefined,
  running: boolean
) {
  const [fallbackStartedAt, setFallbackStartedAt] = useState(() => Date.now())
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    if (!running) return
    setFallbackStartedAt(Date.now())
    setNow(Date.now())
  }, [running, startedAtUtc])

  useEffect(() => {
    if (!running) return
    const timer = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(timer)
  }, [running])

  const parsed = startedAtUtc ? Date.parse(startedAtUtc) : Number.NaN
  const startedAt = Number.isFinite(parsed) ? parsed : fallbackStartedAt
  return Math.max(0, Math.floor((now - startedAt) / 1000))
}

function assessmentStageLabel(status: AssessmentStatusResponse) {
  const raw = `${status.stage || ''} ${status.message || ''}`.toLowerCase()

  if (/generating_shortfall|missing coverage|shortfall/.test(raw)) {
    return 'Generating missing questions'
  }
  if (/final|package|pdf/.test(raw)) return 'Finalising assessment'
  if (/quality|semantic|qa|review gate/.test(raw)) return 'Quality checks'
  if (/marking|mark scheme/.test(raw)) return 'Generating marking schemes'
  if (/validat|verify/.test(raw)) return 'Validating questions'
  if (/generat/.test(raw)) return 'Generating questions'
  if (/rank|retriev|search|official/.test(raw)) return 'Retrieving official questions'
  if (/queue|prepar|start/.test(raw)) return 'Preparing assessment'

  return String(status.stage || 'Processing assessment')
    .replaceAll('_', ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
}

function AssessmentTiming({ status }: { status: AssessmentStatusResponse }) {
  const state = String(status.status || '').toLowerCase()
  const running = state === 'queued' || state === 'running'
  const stageElapsed = useProcessElapsed(
    status.stage_started_at_utc || status.started_at_utc,
    running
  )

  if (!running) return null

  const estimate = status.eta_label
    || 'Learning ETA from recent Agent 2 runs…'

  const basis = status.eta_basis
    || 'Estimate improves automatically as more Agent 2 runs complete.'

  return (
    <div
      style={{
        marginTop: '0.9rem',
        display: 'flex',
        alignItems: 'flex-start',
        gap: '0.65rem',
        padding: '0.78rem 0.9rem',
        borderRadius: '12px',
        background: 'rgba(255,255,255,0.09)',
        border: '1px solid rgba(255,255,255,0.16)',
        textAlign: 'left',
      }}
    >
      <Clock3 size={17} style={{ marginTop: '0.12rem', flex: '0 0 auto' }} />
      <div style={{ display: 'grid', gap: '0.25rem' }}>
        <strong>{assessmentStageLabel(status)}</strong>
        <span>
          Estimated remaining: <strong>{estimate}</strong>
        </span>
        <span className="muted">
          Elapsed in this stage: {formatElapsed(stageElapsed)}
        </span>
        <span className="muted" style={{ fontSize: '0.76rem' }}>
          {basis}
        </span>
      </div>
    </div>
  )
}

type EditableMarkingRow = {
  marks: number
  criterion: string
}

function generatedMarkingRows(value: unknown): EditableMarkingRow[] {
  if (!Array.isArray(value)) return []

  return value.flatMap((item) => {
    if (typeof item === 'string') {
      const criterion = item.trim()
      return criterion ? [{ marks: 1, criterion }] : []
    }

    if (!item || typeof item !== 'object') return []

    const row = item as Record<string, unknown>
    const criterion = String(
      row.criterion ?? row.text ?? row.marking_point ?? row.guidance ?? ''
    ).trim()

    if (!criterion) return []

    const parsedMarks = Number(row.marks ?? row.mark ?? 1)
    const marks = Number.isFinite(parsedMarks) && parsedMarks > 0
      ? Math.max(1, Math.floor(parsedMarks))
      : 1

    return [{ marks, criterion }]
  })
}

function markingRowsToEditor(value: unknown) {
  return generatedMarkingRows(value)
    .map((row) => `${row.marks} | ${row.criterion}`)
    .join('\n')
}

function parseMarkingEditor(value: string): EditableMarkingRow[] {
  return value
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const match = line.match(/^(\d+)\s*(?:\||:|-)\s*(.+)$/)
      if (!match) return { marks: 1, criterion: line }
      return {
        marks: Math.max(1, Number(match[1]) || 1),
        criterion: match[2].trim(),
      }
    })
    .filter((row) => row.criterion.length > 0)
}

function HitlModal({
  title,
  description,
  children,
  onClose,
  footer,
}: {
  title: string
  description?: string
  children: React.ReactNode
  onClose: () => void
  footer: React.ReactNode
}) {
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={title}
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 1000,
        display: 'grid',
        placeItems: 'center',
        padding: '1rem',
        background: 'rgba(19, 11, 42, 0.72)',
        backdropFilter: 'blur(8px)',
      }}
      onMouseDown={(event) => {
        if (event.currentTarget === event.target) onClose()
      }}
    >
      <div
        style={{
          width: 'min(680px, 96vw)',
          maxHeight: '86vh',
          overflow: 'auto',
          borderRadius: '18px',
          border: '1px solid rgba(255,255,255,0.2)',
          background: 'rgba(83, 65, 145, 0.98)',
          boxShadow: '0 24px 70px rgba(0,0,0,0.32)',
          padding: '1rem',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'flex-start',
            justifyContent: 'space-between',
            gap: '1rem',
            marginBottom: '0.9rem',
          }}
        >
          <div>
            <p className="eyebrow" style={{ marginBottom: '0.25rem' }}>QUESTION REVIEW</p>
            <h2 style={{ margin: 0 }}>{title}</h2>
            {description && (
              <p className="muted" style={{ margin: '0.35rem 0 0' }}>{description}</p>
            )}
          </div>
          <button
            type="button"
            aria-label="Close"
            onClick={onClose}
            style={{
              display: 'grid',
              placeItems: 'center',
              width: '34px',
              height: '34px',
              borderRadius: '999px',
              border: '1px solid rgba(255,255,255,0.18)',
              background: 'rgba(255,255,255,0.08)',
              color: 'inherit',
              cursor: 'pointer',
            }}
          >
            <X size={17} />
          </button>
        </div>

        {children}

        <div
          style={{
            display: 'flex',
            justifyContent: 'flex-end',
            flexWrap: 'wrap',
            gap: '0.55rem',
            marginTop: '1rem',
          }}
        >
          {footer}
        </div>
      </div>
    </div>
  )
}

function GeneratedQuestionHITL({
  runId,
  question,
  planIndex,
  quizMode,
  onUpdated,
}: {
  runId: string
  question: AssessmentQuestion
  planIndex: number
  quizMode: 'complete_quiz' | 'fill_shortfall'
  onUpdated: (status: AssessmentStatusResponse) => void
}) {
  type ModalKind = 'question' | 'marking' | 'regenerate' | 'reject' | null
  type ReviewState = 'review' | 'approved' | 'rejected'

  const questionId = String(
    question.question_id || question.generated_question_id || ''
  ).trim()
  const storageKey = `edtech-question-review:${runId}:${quizMode}:${questionId || planIndex}`

  const [modal, setModal] = useState<ModalKind>(null)
  const [busyAction, setBusyAction] = useState<string | null>(null)
  const [reviewState, setReviewState] = useState<ReviewState>('review')
  const [questionDraft, setQuestionDraft] = useState(question.question_text || '')
  const [markingDraft, setMarkingDraft] = useState(markingRowsToEditor(question.marking_guidance))
  const [reason, setReason] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [actionStartedAt, setActionStartedAt] = useState<number | null>(null)
  const [actionNow, setActionNow] = useState(() => Date.now())
  const [regenerationEta, setRegenerationEta] = useState<Agent2EtaResponse | null>(null)
  const [regenerationEtaLoading, setRegenerationEtaLoading] = useState(false)

  useEffect(() => {
    setQuestionDraft(question.question_text || '')
    setMarkingDraft(markingRowsToEditor(question.marking_guidance))
  }, [question.question_text, question.marking_guidance])

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(storageKey)
      if (stored === 'approved' || stored === 'rejected' || stored === 'review') {
        setReviewState(stored)
      }
    } catch {
      // Backend persistence is authoritative; local storage is only a badge cache.
    }
  }, [storageKey])

  useEffect(() => {
    if (!busyAction || !actionStartedAt) return
    const timer = window.setInterval(() => setActionNow(Date.now()), 1000)
    return () => window.clearInterval(timer)
  }, [busyAction, actionStartedAt])

  useEffect(() => {
    if (modal !== 'regenerate') return

    let cancelled = false
    setRegenerationEtaLoading(true)

    getAgent2Eta(runId, 'question_regeneration')
      .then((value) => {
        if (!cancelled) setRegenerationEta(value)
      })
      .catch(() => {
        if (!cancelled) setRegenerationEta(null)
      })
      .finally(() => {
        if (!cancelled) setRegenerationEtaLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [modal, runId])

  const persistReviewState = (value: ReviewState) => {
    setReviewState(value)
    try {
      window.localStorage.setItem(storageKey, value)
    } catch {
      // Ignore browser cache failures.
    }
  }

  const closeModal = () => {
    if (busyAction) return
    setModal(null)
    setReason('')
    setError(null)
  }

  const performAction = async (
    action: 'approve' | 'edit_question' | 'edit_marking_guidance' | 'regenerate' | 'reject',
    extra: Record<string, unknown> = {}
  ) => {
    if (busyAction) return

    if (!questionId) {
      setError('This generated question has no question ID, so the review cannot be saved safely.')
      return
    }

    if (action !== 'approve' && !reason.trim()) {
      setError('Please add a short reason before saving this review action.')
      return
    }

    if (action === 'edit_question' && !questionDraft.trim()) {
      setError('Question text cannot be empty.')
      return
    }

    if (action === 'edit_marking_guidance') {
      const rows = parseMarkingEditor(markingDraft)
      if (rows.length === 0) {
        setError('Add at least one marking criterion.')
        return
      }
      extra.marking_guidance = rows
    }

    setBusyAction(action)
    setActionStartedAt(Date.now())
    setActionNow(Date.now())
    setError(null)
    setSuccess(null)

    try {
      const updated = await submitGeneratedQuestionReview(runId, {
        quiz_mode: quizMode,
        question_id: questionId,
        plan_index: planIndex,
        action,
        reason: reason.trim(),
        ...extra,
      })

      onUpdated(updated)

      if (action === 'approve') {
        persistReviewState('approved')
        setSuccess('Question approved successfully.')
      } else if (action === 'reject') {
        persistReviewState('rejected')
        setSuccess('Question rejected.')
      } else if (action === 'edit_question') {
        persistReviewState('review')
        setSuccess('Question updated successfully.')
      } else if (action === 'edit_marking_guidance') {
        persistReviewState('review')
        setSuccess('Marking scheme updated successfully.')
      } else {
        persistReviewState('review')
        setSuccess('Question regenerated and revalidated successfully.')
      }

      setModal(null)
      setReason('')
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Question review action failed. Please try again.'
      )
    } finally {
      setBusyAction(null)
      setActionStartedAt(null)
    }
  }

  const elapsedActionSeconds = actionStartedAt
    ? Math.max(0, Math.floor((actionNow - actionStartedAt) / 1000))
    : 0

  const regenerationRemainingLabel = formatHistoricalEtaRange(
    regenerationEta?.eta_total_low_seconds,
    regenerationEta?.eta_total_high_seconds,
    elapsedActionSeconds
  )

  const badge = reviewState === 'approved'
    ? <Badge tone="success">Approved</Badge>
    : reviewState === 'rejected'
      ? <Badge tone="warning">Rejected</Badge>
      : <Badge tone="warning">Review available</Badge>

  const actionButton = (
    label: string,
    action: string,
    onClick: () => void,
    variant: string = 'secondary'
  ) => (
    <Button
      key={action}
      variant={variant}
      disabled={Boolean(busyAction)}
      onClick={onClick}
    >
      {busyAction === action && (
        <span
          className="spinner"
          aria-hidden="true"
          style={{ width: '13px', height: '13px', margin: '0 0.38rem 0 0', flex: '0 0 auto' }}
        />
      )}
      {busyAction === action
        ? action === 'approve' ? 'Approving...'
          : action === 'edit_question' ? 'Saving...'
            : action === 'edit_marking_guidance' ? 'Saving...'
              : action === 'regenerate' ? 'Regenerating...'
                : 'Rejecting...'
        : label}
    </Button>
  )

  const textareaStyle: React.CSSProperties = {
    width: '100%',
    minHeight: '100px',
    resize: 'vertical',
    borderRadius: '12px',
    border: '1px solid rgba(255,255,255,0.24)',
    background: 'rgba(255,255,255,0.1)',
    color: 'inherit',
    padding: '0.8rem',
    font: 'inherit',
  }

  return (
    <>
      <section
        style={{
          marginTop: '1rem',
          overflow: 'hidden',
          borderRadius: '18px',
          border: '1px solid rgba(255,255,255,0.2)',
          background: 'linear-gradient(135deg, rgba(202,184,245,0.16), rgba(255,255,255,0.055))',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '1rem',
            padding: '0.9rem 1rem',
            borderBottom: '1px solid rgba(255,255,255,0.12)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div
              style={{
                width: '36px',
                height: '36px',
                borderRadius: '11px',
                display: 'grid',
                placeItems: 'center',
                background: 'rgba(255,255,255,0.12)',
                border: '1px solid rgba(255,255,255,0.17)',
              }}
            >
              <Check size={17} />
            </div>
            <div>
              <p className="eyebrow" style={{ margin: 0, opacity: 0.8 }}>HUMAN-IN-THE-LOOP</p>
              <strong style={{ display: 'block', marginTop: '0.18rem' }}>Question-level review</strong>
            </div>
          </div>
          {badge}
        </div>

        <div style={{ padding: '0.9rem 1rem 1rem' }}>
          <p className="muted" style={{ margin: '0 0 0.75rem', lineHeight: 1.55 }}>
            Review this question independently before final quiz approval.
            Notebook 06 supports question-level corrections without regenerating the complete quiz.
          </p>

          {busyAction === 'regenerate' && (
            <div
              style={{
                marginBottom: '0.75rem',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.5rem',
                padding: '0.55rem 0.7rem',
                borderRadius: '10px',
                background: 'rgba(255,255,255,0.085)',
                border: '1px solid rgba(255,255,255,0.15)',
              }}
            >
              <Clock3 size={15} />
              <div style={{ display: 'grid', gap: '0.15rem' }}>
                <span>
                  Regenerating this question only · Estimated remaining:{' '}
                  <strong>
                    {regenerationRemainingLabel
                      || regenerationEta?.eta_label
                      || 'learning from recent Agent 2 runs…'}
                  </strong>
                  {' '}· Elapsed {formatElapsed(elapsedActionSeconds)}
                </span>
                {regenerationEta?.eta_basis && (
                  <span className="muted" style={{ fontSize: '0.74rem' }}>
                    {regenerationEta.eta_basis}
                  </span>
                )}
              </div>
            </div>
          )}

          {success && (
            <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '0.75rem' }}>
              <div
                style={{
                  width: 'fit-content',
                  maxWidth: '100%',
                  padding: '0.55rem 0.8rem',
                  borderRadius: '10px',
                  background: 'rgba(104, 211, 145, 0.22)',
                  border: '1px solid rgba(167, 243, 208, 0.36)',
                  color: '#fff',
                  fontWeight: 700,
                }}
              >
                {success}
              </div>
            </div>
          )}

          {error && !modal && (
            <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '0.75rem' }}>
              <div
                style={{
                  width: 'fit-content',
                  maxWidth: '100%',
                  padding: '0.55rem 0.8rem',
                  borderRadius: '10px',
                  background: 'rgba(239, 68, 68, 0.2)',
                  border: '1px solid rgba(254, 202, 202, 0.32)',
                  color: '#fff',
                  fontWeight: 700,
                }}
              >
                {error}
              </div>
            </div>
          )}

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
            {actionButton('Approve', 'approve', () => performAction('approve'))}
            {actionButton('Edit Question', 'edit_question', () => {
              setError(null)
              setSuccess(null)
              setQuestionDraft(question.question_text || '')
              setReason('')
              setModal('question')
            })}
            {actionButton('Edit Marking Scheme', 'edit_marking_guidance', () => {
              setError(null)
              setSuccess(null)
              setMarkingDraft(markingRowsToEditor(question.marking_guidance))
              setReason('')
              setModal('marking')
            })}
            {actionButton('Regenerate', 'regenerate', () => {
              setError(null)
              setSuccess(null)
              setReason('')
              setModal('regenerate')
            })}
            {actionButton('Reject', 'reject', () => {
              setError(null)
              setSuccess(null)
              setReason('')
              setModal('reject')
            }, 'danger')}
          </div>
        </div>
      </section>

      {modal === 'question' && (
        <HitlModal
          title="Edit question"
          description="Only this question will be corrected and revalidated."
          onClose={closeModal}
          footer={
            <>
              <Button variant="secondary" disabled={Boolean(busyAction)} onClick={closeModal}>Cancel</Button>
              <Button
                disabled={Boolean(busyAction)}
                onClick={() => performAction('edit_question', { question_text: questionDraft.trim() })}
              >
                {busyAction === 'edit_question' ? 'Saving...' : 'Save Changes'}
              </Button>
            </>
          }
        >
          <label style={{ display: 'grid', gap: '0.42rem' }}>
            <strong>Question</strong>
            <textarea
              value={questionDraft}
              onChange={(event) => setQuestionDraft(event.target.value)}
              style={{ ...textareaStyle, minHeight: '180px' }}
            />
          </label>
          <label style={{ display: 'grid', gap: '0.42rem', marginTop: '0.8rem' }}>
            <strong>Reason for correction</strong>
            <textarea
              value={reason}
              onChange={(event) => setReason(event.target.value)}
              placeholder="Briefly explain why this question needs changing."
              style={textareaStyle}
            />
          </label>
          {error && <p style={{ margin: '0.65rem 0 0' }}><strong>{error}</strong></p>}
        </HitlModal>
      )}

      {modal === 'marking' && (
        <HitlModal
          title="Edit marking scheme"
          description="Use one criterion per line in the format: marks | criterion."
          onClose={closeModal}
          footer={
            <>
              <Button variant="secondary" disabled={Boolean(busyAction)} onClick={closeModal}>Cancel</Button>
              <Button disabled={Boolean(busyAction)} onClick={() => performAction('edit_marking_guidance')}>
                {busyAction === 'edit_marking_guidance' ? 'Saving...' : 'Save Changes'}
              </Button>
            </>
          }
        >
          <label style={{ display: 'grid', gap: '0.42rem' }}>
            <strong>Marking Scheme</strong>
            <textarea
              value={markingDraft}
              onChange={(event) => setMarkingDraft(event.target.value)}
              placeholder={'1 | First marking point\n1 | Second marking point'}
              style={{ ...textareaStyle, minHeight: '200px' }}
            />
          </label>
          <label style={{ display: 'grid', gap: '0.42rem', marginTop: '0.8rem' }}>
            <strong>Reason for correction</strong>
            <textarea
              value={reason}
              onChange={(event) => setReason(event.target.value)}
              placeholder="Briefly explain what is wrong with the current marking scheme."
              style={textareaStyle}
            />
          </label>
          {error && <p style={{ margin: '0.65rem 0 0' }}><strong>{error}</strong></p>}
        </HitlModal>
      )}

      {modal === 'regenerate' && (
        <HitlModal
          title="Regenerate this question?"
          description={
            regenerationEtaLoading
              ? 'Only this question will be regenerated. Calculating ETA from recent Agent 2 runs…'
              : regenerationEta?.eta_total_label
                ? `Only this question will be regenerated. Historical ETA: ${regenerationEta.eta_total_label}.`
                : 'Only this question will be regenerated. ETA will improve as Agent 2 timing history builds.'
          }
          onClose={closeModal}
          footer={
            <>
              <Button variant="secondary" disabled={Boolean(busyAction)} onClick={closeModal}>Cancel</Button>
              <Button disabled={Boolean(busyAction)} onClick={() => performAction('regenerate')}>
                {busyAction === 'regenerate' ? 'Regenerating...' : 'Regenerate Question'}
              </Button>
            </>
          }
        >
          <label style={{ display: 'grid', gap: '0.42rem' }}>
            <strong>Why should this question be regenerated?</strong>
            <textarea
              value={reason}
              onChange={(event) => setReason(event.target.value)}
              placeholder="Example: the question is ambiguous or the marking scheme does not match."
              style={textareaStyle}
            />
          </label>
          {regenerationEta?.eta_basis && (
            <div
              style={{
                marginTop: '0.7rem',
                padding: '0.62rem 0.72rem',
                borderRadius: '10px',
                background: 'rgba(255,255,255,0.07)',
                border: '1px solid rgba(255,255,255,0.12)',
              }}
            >
              <Clock3 size={14} style={{ marginRight: '0.4rem', verticalAlign: '-2px' }} />
              <span className="muted">{regenerationEta.eta_basis}</span>
            </div>
          )}
          {error && <p style={{ margin: '0.65rem 0 0' }}><strong>{error}</strong></p>}
        </HitlModal>
      )}

      {modal === 'reject' && (
        <HitlModal
          title="Reject this question?"
          description="The rejection will be stored as question-level HITL feedback."
          onClose={closeModal}
          footer={
            <>
              <Button variant="secondary" disabled={Boolean(busyAction)} onClick={closeModal}>Cancel</Button>
              <Button variant="danger" disabled={Boolean(busyAction)} onClick={() => performAction('reject')}>
                {busyAction === 'reject' ? 'Rejecting...' : 'Reject Question'}
              </Button>
            </>
          }
        >
          <label style={{ display: 'grid', gap: '0.42rem' }}>
            <strong>Reason for rejection</strong>
            <textarea
              value={reason}
              onChange={(event) => setReason(event.target.value)}
              placeholder="Explain why this question should not be used."
              style={textareaStyle}
            />
          </label>
          {error && <p style={{ margin: '0.65rem 0 0' }}><strong>{error}</strong></p>}
        </HitlModal>
      )}
    </>
  )
}


function AssessmentQuestionCard({
  runId,
  question,
  position,
  official,
  planIndex,
  quizMode,
  onStatusUpdated,
}: {
  runId: string
  question: AssessmentQuestion
  position: number
  official: boolean
  planIndex?: number
  quizMode?: 'complete_quiz' | 'fill_shortfall' | string
  onStatusUpdated: (status: AssessmentStatusResponse) => void
}) {
  const marking = markingGuidanceLines(question.marking_guidance)

  return (
    <Card>
      <div
        style={{
          display: 'grid',
          gap: '1rem',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'flex-start',
            justifyContent: 'space-between',
            gap: '1rem',
            paddingBottom: '1rem',
            borderBottom: '1px solid rgba(255,255,255,0.16)',
          }}
        >
          <div
            style={{
              display: 'flex',
              alignItems: 'flex-start',
              gap: '0.9rem',
              minWidth: 0,
            }}
          >
            <div
              style={{
                width: '46px',
                height: '46px',
                flex: '0 0 auto',
                display: 'grid',
                placeItems: 'center',
                borderRadius: '14px',
                background: 'rgba(255,255,255,0.13)',
                border: '1px solid rgba(255,255,255,0.2)',
                fontWeight: 850,
              }}
            >
              Q{position}
            </div>

            <div>
              <p className="eyebrow" style={{ marginBottom: '0.28rem' }}>
                {official
                  ? 'OFFICIAL AQA RETRIEVAL'
                  : 'AI-GENERATED AQA-ALIGNED'}
              </p>

              <h2 style={{ margin: 0 }}>
                {question.topic || 'Assessment question'}
              </h2>

              <div
                style={{
                  display: 'flex',
                  flexWrap: 'wrap',
                  alignItems: 'center',
                  gap: '0.45rem',
                  marginTop: '0.55rem',
                }}
              >
                {question.official_reference && (
                  <span
                    style={{
                      padding: '0.34rem 0.55rem',
                      borderRadius: '999px',
                      background: 'rgba(255,255,255,0.1)',
                      border: '1px solid rgba(255,255,255,0.16)',
                      fontSize: '0.8rem',
                    }}
                  >
                    AQA {question.official_reference}
                  </span>
                )}

                {question.paper && (
                  <span
                    style={{
                      padding: '0.34rem 0.55rem',
                      borderRadius: '999px',
                      background: 'rgba(255,255,255,0.1)',
                      border: '1px solid rgba(255,255,255,0.16)',
                      fontSize: '0.8rem',
                    }}
                  >
                    {question.paper}
                  </span>
                )}

                {question.role && (
                  <Badge
                    tone={
                      String(question.role).toLowerCase() === 'primary'
                        ? 'teal'
                        : 'warning'
                    }
                  >
                    {question.role}
                  </Badge>
                )}
              </div>
            </div>
          </div>

          <Badge tone={official ? 'teal' : 'info'}>
            {question.marks || 0} marks
          </Badge>
        </div>

        {question.context && (
          <section
            style={{
              padding: '0.9rem 1rem',
              borderRadius: '14px',
              background: 'rgba(255,255,255,0.065)',
              border: '1px solid rgba(255,255,255,0.12)',
            }}
          >
            <p className="eyebrow" style={{ marginBottom: '0.4rem' }}>
              CONTEXT
            </p>
            <p style={{ margin: 0, whiteSpace: 'pre-wrap', lineHeight: 1.65 }}>
              {question.context}
            </p>
          </section>
        )}

        <section
          style={{
            padding: '1rem 1.05rem',
            borderRadius: '16px',
            background: 'rgba(35,28,50,0.28)',
            border: '1px solid rgba(255,255,255,0.14)',
          }}
        >
          <p className="eyebrow" style={{ marginBottom: '0.5rem' }}>
            QUESTION
          </p>

          <p
            style={{
              margin: 0,
              whiteSpace: 'pre-wrap',
              lineHeight: 1.72,
              fontSize: '1rem',
            }}
          >
            {question.question_text || 'Question text unavailable.'}
          </p>
        </section>

        {/* Requested order: question first, then visual if available. */}
        <QuestionVisuals
          runId={runId}
          question={question}
        />

        {marking.length > 0 && (
          <section
            style={{
              borderRadius: '18px',
              border: '1px solid rgba(255,255,255,0.17)',
              overflow: 'hidden',
              background: 'rgba(255,255,255,0.055)',
            }}
          >
            <div
              style={{
                padding: '0.9rem 1rem',
                borderBottom: '1px solid rgba(255,255,255,0.12)',
                background: 'rgba(255,255,255,0.05)',
              }}
            >
              <p className="eyebrow" style={{ margin: 0, opacity: 0.8 }}>
                MARKING SCHEME
              </p>
              <strong
                style={{
                  display: 'block',
                  marginTop: '0.18rem',
                }}
              >
                Marking guidance
              </strong>
            </div>

            <div
              style={{
                display: 'grid',
                gap: '0.55rem',
                padding: '0.9rem 1rem 1rem',
              }}
            >
              {marking.map((line, index) => (
                <div
                  key={`${question.question_id}-mark-${index}`}
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '36px 1fr',
                    gap: '0.7rem',
                    alignItems: 'start',
                    padding: '0.72rem 0.78rem',
                    borderRadius: '12px',
                    background: 'rgba(255,255,255,0.095)',
                    border: '1px solid rgba(255,255,255,0.08)',
                    whiteSpace: 'pre-wrap',
                  }}
                >
                  <span
                    style={{
                      width: '30px',
                      height: '30px',
                      display: 'grid',
                      placeItems: 'center',
                      borderRadius: '9px',
                      background: 'rgba(255,255,255,0.11)',
                      fontSize: '0.76rem',
                      fontWeight: 800,
                    }}
                  >
                    {String(index + 1).padStart(2, '0')}
                  </span>

                  <span style={{ lineHeight: 1.55 }}>
                    {line}
                  </span>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Official retrieval HITL stays fully functional. */}
        {official ? (
          <RetrievalHITL
            runId={runId}
            question={question}
            onUpdated={onStatusUpdated}
          />
        ) : (
          <GeneratedQuestionHITL
            runId={runId}
            question={question}
            planIndex={Number(planIndex || question.plan_index || 1)}
            quizMode={quizMode === 'fill_shortfall' ? 'fill_shortfall' : 'complete_quiz'}
            onUpdated={onStatusUpdated}
          />
        )}
      </div>
    </Card>
  )
}

function GeneratedQuizHITL({
  runId,
  status,
  onUpdated,
}: {
  runId: string
  status: AssessmentStatusResponse
  onUpdated: (status: AssessmentStatusResponse) => void
}) {
  const generated = status.generated
  const [reason, setReason] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  if (!generated) return null

  const reviewState = String(generated.human_review_state || '').toUpperCase()
  const hasCandidates = generated.candidate_questions.length > 0
  const needsWholeQuizReview =
    hasCandidates
    && !generated.generated_quality_accepted
    && !reviewState.includes('QUESTION_LEVEL')

  if (!needsWholeQuizReview && !reviewState.includes('QUESTION_LEVEL')) {
    if (generated.generated_quality_accepted || generated.release_ready) {
      return (
        <div className="success-banner">
          <CheckCircle2 size={19} />
          <div>
            <strong>Generated assessment passed human/quality gates</strong>
            <span>The accepted version is ready for the final PDF.</span>
          </div>
        </div>
      )
    }
    return null
  }

  if (reviewState.includes('QUESTION_LEVEL')) {
    return (
      <div className="guidance">
        <Info size={18} />
        <div>
          <strong>Question-level HITL is active in Notebook 06</strong>
          <p>
            The notebook is waiting for per-question actions. The whole-quiz Approve / Regenerate / Reject gate below is intentionally not used for this state.
          </p>
        </div>
      </div>
    )
  }

  const submit = async (decision: 'approve' | 'regenerate' | 'reject') => {
    if (!reason.trim()) {
      setError('Add a written review reason before submitting the HITL decision.')
      return
    }
    setSaving(true)
    setError(null)
    try {
      const updated = await submitQuizReview(runId, {
        quiz_mode: generated.quiz_mode === 'fill_shortfall'
          ? 'fill_shortfall'
          : 'complete_quiz',
        decision,
        reason: reason.trim(),
      })
      onUpdated(updated)
      setReason('')
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Could not save generated-quiz HITL decision.'
      )
    } finally {
      setSaving(false)
    }
  }

  return (
    <Card>
      <div className="card-heading">
        <div>
          <p className="eyebrow">GENERATED QUIZ HUMAN GATE</p>
          <h2>Review AI-generated assessment</h2>
        </div>
        <Badge tone="warning">Human review required</Badge>
      </div>
      <p className="muted">
        Approve the candidate set, regenerate it with your reason, or reject it. The decision is persisted through the existing Agent 2 HITL workflow.
      </p>
      <textarea
        aria-label="Quiz review reason"
        value={reason}
        onChange={(event) => setReason(event.target.value)}
        placeholder="Required: explain your approval, regeneration request, or rejection."
        style={{
          width: '100%',
          minHeight: '95px',
          marginTop: '0.9rem',
          resize: 'vertical',
          borderRadius: '10px',
          border: '1px solid rgba(255,255,255,0.24)',
          background: 'rgba(255,255,255,0.12)',
          color: 'inherit',
          padding: '0.75rem',
          font: 'inherit',
        }}
      />
      {error && (
        <div className="guidance" style={{ marginTop: '0.8rem' }}>
          <XCircle size={17} />
          <div><strong>Review could not be saved</strong><p>{error}</p></div>
        </div>
      )}
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '0.65rem',
          marginTop: '0.9rem',
        }}
      >
        <Button disabled={saving} onClick={() => submit('approve')}>
          Approve Quiz <Check size={16} />
        </Button>
        <Button disabled={saving} variant="secondary" onClick={() => submit('regenerate')}>
          Regenerate
        </Button>
        <Button disabled={saving} variant="danger" onClick={() => submit('reject')}>
          Reject
        </Button>
      </div>
    </Card>
  )
}

function Assessment({ runId }: { runId: string }) {
  const [config, setConfig] = useState<AssessmentConfigResponse | null>(null)
  const [status, setStatus] = useState<AssessmentStatusResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [mode, setMode] = useState<AssessmentMode>('retrieve_hybrid')
  const [paper, setPaper] = useState<'Any' | 'Paper 1' | 'Paper 2'>('Any')
  const [numberOfQuestions, setNumberOfQuestions] = useState(5)
  const [targetMarks, setTargetMarks] = useState(20)
  const [minimumMarks, setMinimumMarks] = useState(1)
  const [maximumMarks, setMaximumMarks] = useState(12)
  const [minimumPrimary, setMinimumPrimary] = useState(1)
  const [minimumSupporting, setMinimumSupporting] = useState(0)
  const [coverAll, setCoverAll] = useState(true)
  const [includeCode, setIncludeCode] = useState(true)
  const [includeVisuals, setIncludeVisuals] = useState(true)
  const [programmingLanguage, setProgrammingLanguage] = useState<'Automatic' | 'Python'>('Automatic')
  const [modelKey, setModelKey] = useState('')
  const [quizPlan, setQuizPlan] = useState<'plan_a' | 'plan_b' | 'plan_c'>('plan_c')
  const [instructions, setInstructions] = useState('')

  const loadStatus = async () => {
    if (!runId) return
    const next = await getAssessmentStatus(runId)
    setStatus(next)
    return next
  }

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      if (!runId) {
        setError('No active lesson run is available. Approve Agent 1 topics first.')
        setLoading(false)
        return
      }
      setLoading(true)
      setError(null)
      try {
        const [nextConfig, nextStatus] = await Promise.all([
          getAssessmentConfig(runId),
          getAssessmentStatus(runId),
        ])
        if (cancelled) return
        setConfig(nextConfig)
        setStatus(nextStatus)
        setNumberOfQuestions(Math.max(5, nextConfig.topic_count || 1))
        setTargetMarks(Math.max(20, (nextConfig.topic_count || 1) * 2))
        setMinimumPrimary(nextConfig.primary_topic_count > 0 ? 1 : 0)
        setMinimumSupporting(nextConfig.supporting_topic_count > 0 ? 1 : 0)
        const planC = nextConfig.notebook_options.find(
          (item) => item.key === 'plan_c' && item.available
        )
        const firstAvailable = nextConfig.notebook_options.find((item) => item.available)
        const selectedPlan = String(planC?.key || firstAvailable?.key || 'plan_c')
        if (selectedPlan === 'plan_a' || selectedPlan === 'plan_b' || selectedPlan === 'plan_c') {
          setQuizPlan(selectedPlan)
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Could not load Agent 2.')
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    load()
    return () => { cancelled = true }
  }, [runId])

  useEffect(() => {
    if (!runId || !status) return
    const live = ['queued', 'running'].includes(String(status.status).toLowerCase())
    if (!live) {
      setRunning(false)
      return
    }
    setRunning(true)
    const timer = window.setInterval(async () => {
      try {
        const next = await getAssessmentStatus(runId)
        setStatus(next)
        if (!['queued', 'running'].includes(String(next.status).toLowerCase())) {
          window.clearInterval(timer)
          setRunning(false)
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Could not refresh assessment status.')
        window.clearInterval(timer)
        setRunning(false)
      }
    }, 1400)
    return () => window.clearInterval(timer)
  }, [runId, status?.status])

  const updateStatus = (next: AssessmentStatusResponse) => {
    setStatus(next)
  }

  const runAssessment = async () => {
    if (!config) return
    if (!modelKey) {
      setError('Select a generation model. It is required because retrieval may need AI fallback.')
      return
    }
    if (maximumMarks < minimumMarks) {
      setError('Maximum marks per question must be at least minimum marks.')
      return
    }
    if (coverAll && numberOfQuestions < config.topic_count) {
      setError(`Use at least ${config.topic_count} questions to cover all approved topics.`)
      return
    }

    const input: AssessmentStartInput = {
      mode,
      paper,
      number_of_questions: numberOfQuestions,
      target_total_marks: targetMarks,
      minimum_question_marks: minimumMarks,
      maximum_question_marks: maximumMarks,
      minimum_primary_questions: minimumPrimary,
      minimum_supporting_questions: minimumSupporting,
      cover_all_approved_topics: coverAll,
      include_code_questions: includeCode,
      include_visual_questions: includeVisuals,
      programming_language: programmingLanguage,
      model_key: modelKey,
      quiz_plan: quizPlan,
      special_instructions: instructions.trim(),
    }

    // Start each Agent 2 attempt with a clean result area.
    // Previous retrieval / generated output must not remain visible
    // while the new assessment is running.
    setStatus(null)
    setRunning(true)
    setError(null)
    try {
      const next = await startAssessment(runId, input)
      setStatus(next)
    } catch (err) {
      setRunning(false)
      setError(err instanceof Error ? err.message : 'Could not start Agent 2.')
    }
  }

  const assessmentState = String(status?.status || '').toLowerCase()
  const resultsReady = assessmentState === 'complete'
  const currentResultMode = status?.mode
  const hybridFallbackFailed =
    assessmentState === 'failed'
    && currentResultMode === 'retrieve_hybrid'
    && Boolean(status?.shortfall)

  // Keep hybrid fallback failures user-facing. Backend status/error values are
  // still preserved for diagnostics, but the assessment page should explain
  // what happened in normal language rather than exposing internal state names.
  const workflowDisplayTitle = hybridFallbackFailed
    ? 'Assessment incomplete'
    : (status?.message || status?.stage || 'Assessment status')

  const workflowDisplayBadge = hybridFallbackFailed
    ? 'incomplete'
    : String(status?.status || '').replaceAll('_', ' ')

  const workflowDisplayStage = hybridFallbackFailed
    ? 'Official questions retrieved · AI fallback incomplete'
    : String(status?.stage || '').replaceAll('_', ' ')

  // Render only CURRENT-attempt results. If official retrieval succeeded but
  // the AI fallback failed, keep the retrieved official questions visible.
  const officialQuestions =
    currentResultMode === 'retrieve_hybrid'
    && (resultsReady || hybridFallbackFailed)
      ? status?.official?.questions || []
      : []

  const shouldShowGenerated =
    resultsReady
    && Boolean(status?.generated)
    && (
      currentResultMode === 'complete_quiz'
      || (
        currentResultMode === 'retrieve_hybrid'
        && status?.shortfall?.sufficient === false
      )
    )

  const generatedQuestions =
    shouldShowGenerated && status?.generated
      ? (
          status.generated.accepted_questions.length > 0
            ? status.generated.accepted_questions
            : status.generated.candidate_questions
        )
      : []

  const generatedShortfallMarks = generatedQuestions.reduce(
    (total, question) => total + Math.max(0, Number(question.marks || 0)),
    0
  )
  const generatedShortfallCount = generatedQuestions.length
  const requestedQuestionCount = Number(
    status?.shortfall?.requested_questions
    || (
      Number(status?.shortfall?.selected_questions || 0)
      + Number(status?.shortfall?.missing_questions || 0)
    )
    || 0
  )
  const requestedMarkTotal = Number(
    status?.shortfall?.target_marks || 0
  )
  const officialQuestionCount = Number(
    status?.shortfall?.selected_questions || 0
  )
  const officialMarkTotal = Number(
    status?.shortfall?.selected_marks || 0
  )
  const finalQuestionCount = officialQuestionCount + generatedShortfallCount
  const finalMarkTotal = officialMarkTotal + generatedShortfallMarks
  const remainingQuestionCount = Math.max(0, requestedQuestionCount - finalQuestionCount)
  const remainingMarkCount = Math.max(0, requestedMarkTotal - finalMarkTotal)
  const markOverTarget = Math.max(0, finalMarkTotal - requestedMarkTotal)
  const hybridShortfallFilled =
    resultsReady
    && currentResultMode === 'retrieve_hybrid'
    && status?.shortfall?.sufficient === false
    && generatedShortfallCount > 0
    && (requestedQuestionCount <= 0 || finalQuestionCount >= requestedQuestionCount)
    && (requestedMarkTotal <= 0 || finalMarkTotal >= requestedMarkTotal)

  const pdfPaths =
    resultsReady
      ? (
          status?.generated?.pdf_paths?.length
            ? status.generated.pdf_paths
            : status?.official?.pdf_paths || []
        )
      : []

  const primaryPdf = pdfPaths[0]
  const isBusy = running || ['queued', 'running'].includes(assessmentState)

  const fieldStyle = {
    width: '100%',
    borderRadius: '10px',
    border: '1px solid rgba(255,255,255,0.24)',
    background: 'rgba(255,255,255,0.12)',
    color: 'inherit',
    padding: '0.72rem 0.78rem',
    font: 'inherit',
  }

  return (
    <>
      <Header
        eyebrow="Agent 2 / Assessment intelligence"
        title="Build the Assessment"
        description="Choose the assessment requirements once. Agent 2 will either retrieve official questions with automatic AI shortfall generation, or generate a complete new quiz."
      />
      <Stepper current={4} />

      <style>{`
        .assessment-select {
          background: #8878b8 !important;
          color: #ffffff !important;
          color-scheme: dark;
        }

        .assessment-select option {
          background: #8878b8 !important;
          color: #ffffff !important;
        }

        .assessment-select option:checked,
        .assessment-select option:hover {
          background: #7564aa !important;
          color: #ffffff !important;
        }
      `}</style>

      {loading && (
        <Card className="processing-card">
          <div className="spinner" />
          <p className="eyebrow">LOADING AGENT 2</p>
          <h2>Preparing approved topics and assessment controls...</h2>
        </Card>
      )}

      {!loading && error && (
        <div className="guidance" style={{ marginBottom: '1rem' }}>
          <XCircle size={18} />
          <div><strong>Agent 2 needs attention</strong><p>{error}</p></div>
        </div>
      )}

      {!loading && config && (
        <div style={{ display: 'grid', gap: '1rem' }}>
          <Card>
            <div className="card-heading">
              <div>
                <p className="eyebrow">APPROVED AGENT 1 HANDOFF</p>
                <h2>{config.topic_count} topics ready for assessment</h2>
              </div>
              <Badge tone="success">Human approved</Badge>
            </div>
            <div
              style={{
                display: 'flex',
                flexWrap: 'wrap',
                gap: '0.55rem',
                marginTop: '0.85rem',
              }}
            >
              {config.approved_topics.map((topic, index) => (
                <span
                  key={`${topic.concept_id || topic.official_reference}-${index}`}
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '0.45rem',
                    padding: '0.48rem 0.7rem',
                    borderRadius: '999px',
                    background: 'rgba(255,255,255,0.12)',
                    border: '1px solid rgba(255,255,255,0.2)',
                    fontSize: '0.86rem',
                  }}
                >
                  <strong>{topicName(topic)}</strong>
                  <span className="muted">{topic.official_reference}</span>
                </span>
              ))}
            </div>
          </Card>

          <div className="mode-grid">
            <Card>
              <div className="mode-icon teal"><Search /></div>
              <p className="eyebrow">RECOMMENDED · HYBRID</p>
              <h2>Official Retrieval + AI Fallback</h2>
              <p>
                Retrieve quality-safe official AQA questions first. If the requested count, marks, or topic coverage is short, Agent 2 automatically generates only the missing questions.
              </p>
              <Button
                variant={mode === 'retrieve_hybrid' ? 'primary' : 'secondary'}
                onClick={() => setMode('retrieve_hybrid')}
                disabled={isBusy}
              >
                {mode === 'retrieve_hybrid' ? <Check size={16} /> : <Search size={16} />}
                Use Hybrid Retrieval
              </Button>
            </Card>
            <Card>
              <div className="mode-icon navy"><Sparkles /></div>
              <p className="eyebrow">NEW MATERIAL</p>
              <h2>Generate Complete Quiz</h2>
              <p>
                Skip past-paper retrieval and create the full assessment from the approved lesson evidence, using the selected Notebook 06 strategy and generation model.
              </p>
              <Button
                variant={mode === 'complete_quiz' ? 'primary' : 'secondary'}
                onClick={() => setMode('complete_quiz')}
                disabled={isBusy}
              >
                {mode === 'complete_quiz' ? <Check size={16} /> : <Sparkles size={16} />}
                Generate Complete Quiz
              </Button>
            </Card>
          </div>

          <Card>
            <div className="card-heading">
              <div>
                <p className="eyebrow">ASSESSMENT REQUIREMENTS</p>
                <h2>Configure the paper</h2>
              </div>
              <Badge tone="info">
                {mode === 'retrieve_hybrid' ? 'Official first' : 'AI generation'}
              </Badge>
            </div>

            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))',
                gap: '0.85rem',
                marginTop: '1rem',
              }}
            >
              <label style={{ display: 'grid', gap: '0.38rem' }}>
                <strong>Number of questions</strong>
                <input style={fieldStyle} type="number" min={1} max={30} value={numberOfQuestions} onChange={(e) => setNumberOfQuestions(Number(e.target.value))} disabled={isBusy} />
              </label>
              <label style={{ display: 'grid', gap: '0.38rem' }}>
                <strong>Target total marks</strong>
                <input style={fieldStyle} type="number" min={1} max={200} value={targetMarks} onChange={(e) => setTargetMarks(Number(e.target.value))} disabled={isBusy} />
              </label>
              <label style={{ display: 'grid', gap: '0.38rem' }}>
                <strong>Paper</strong>
                <select className="assessment-select" style={fieldStyle} value={paper} onChange={(e) => setPaper(e.target.value as 'Any' | 'Paper 1' | 'Paper 2')} disabled={isBusy}>
                  <option>Any</option><option>Paper 1</option><option>Paper 2</option>
                </select>
              </label>
              <label style={{ display: 'grid', gap: '0.38rem' }}>
                <strong>Programming language</strong>
                <select className="assessment-select" style={fieldStyle} value={programmingLanguage} onChange={(e) => setProgrammingLanguage(e.target.value as 'Automatic' | 'Python')} disabled={isBusy}>
                  <option>Automatic</option><option>Python</option>
                </select>
              </label>
              <label style={{ display: 'grid', gap: '0.38rem' }}>
                <strong>Minimum marks / question</strong>
                <input style={fieldStyle} type="number" min={1} max={50} value={minimumMarks} onChange={(e) => setMinimumMarks(Number(e.target.value))} disabled={isBusy} />
              </label>
              <label style={{ display: 'grid', gap: '0.38rem' }}>
                <strong>Maximum marks / question</strong>
                <input style={fieldStyle} type="number" min={1} max={50} value={maximumMarks} onChange={(e) => setMaximumMarks(Number(e.target.value))} disabled={isBusy} />
              </label>
              <label style={{ display: 'grid', gap: '0.38rem' }}>
                <strong>Minimum primary questions</strong>
                <input style={fieldStyle} type="number" min={0} max={30} value={minimumPrimary} onChange={(e) => setMinimumPrimary(Number(e.target.value))} disabled={isBusy} />
              </label>
              <label style={{ display: 'grid', gap: '0.38rem' }}>
                <strong>Minimum supporting questions</strong>
                <input style={fieldStyle} type="number" min={0} max={30} value={minimumSupporting} onChange={(e) => setMinimumSupporting(Number(e.target.value))} disabled={isBusy} />
              </label>
              <label style={{ display: 'grid', gap: '0.38rem' }}>
                <strong>Generation model</strong>
                <select className="assessment-select" style={fieldStyle} value={modelKey} onChange={(e) => setModelKey(e.target.value)} disabled={isBusy}>
                  <option value="">Select model...</option>
                  {config.models.map((model) => (
                    <option key={model.key} value={model.key}>
                      {model.display_name}{model.provider ? ` · ${model.provider}` : ''}
                    </option>
                  ))}
                </select>
              </label>
              <label style={{ display: 'grid', gap: '0.38rem' }}>
                <strong>Quiz generation strategy</strong>
                <select className="assessment-select" style={fieldStyle} value={quizPlan} onChange={(e) => setQuizPlan(e.target.value as 'plan_a' | 'plan_b' | 'plan_c')} disabled={isBusy}>
                  {config.notebook_options.map((option) => (
                    <option key={option.key} value={option.key} disabled={!option.available}>
                      {option.label}{!option.available ? ' — unavailable' : ''}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(230px, 1fr))',
                gap: '0.65rem',
                marginTop: '1rem',
              }}
            >
              {[
                ['Cover every approved topic', coverAll, setCoverAll],
                ['Allow code questions', includeCode, setIncludeCode],
                ['Allow diagram / visual questions', includeVisuals, setIncludeVisuals],
              ].map(([label, checked, setter]) => (
                <label
                  key={String(label)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.65rem',
                    padding: '0.72rem',
                    borderRadius: '11px',
                    background: 'rgba(255,255,255,0.09)',
                    border: '1px solid rgba(255,255,255,0.16)',
                    cursor: isBusy ? 'default' : 'pointer',
                  }}
                >
                  <input
                    type="checkbox"
                    checked={Boolean(checked)}
                    onChange={(e) => (setter as (value: boolean) => void)(e.target.checked)}
                    disabled={isBusy}
                  />
                  <strong>{String(label)}</strong>
                </label>
              ))}
            </div>

            <label style={{ display: 'grid', gap: '0.4rem', marginTop: '1rem' }}>
              <strong>Special instructions <span className="muted">(optional)</span></strong>
              <textarea
                value={instructions}
                onChange={(e) => setInstructions(e.target.value)}
                disabled={isBusy}
                placeholder="e.g. Prefer scenario-based questions, include one trace-table question, avoid networking content not taught in this lesson..."
                style={{ ...fieldStyle, minHeight: '92px', resize: 'vertical' }}
              />
            </label>

            <div className="card-footer" style={{ marginTop: '1.1rem' }}>
              <span className="muted">
                {mode === 'retrieve_hybrid'
                  ? 'Notebook 05 retrieval → question count first → marks completed next → Notebook 06 only if needed.'
                  : 'Notebook 06 creates the complete assessment from approved lesson evidence.'}
              </span>
              <Button disabled={isBusy} onClick={runAssessment}>
                {isBusy
                  ? 'Agent 2 Running...'
                  : mode === 'retrieve_hybrid'
                    ? 'Run Retrieval + Fallback'
                    : 'Generate Complete Quiz'}
                {!isBusy && <ArrowRight size={17} />}
              </Button>
            </div>
          </Card>

          {status && status.status !== 'idle' && (
            <Card className={isBusy ? 'processing-card' : ''}>
              <div className="card-heading">
                <div>
                  <p className="eyebrow">AGENT 2 WORKFLOW STATUS</p>
                  <h2>{workflowDisplayTitle}</h2>
                </div>
                <Badge tone={status.status === 'failed' ? 'warning' : status.status === 'complete' ? 'success' : 'info'}>
                  {workflowDisplayBadge}
                </Badge>
              </div>
              <div style={{ marginTop: '0.9rem' }}>
                <div style={{ height: '9px', borderRadius: '99px', overflow: 'hidden', background: 'rgba(255,255,255,0.16)' }}>
                  <div style={{ width: `${Math.max(0, Math.min(100, Number(status.progress || 0)))}%`, height: '100%', background: 'currentColor', opacity: 0.85 }} />
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.45rem' }}>
                  <span className="muted">
                    {workflowDisplayStage}
                  </span>
                  <strong>{status.progress || 0}%</strong>
                </div>
              </div>

              {isBusy && (
                <AssessmentTiming status={status} />
              )}

              {status.error && !hybridFallbackFailed && (
                <div className="guidance" style={{ marginTop: '0.85rem' }}>
                  <XCircle size={17} /><div><strong>Agent 2 could not complete this assessment</strong><p>{status.error}</p></div>
                </div>
              )}
            </Card>
          )}

          {status?.shortfall && status.mode === 'retrieve_hybrid' && ['complete', 'failed'].includes(assessmentState) && (
            <div
              className={
                status.shortfall.sufficient || hybridShortfallFilled
                  ? 'success-banner'
                  : 'guidance'
              }
            >
              {status.shortfall.sufficient || hybridShortfallFilled
                ? <CheckCircle2 size={19} />
                : <XCircle size={19} />}
              <div>
                <strong>
                  {status.shortfall.sufficient
                    ? 'Assessment completed with official AQA questions'
                    : hybridShortfallFilled
                      ? 'Assessment completed'
                      : 'Assessment incomplete'}
                </strong>
                <span>
                  {status.shortfall.sufficient
                    ? `EDTech found ${officialQuestionCount} suitable official AQA question${officialQuestionCount === 1 ? '' : 's'} worth ${officialMarkTotal} marks, so no AI generation was needed.`
                    : hybridShortfallFilled
                      ? markOverTarget > 0
                        ? `EDTech found ${officialQuestionCount} official AQA question${officialQuestionCount === 1 ? '' : 's'} worth ${officialMarkTotal} marks and generated ${generatedShortfallCount} additional question${generatedShortfallCount === 1 ? '' : 's'}. The final assessment contains ${finalQuestionCount} questions worth ${finalMarkTotal} marks. This is ${markOverTarget} mark${markOverTarget === 1 ? '' : 's'} above your ${requestedMarkTotal}-mark target because EDTech prioritised completing your requested ${requestedQuestionCount} questions first.`
                        : `EDTech found ${officialQuestionCount} official AQA question${officialQuestionCount === 1 ? '' : 's'} worth ${officialMarkTotal} marks and generated ${generatedShortfallCount} additional question${generatedShortfallCount === 1 ? '' : 's'}. The final assessment contains ${finalQuestionCount} questions worth ${finalMarkTotal} marks.`
                      : remainingQuestionCount > 0
                        ? `EDTech found ${officialQuestionCount} suitable official AQA question${officialQuestionCount === 1 ? '' : 's'} worth ${officialMarkTotal} marks, but your request was for ${requestedQuestionCount} questions. The remaining ${remainingQuestionCount} question${remainingQuestionCount === 1 ? '' : 's'} could not be generated, so the assessment is currently incomplete. Your ${officialQuestionCount} retrieved official question${officialQuestionCount === 1 ? ' is' : 's are'} still available below.`
                        : remainingMarkCount > 0
                          ? `EDTech reached your requested ${requestedQuestionCount} questions, but the assessment is still ${remainingMarkCount} mark${remainingMarkCount === 1 ? '' : 's'} below your ${requestedMarkTotal}-mark target. The remaining mark coverage could not be generated, so the assessment is currently incomplete. The completed questions are still available below.`
                          : 'EDTech could not complete the AI fallback for this assessment. Any successfully retrieved official AQA questions are still available below.'}
                </span>
                {!status.shortfall.sufficient && !hybridShortfallFilled && status.error && (
                  <details style={{ marginTop: '0.55rem', fontSize: '0.78rem' }}>
                    <summary style={{ cursor: 'pointer', fontWeight: 700, opacity: 0.82 }}>
                      Technical details
                    </summary>
                    <p className="muted" style={{ margin: '0.35rem 0 0', lineHeight: 1.5 }}>
                      {status.error}
                    </p>
                  </details>
                )}
              </div>
            </div>
          )}

          {officialQuestions.length > 0 && (
            <section style={{ display: 'grid', gap: '0.85rem' }}>
              <div className="card-heading">
                <div><p className="eyebrow">OFFICIAL MATERIAL</p><h2>Retrieved AQA questions</h2></div>
                <Badge tone="teal">{officialQuestions.length} questions</Badge>
              </div>
              {officialQuestions.map((question, index) => (
                <AssessmentQuestionCard
                  key={`official-${question.question_id}-${index}`}
                  runId={runId}
                  question={question}
                  position={index + 1}
                  official
                  onStatusUpdated={updateStatus}
                />
              ))}
            </section>
          )}

          {generatedQuestions.length > 0 && (
            <section style={{ display: 'grid', gap: '0.85rem' }}>
              <div className="card-heading">
                <div>
                  <p className="eyebrow">
                    {status?.mode === 'retrieve_hybrid' ? 'AI SHORTFALL FILL' : 'AI-GENERATED ASSESSMENT'}
                  </p>
                  <h2>
                    {status?.mode === 'retrieve_hybrid'
                      ? 'Generated questions for missing coverage'
                      : 'Generated quiz questions'}
                  </h2>
                </div>
                <Badge tone="info">{generatedQuestions.length} questions</Badge>
              </div>
              {generatedQuestions.map((question, index) => (
                <AssessmentQuestionCard
                  key={`generated-${question.question_id}-${index}`}
                  runId={runId}
                  question={question}
                  position={officialQuestions.length + index + 1}
                  official={false}
                  planIndex={Number(question.plan_index || index + 1)}
                  quizMode={status?.generated?.quiz_mode}
                  onStatusUpdated={updateStatus}
                />
              ))}
            </section>
          )}

          {resultsReady && generatedQuestions.length > 0 && status?.generated && (
            <GeneratedQuizHITL
              runId={runId}
              status={status}
              onUpdated={updateStatus}
            />
          )}

          {primaryPdf && (
            <Card>
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: '1rem',
                  flexWrap: 'wrap',
                }}
              >
                <div>
                  <p className="eyebrow" style={{ marginBottom: '0.25rem' }}>
                    FINAL ASSESSMENT
                  </p>
                  <h2 style={{ margin: 0 }}>Assessment PDF ready</h2>
                  <p
                    className="muted"
                    style={{
                      marginTop: '0.45rem',
                      marginBottom: 0,
                    }}
                  >
                    Questions, marking schemes and generated diagrams are included
                    in the saved Agent 2 PDF.
                  </p>
                </div>

                <a
                  className="button button-primary"
                  href={assessmentAssetUrl(runId, primaryPdf)}
                  download
                  target="_blank"
                  rel="noreferrer"
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    minWidth: '170px',
                    justifyContent: 'center',
                  }}
                >
                  Download PDF
                  <ArrowRight size={16} />
                </a>
              </div>
            </Card>
          )}
        </div>
      )}
    </>
  )
}

export default function Page() {
  const [page, setPage] = useState('home')
  const [open, setOpen] = useState(true)

  const [activeRun, setActiveRun] =
    useState<CreateRunResponse | null>(null)

  const [runProgress, setRunProgress] =
    useState<RunProgressResponse | null>(null)

  const [preprocessingData, setPreprocessingData] =
    useState<PreprocessingResponse | null>(null)

  const [semanticData, setSemanticData] =
    useState<SemanticResponse | null>(null)

  const [topicsData, setTopicsData] =
    useState<TopicsResponse | null>(null)

  const go = (p: string) => setPage(p)

  // ----------------------------------------------------------
  // REAL Agent 1 progress polling.
  //
  // The percentage is derived from actual backend artifacts:
  // Module 1 -> 40%, Module 2 -> 70%, Module 3/gate -> 100%.
  // As each real output becomes available, load it automatically.
  // ----------------------------------------------------------
  useEffect(() => {
    const runId = activeRun?.run_id

    if (!runId) return

    let cancelled = false
    let timer: number | null = null

    const poll = async () => {
      try {
        const progress = await getRunProgress(runId)

        if (cancelled) return

        setRunProgress(progress)

        if (
          progress.module1_ready
          && !preprocessingData
        ) {
          const value = await getPreprocessing(runId)

          if (!cancelled) {
            setPreprocessingData(value)
          }
        }

        if (
          progress.module2_ready
          && !semanticData
        ) {
          const value = await getSemantic(runId)

          if (!cancelled) {
            setSemanticData(value)
          }
        }

        if (progress.module3_ready) {
          // Load while the graph is finishing and once more after it
          // reaches its authoritative human gate.
          if (
            !topicsData
            || progress.background_status !== 'running'
          ) {
            const value = await getTopics(runId)

            if (!cancelled) {
              setTopicsData(value)
            }
          }
        }

        if (
          !cancelled
          && ['queued', 'running'].includes(
            progress.background_status
          )
        ) {
          timer = window.setTimeout(poll, 750)
        }
      } catch (error) {
        if (cancelled) return

        console.error(
          'Could not refresh Agent 1 progress:',
          error
        )

        timer = window.setTimeout(poll, 1200)
      }
    }

    poll()

    return () => {
      cancelled = true

      if (timer !== null) {
        window.clearTimeout(timer)
      }
    }
  }, [
    activeRun?.run_id,
    preprocessingData,
    semanticData,
    topicsData,
  ])

  const resetRun = () => {
    setActiveRun(null)
    setRunProgress(null)
    setPreprocessingData(null)
    setSemanticData(null)
    setTopicsData(null)
  }

  const content = useMemo(() => {
    switch (page) {
      case 'home':
        return <Home go={go} />

      case 'dashboard':
        return (
          <Dashboard
            go={go}
            activeRun={activeRun}
          />
        )

      case 'transcript':
        return (
          <Transcript
            go={go}
            onRunCreated={setActiveRun}
            onProcessingStarted={resetRun}
          />
        )

      case 'preprocessing':
      case 'preprocessed':
        return preprocessingData ? (
          <Preprocessed
            go={go}
            data={preprocessingData}
          />
        ) : (
          <Processing
            kind="preprocessing"
            progress={runProgress}
          />
        )

      case 'semantic':
      case 'semantic-done':
        return semanticData ? (
          <SemanticDone
            go={go}
            data={semanticData}
          />
        ) : (
          <Processing
            kind="semantic"
            progress={runProgress}
          />
        )

      case 'topics':
        return topicsData ? (
          <Topics
            go={go}
            data={topicsData}
            semanticData={semanticData}
            onUpdated={setTopicsData}
          />
        ) : (
          <Processing
            kind="topics"
            progress={runProgress}
          />
        )

      case 'assessment':
        return (
          <Assessment
            runId={
              activeRun?.run_id
              ?? topicsData?.run_id
              ?? ''
            }
          />
        )

      default:
        return <Home go={go} />
    }
  }, [
    page,
    activeRun,
    runProgress,
    preprocessingData,
    semanticData,
    topicsData,
  ])

  return (
    <div
      className={`app-shell ${
        page === 'home' ? 'home-shell' : ''
      }`}
    >
      {page !== 'home' && (
        <Sidebar
          page={page}
          setPage={setPage}
          open={open}
          setOpen={setOpen}
        />
      )}

      <style>{`
        .guidance,
        .success-banner {
          position: relative;
          display: grid;
          grid-template-columns: auto minmax(0, 1fr);
          align-items: start;
          gap: 0.95rem;
          padding: 1.05rem 1.15rem;
          border-radius: 18px;
          border: 1px solid rgba(255,255,255,0.16);
          backdrop-filter: blur(14px);
          -webkit-backdrop-filter: blur(14px);
          box-shadow:
            0 18px 40px rgba(31, 24, 47, 0.12),
            inset 0 1px 0 rgba(255,255,255,0.08);
          overflow: hidden;
        }

        .guidance::before,
        .success-banner::before {
          content: "";
          position: absolute;
          inset: 0;
          pointer-events: none;
          background:
            linear-gradient(
              135deg,
              rgba(255,255,255,0.10),
              rgba(255,255,255,0.02) 45%,
              transparent 100%
            );
        }

        .guidance {
          background:
            linear-gradient(
              135deg,
              rgba(120, 44, 68, 0.88),
              rgba(88, 34, 54, 0.84)
            );
          border-color: rgba(255, 194, 209, 0.20);
        }

        .success-banner {
          background:
            linear-gradient(
              135deg,
              rgba(128, 214, 158, 0.98),
              rgba(104, 200, 140, 0.96)
            );
          border-color: rgba(255, 255, 255, 0.22);
          color: #ffffff;
          grid-template-columns: auto minmax(0, max-content);
          justify-content: center;
          align-items: center;
          text-align: center;
          width: fit-content;
          max-width: min(92%, 760px);
          margin-left: auto;
          margin-right: auto;
          padding: 1rem 1.35rem;
          box-shadow:
            0 18px 40px rgba(49, 121, 79, 0.18),
            inset 0 1px 0 rgba(255,255,255,0.18);
        }

        .guidance > svg,
        .success-banner > svg {
          position: relative;
          z-index: 1;
          width: 22px;
          height: 22px;
          flex: 0 0 auto;
          margin-top: 0.05rem;
          padding: 0.5rem;
          border-radius: 999px;
          background: rgba(255,255,255,0.13);
          border: 1px solid rgba(255,255,255,0.16);
          box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
        }

        .success-banner > svg {
          background: rgba(255,255,255,0.18);
          border-color: rgba(255,255,255,0.28);
          color: #ffffff;
          margin-top: 0;
        }

        .guidance > div,
        .success-banner > div {
          position: relative;
          z-index: 1;
          min-width: 0;
        }

        .success-banner > div {
          text-align: center;
        }

        .guidance strong,
        .success-banner strong {
          display: block;
          margin: 0 0 0.18rem;
          font-size: 1.04rem;
          line-height: 1.25;
          letter-spacing: -0.01em;
          color: #fff;
        }

        .success-banner strong {
          color: #ffffff;
        }

        .guidance p,
        .success-banner p {
          margin: 0;
          line-height: 1.6;
          color: rgba(255,255,255,0.92);
        }

        .success-banner p {
          color: rgba(255,255,255,0.92);
        }

        .guidance small,
        .success-banner small {
          color: rgba(255,255,255,0.78);
        }

        .success-banner small {
          color: rgba(255,255,255,0.82);
        }
      `}</style>

      <main className="main-content">
        {content}
      </main>
    </div>
  )
}
