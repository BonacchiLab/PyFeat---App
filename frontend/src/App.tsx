import { useEffect, useRef, useState } from 'react'
import introGuide from './introGuide.html?raw'

const API = 'http://127.0.0.1:8000'
type Stimulus = { nome: string; inicio: number; fim: number }
type VideoEntry = { file: File; subject: string; stimuli: Stimulus[]; fps?: number; frameCount?: number; duration?: number; metadataError?: boolean }
type NamedStimulus = { id: number; name: string }
type EmotionFrame = { frame: number | null; timestamp: string | number | null; dominant_emotion: string | null; dominant_emotions: string[]; dominant_score: number | null; emotions: Record<string, number | null> }
type FaceDetection = { dominant_emotion: string | null; dominant_emotions: string[]; emotions: Record<string, number | null>; action_units: Record<string, number | null>; face_box: Record<string, number | null> | null; landmarks: { x: number; y: number }[] | null }
type InspectedFrame = EmotionFrame & { time_sec: number | null; approx_time: string | null; action_units: Record<string, number | null>; landmarks: { x: number; y: number }[] | null; detections: FaceDetection[] }
type PreviewData = { columns: string[]; rows: number; preview: Record<string, unknown>[]; face_score_median: number | null; face_score_missing: number | null; face_score_flagged: number | null; detected_frames: number; expected_rows: number; time_column: string | null; chart_points: Record<string, number | null>[]; chart_columns: { action_units: string[]; emotions: string[]; head_pose: string[] }; emotion_timeline: EmotionFrame[]; stimulus_statistics?: Record<string, Record<string, { before: number | null; during: number | null }>> }
type ResultFile = { name: string; url: string; preview: PreviewData; stimuli: Stimulus[]; subject: string; taskId: string; videoFile: File; samplingMode: string; fps: number }

function roundLikePython(value: number) {
  const lower = Math.floor(value)
  const fraction = value - lower
  if (fraction < 0.5) return lower
  if (fraction > 0.5) return lower + 1
  return lower % 2 === 0 ? lower : lower + 1
}

export default function App() {
  const [page, setPage] = useState<'introduction' | 'workspace'>('introduction')
  const [videos, setVideos] = useState<VideoEntry[]>([])
  const [mode, setMode] = useState('fast_fps')
  const [targetFps, setTargetFps] = useState(1)
  const [shared, setShared] = useState(false)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('Add videos to begin.')
  const [downloads, setDownloads] = useState<ResultFile[]>([])
  const [addMetadata, setAddMetadata] = useState(false)
  const [stimuli, setStimuli] = useState<NamedStimulus[]>([{ id: 1, name: 'Stimulus_1' }])
  const [sharedWindows, setSharedWindows] = useState<Stimulus[]>([{ nome: 'Stimulus_1', inicio: 30, fim: 90 }])
  const [runtimeHistory, setRuntimeHistory] = useState<Record<string, { rows: number; seconds: number }>>({})
  const downloadUrls = useRef<string[]>([])

  useEffect(() => () => downloadUrls.current.forEach(url => URL.revokeObjectURL(url)), [])

  function addFiles(files: FileList | null) {
    if (!files) return
    const additions = Array.from(files).filter(file => !videos.some(v => v.file.name === file.name)).map(file => ({ file, subject: '', stimuli: stimuli.map((item, i) => ({ nome: item.name, inicio: 30 * (i + 1), fim: 90 * (i + 1) })) }))
    setVideos(current => [...current, ...additions])
    additions.forEach(async video => {
      const form = new FormData(); form.append('file', video.file)
      try {
        const response = await fetch(`${API}/api/extractions/metadata`, { method: 'POST', body: form })
        if (!response.ok) throw new Error('Could not read video metadata')
        const info = await response.json()
        setVideos(current => current.map(item => item.file === video.file ? { ...item, fps: info.fps, frameCount: info.frame_count, duration: info.duration_seconds, metadataError: false } : item))
      } catch {
        setVideos(current => current.map(item => item.file === video.file ? { ...item, metadataError: true } : item))
      }
    })
  }
  function updateVideo(index: number, update: Partial<VideoEntry>) {
    setVideos(current => current.map((video, i) => i === index ? { ...video, ...update } : video))
  }
  function setStimulusCount(count: number) {
    const next = Array.from({ length: count }, (_, i) => stimuli[i] ?? { id: i + 1, name: `Stimulus_${i + 1}` })
    setStimuli(next)
    setSharedWindows(current => Array.from({ length: count }, (_, i) => current[i] ?? { nome: next[i].name, inicio: 30 * (i + 1), fim: 90 * (i + 1) }))
    setVideos(current => current.map(video => ({ ...video, stimuli: Array.from({ length: count }, (_, i) => video.stimuli[i] ?? { nome: next[i].name, inicio: 30 * (i + 1), fim: 90 * (i + 1) }) })))
  }
  function renameStimulus(index: number, name: string) {
    const nextNames = stimuli.map((stimulus, i) => i === index ? { ...stimulus, name } : stimulus)
    setStimuli(nextNames)
    setSharedWindows(current => current.map((window, i) => i === index ? { ...window, nome: name } : window))
    setVideos(current => current.map(video => ({ ...video, stimuli: video.stimuli.map((window, i) => i === index ? { ...window, nome: name } : window) })))
  }
  function updateWindow(index: number, stimulusIndex: number, field: 'inicio' | 'fim', value: number, isShared: boolean) {
    if (isShared) setSharedWindows(current => current.map((window, i) => i === stimulusIndex ? { ...window, nome: stimuli[i].name, [field]: value } : window))
    else setVideos(current => current.map((video, i) => i === index ? { ...video, stimuli: video.stimuli.map((window, j) => j === stimulusIndex ? { ...window, nome: stimuli[j].name, [field]: value } : window) } : video))
  }
  async function extract() {
    if (!videos.length || videos.some(video => !video.subject.trim())) { setMessage('Add at least one video and a subject ID for each.'); return }
    setBusy(true)
    downloadUrls.current.forEach(url => URL.revokeObjectURL(url))
    downloadUrls.current = []
    setDownloads([])
    try {
      const results = []
      for (const video of videos) {
        const videoStartedAt = performance.now()
        setMessage(`Uploading and processing ${video.file.name}…`)
        const form = new FormData()
        form.append('file', video.file); form.append('subject_id', video.subject)
        form.append('sampling_mode', mode); form.append('target_fps', String(targetFps))
        const videoStimuli = addMetadata ? (shared ? sharedWindows : video.stimuli) : []
        form.append('stimuli_json', JSON.stringify(videoStimuli))
        const start = await fetch(`${API}/api/extractions`, { method: 'POST', body: form })
        if (!start.ok) throw new Error((await start.text()) || `API error ${start.status}`)
        const { task_id } = await start.json()
        let task
        do {
          await new Promise(resolve => setTimeout(resolve, 700))
          const status = await fetch(`${API}/api/extractions/${task_id}`)
          if (!status.ok) throw new Error(`Could not read extraction progress (${status.status}).`)
          task = await status.json()
          setMessage(`${video.file.name}: ${task.message} (${Math.round(task.progress * 100)}%)`)
          if (task.status === 'failed') throw new Error(task.error || 'Extraction failed')
        } while (task.status !== 'completed')
        const response = await fetch(`${API}/api/extractions/${task_id}/download`)
        if (!response.ok) throw new Error('Could not download the Parquet result.')
        const previewResponse = await fetch(`${API}/api/extractions/${task_id}/preview?limit=50&threshold=0.9&include_timeline=${mode === 'all_frames' ? 'false' : 'true'}`)
        if (!previewResponse.ok) throw new Error(`Could not load data preview (${previewResponse.status}): ${await previewResponse.text()}`)
        const preview: PreviewData = await previewResponse.json()
        const url = URL.createObjectURL(await response.blob())
        downloadUrls.current.push(url)
        const safeSubject = video.subject.trim().replace(/[<>:"/\\|?*\x00-\x1f]/g, '_').replace(/[ .]+$/, '') || 'subject'
        results.push({ name: `${safeSubject}.parquet`, url, preview, stimuli: videoStimuli, subject: video.subject.trim(), taskId: task_id, videoFile: video.file, samplingMode: mode, fps: video.fps ?? 30 })
        setDownloads([...results])
        const elapsedSeconds = (performance.now() - videoStartedAt) / 1000
        setRuntimeHistory(current => {
          const previous = current[mode] ?? { rows: 0, seconds: 0 }
          return { ...current, [mode]: { rows: previous.rows + (preview.expected_rows || preview.rows), seconds: previous.seconds + elapsedSeconds } }
        })
      }
      setMessage('All videos processed successfully.')
    } catch (error) { setMessage(error instanceof Error ? error.message : 'Unexpected error.') }
    finally { setBusy(false) }
  }

  const estimatedRows = videos.reduce((sum, video) => {
    const frames = video.frameCount ?? 0, fps = video.fps ?? 30
    const skip = mode === 'all_frames' ? 1 : Math.max(1, roundLikePython(fps / (mode === 'custom_fps' ? targetFps : 1)))
    return sum + Math.ceil(frames / skip)
  }, 0)
  const modeHistory = runtimeHistory[mode]
  const measuredRows = Object.values(runtimeHistory).reduce((sum, history) => sum + history.rows, 0)
  const measuredSeconds = Object.values(runtimeHistory).reduce((sum, history) => sum + history.seconds, 0)
  const estimatedSeconds = modeHistory?.rows
    ? estimatedRows * modeHistory.seconds / modeHistory.rows
    : measuredRows
      ? estimatedRows * measuredSeconds / measuredRows
      : estimatedRows * 4

  return <>
    <header className="topbar"><div className="brand">FEDET</div><nav><button className={page === 'introduction' ? 'selected' : ''} onClick={() => setPage('introduction')}>Introduction</button><button className={page === 'workspace' ? 'selected' : ''} onClick={() => setPage('workspace')}>Analysis Tool</button></nav></header>
    {page === 'introduction' ? <IntroPage onStart={() => setPage('workspace')} /> : <div className="workspace-layout"><aside className="sampling-sidebar"><h2>Extraction controls</h2><label>Frame sampling<select value={mode} onChange={event=>setMode(event.target.value)}><option value="all_frames">Every frame</option><option value="fast_fps">1 FPS (fast mode)</option><option value="custom_fps">Custom sampling</option></select></label>{mode==='fast_fps'&&<small>Fast mode analyzes one representative frame per second.</small>}{mode==='custom_fps'&&<><label>Target sampling frequency (FPS)<input type="number" min="0.1" step="0.1" value={targetFps} onChange={event=>setTargetFps(Number(event.target.value))}/></label><small>The selected frequency is approximate and depends on the source video's frame rate.</small></>}<hr/><h3>Processing time estimate</h3>{videos.length?<><div className="sidebar-metric"><span>Frames to analyze</span><strong>{estimatedRows.toLocaleString()}</strong></div><div className="sidebar-metric"><span>Output rows</span><strong>{estimatedRows.toLocaleString()}</strong></div><div className="sidebar-metric"><span>Estimated time</span><strong>{Math.floor(estimatedSeconds/60)}m {Math.round(estimatedSeconds%60)}s</strong></div>{measuredRows===0&&<small>This is an initial estimate; it improves after extraction.</small>}</>:<p className="muted">Upload videos to calculate the estimate.</p>}</aside><main className="shell workspace-main">
    <section className="workspace-heading"><h1>Data Upload and Subject Identification</h1></section>
    <section className="panel"><h2>1. Videos and subjects</h2><label className="drop">Choose video files<input type="file" accept=".mp4,.avi,.mov" multiple onChange={event => { addFiles(event.target.files); event.currentTarget.value = '' }}/><small>MP4, AVI, MOV · Videos are processed locally</small></label>
      {videos.map((video, index) => <div className="video" key={`${video.file.name}-${index}`}><div><strong>{video.file.name}</strong><small className="video-meta">{video.fps ? `${video.frameCount?.toLocaleString()} frames · ${video.fps.toFixed(2)} FPS · ${Math.floor((video.duration ?? 0)/60)}:${String(Math.floor((video.duration ?? 0)%60)).padStart(2,'0')}` : video.metadataError ? 'Video metadata unavailable; estimate may be inaccurate.' : 'Reading video metadata…'}</small></div><label>Subject ID<input value={video.subject} onChange={event => updateVideo(index, { subject: event.target.value })} placeholder="e.g. Subject_01"/></label><button className="remove-video" type="button" disabled={busy} onClick={() => setVideos(current => current.filter((_, i) => i !== index))}>Remove</button></div>)}
    </section>
    <section className="panel"><h2>2. Experimental design metadata</h2>
      <label className="inline"><input type="checkbox" checked={addMetadata} onChange={event => setAddMetadata(event.target.checked)}/> Add stimulus markers to the dataset</label><p className="muted metadata-help">Stimulus names are shared across videos, but each video can have different time intervals. Overlapping intervals are marked in both columns.</p>
      {addMetadata && <><p className="muted">Define the stimuli, then enter each stimulus time window in seconds.</p>
        <label>Number of stimuli<input type="number" min="1" max="20" value={stimuli.length} onChange={event => setStimulusCount(Math.max(1, Number(event.target.value) || 1))}/></label>
        {stimuli.map((stimulus, i) => <label key={stimulus.id}>{`Stimulus ${i + 1} name`}<input value={stimulus.name} onChange={event => renameStimulus(i, event.target.value)}/></label>)}
        <label className="inline"><input type="checkbox" checked={shared} onChange={event => setShared(event.target.checked)}/> Use the same stimulus windows for every video</label>
        {shared ? sharedWindows.map((window, i) => <fieldset key={i}><legend>{stimuli[i].name}</legend><div className="window-fields"><label>Start (seconds)<input type="number" min="0" value={window.inicio} onChange={event => updateWindow(0, i, 'inicio', Number(event.target.value), true)}/></label><label>End (seconds)<input type="number" min="0" value={window.fim} onChange={event => updateWindow(0, i, 'fim', Number(event.target.value), true)}/></label></div></fieldset>) : videos.map((video, videoIndex) => <details key={video.file.name}><summary>Stimulus windows — {video.subject || video.file.name}</summary>{video.stimuli.map((window, i) => <fieldset key={i}><legend>{stimuli[i].name}</legend><div className="window-fields"><label>Start (seconds)<input type="number" min="0" value={window.inicio} onChange={event => updateWindow(videoIndex, i, 'inicio', Number(event.target.value), false)}/></label><label>End (seconds)<input type="number" min="0" value={window.fim} onChange={event => updateWindow(videoIndex, i, 'fim', Number(event.target.value), false)}/></label></div></fieldset>)}</details>)}
      </>}
    </section>
    <section className="panel processing-panel"><h2>3. Processing</h2><p className="muted">Upload a video and select Extract data to generate the output and its interactive visualizations.</p><button className="primary extract-button" disabled={busy} onClick={extract}>{busy ? 'Processing…' : 'Extract data'}</button><p className="status" role="status">{message}</p></section>
    {downloads.length > 0 && <StimulusAnalysis results={downloads} />}
    {downloads.length > 0 && <AdvancedStimulusAnalysis results={downloads} />}
    {downloads.map(result => <ResultCard key={result.taskId} result={result} />)}
  </main></div>}
  </>
}

function ResultCard({ result }: { result: ResultFile }) {
  const [category, setCategory] = useState<'action_units' | 'emotions' | 'head_pose'>('action_units')
  const [threshold, setThreshold] = useState(0.9)
  const [flaggedScores, setFlaggedScores] = useState(result.preview.face_score_flagged)
  const chartColumns = result.preview.chart_columns ?? { action_units: [], emotions: [], head_pose: [] }
  const chartPoints = result.preview.chart_points ?? []
  const available = chartColumns[category] ?? []
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>([])
  const activeMetrics = selectedMetrics.filter(item => available.includes(item))
  const displayedMetrics = activeMetrics.length ? activeMetrics : available.slice(0, 1)
  useEffect(() => {
    fetch(`${API}/api/extractions/${result.taskId}/preview?limit=1&threshold=${threshold}&include_timeline=false`).then(response => response.ok ? response.json() : null).then(data => { if (data) setFlaggedScores(data.face_score_flagged) }).catch(() => undefined)
  }, [result.taskId, threshold])
  return <section className="panel results"><h2>Extraction complete — {result.name.replace(/\.parquet$/i, '')}</h2><div className="quality-cards"><div><small>Output rows</small><strong>{result.preview.rows.toLocaleString()}</strong></div><div><small>Median FaceScore</small><strong>{result.preview.face_score_median?.toFixed(3) ?? 'N/A'}</strong></div><div><small>Missing FaceScore</small><strong>{result.preview.face_score_missing?.toLocaleString() ?? 'N/A'}</strong></div></div>
    {result.samplingMode === 'all_frames' ? <VideoFrameInspector result={result} /> : <div className="frame-inspection-warning">Frame-synchronized video inspection is available only with <strong>Every frame (all_frames)</strong> sampling. This result used {result.samplingMode === 'fast_fps' ? '1 FPS sampling' : 'custom FPS sampling'}; its existing emotion timeline represents sampled output rows.</div>}
    {result.preview.emotion_timeline?.length > 0 && <EmotionTimeline frames={result.preview.emotion_timeline} emotionNames={chartColumns.emotions} />}
    <h3>Data quality</h3><div className="quality-cards"><div><small>Frames with detections</small><strong>{result.preview.detected_frames.toLocaleString()} / {result.preview.expected_rows.toLocaleString()}</strong></div><div><small>Detection coverage</small><strong>{result.preview.expected_rows ? `${Math.min(result.preview.detected_frames/result.preview.expected_rows,1).toLocaleString(undefined,{style:"percent",maximumFractionDigits:1})}` : "N/A"}</strong></div><div><small>FaceScores below limit</small><strong>{flaggedScores?.toLocaleString() ?? "N/A"}</strong></div></div>{chartPoints.some(point => point.FaceScore != null) ? <><label>Flag FaceScore below {threshold.toFixed(2)}<input type="range" min="0" max="1" step="0.05" value={threshold} onChange={event=>setThreshold(Number(event.target.value))}/></label><LineChart points={chartPoints} metrics={["FaceScore"]} timeColumn={result.preview.time_column ?? "frame"} scatter threshold={threshold}/><ChartInfo text="Each point is a frame-level detection. The horizontal axis is time or frame; the vertical axis is FaceScore (0?1). The red line is your review threshold. A low score calls for image inspection; it does not tell you whether an AU or emotion is correct."/></> : <p className="muted">FaceScore is not available.</p>}
    <h3>Signals over time</h3><div className="chart-controls"><select value={category} onChange={event=>{setCategory(event.target.value as typeof category);setSelectedMetrics([])}}><option value="action_units">Action Units</option><option value="emotions">Emotions</option><option value="head_pose">Head pose</option></select>{available.length>0&&<MetricDropdown options={available} selected={displayedMetrics} onChange={setSelectedMetrics}/>}</div>{available.length?<><LineChart points={chartPoints} metrics={displayedMetrics} timeColumn={result.preview.time_column??"frame"} stimuli={result.stimuli}/><ChartInfo text="The horizontal axis is time or frame; the vertical axis uses a shared scale for the selected signals. Choose multiple AUs, emotions, or pose metrics to compare their trends. Shaded bands mark stimuli; these values are model estimates."/></>:<p className="muted">No signals in this category.</p>}
    {chartColumns.action_units.length>0&&<><h3>AU heatmap</h3><Heatmap points={chartPoints} columns={chartColumns.action_units} timeColumn={result.preview.time_column??"frame"} stimuli={result.stimuli}/><ChartInfo text="Each row is an AU and each column is a sampled frame. Color shows the model estimate; blank cells mean missing data, not zero. Blue indicates lower activation and red indicates higher activation. Shaded bands mark stimuli and help locate changes over time."/></>}
    <h3>Data preview (first 50 rows)</h3><div className="table-scroll"><table><thead><tr>{result.preview.columns.map(column => <th key={column}>{column}</th>)}</tr></thead><tbody>{result.preview.preview.map((row, i) => <tr key={i}>{result.preview.columns.map(column => <td key={column}>{row[column] == null ? '' : typeof row[column] === 'object' ? JSON.stringify(row[column]) : String(row[column])}</td>)}</tr>)}</tbody></table></div><a className="download" href={result.url} download={result.name}>Download Parquet</a>
  </section>
}

const FRAME_PAGE_SIZE = 500

function VideoFrameInspector({ result }: { result: ResultFile }) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [videoUrl, setVideoUrl] = useState('')
  const [frames, setFrames] = useState<InspectedFrame[]>([])
  const [total, setTotal] = useState(0)
  const [coordinateSize, setCoordinateSize] = useState<{ width: number; height: number } | null>(null)
  const [offset, setOffset] = useState(0)
  const [currentFrame, setCurrentFrame] = useState(0)
  const [emotionFilter, setEmotionFilter] = useState('all')
  const [overlayEnabled, setOverlayEnabled] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    const url = URL.createObjectURL(result.videoFile)
    setVideoUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [result.videoFile])

  useEffect(() => {
    let cancelled = false
    fetch(`${API}/api/extractions/${result.taskId}/frames?offset=${offset}&limit=${FRAME_PAGE_SIZE}`)
      .then(async response => {
        if (!response.ok) throw new Error(`Could not load frame data (${response.status}).`)
        return response.json()
      })
      .then(data => { if (!cancelled) { setFrames(data.frames); setTotal(data.total); setCoordinateSize(data.coordinate_width && data.coordinate_height ? { width: data.coordinate_width, height: data.coordinate_height } : null); setError('') } })
      .catch(reason => { if (!cancelled) setError(reason instanceof Error ? reason.message : 'Could not load frame data.') })
    return () => { cancelled = true }
  }, [result.taskId, offset])

  const seekToFrame = (frame: InspectedFrame) => {
    const time = frame.time_sec ?? (frame.frame ?? 0) / result.fps
    if (videoRef.current && Number.isFinite(time)) videoRef.current.currentTime = time
    setCurrentFrame(frame.frame ?? 0)
  }
  const syncCurrentFrame = () => {
    const frameNumber = Math.max(0, Math.round((videoRef.current?.currentTime ?? 0) * result.fps))
    setCurrentFrame(frameNumber)
    if (frameNumber >= offset + FRAME_PAGE_SIZE && offset + FRAME_PAGE_SIZE < total) setOffset(offset + FRAME_PAGE_SIZE)
    else if (frameNumber < offset && offset > 0) setOffset(Math.max(0, Math.floor(frameNumber / FRAME_PAGE_SIZE) * FRAME_PAGE_SIZE))
  }

  const current = frames.find(frame => frame.frame === currentFrame) ?? null
  useEffect(() => {
    const canvas = canvasRef.current, video = videoRef.current
    if (!canvas || !video || !overlayEnabled || !current?.detections?.some(detection => detection.landmarks?.length)) {
      if (canvas) canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height)
      return
    }
    const width = video.videoWidth, height = video.videoHeight
    if (!width || !height) return
    const scaleX = coordinateSize?.width ? width / coordinateSize.width : 1
    const scaleY = coordinateSize?.height ? height / coordinateSize.height : 1
    canvas.width = width; canvas.height = height
    const context = canvas.getContext('2d')
    if (!context) return
    context.clearRect(0, 0, width, height)
    for (const detection of current.detections) {
      if (detection.face_box) {
        const { FaceRectX: x, FaceRectY: y, FaceRectWidth: boxWidth, FaceRectHeight: boxHeight } = detection.face_box
        if (x != null && y != null && boxWidth != null && boxHeight != null) {
          context.strokeStyle = '#35e0c1'; context.lineWidth = Math.max(2, width / 500); context.strokeRect(x * scaleX, y * scaleY, boxWidth * scaleX, boxHeight * scaleY)
        }
      }
      if (detection.landmarks?.length) {
        context.fillStyle = '#35e0c1'
        for (const point of detection.landmarks) {
          context.beginPath(); context.arc(point.x * scaleX, point.y * scaleY, Math.max(2, width / 320), 0, Math.PI * 2); context.fill()
        }
      }
    }
  }, [current, coordinateSize, overlayEnabled, videoUrl])

  const emotionNames = Object.keys(EMOTION_LABELS)
  const visibleFrames = emotionFilter === 'all' ? frames : frames.filter(frame => (frame.dominant_emotions ?? emotionNamesFromLabel(frame.dominant_emotion)).includes(emotionFilter))
  const currentDetection = current?.detections?.[0]
  const activeAus = currentDetection ? Object.entries(currentDetection.action_units).filter((entry): entry is [string, number] => typeof entry[1] === 'number' && entry[1] > 0).sort((a, b) => b[1] - a[1]).slice(0, 5) : []
  return <section className="video-inspector">
    <h3>Synced video and frame inspection</h3>
    <p className="muted">Every block is one analyzed source frame. Playback follows the video; selecting a block seeks to its timestamp.</p>
    {error && <p className="frame-inspection-warning">{error}</p>}
    <div className="video-inspector-layout"><div className="video-stage"><video ref={videoRef} src={videoUrl} controls onTimeUpdate={syncCurrentFrame} onSeeked={syncCurrentFrame} onLoadedMetadata={syncCurrentFrame} /><canvas ref={canvasRef} className="landmark-overlay" aria-hidden="true" />{overlayEnabled && current && <div className="video-live-overlay"><strong>{current.detections.length} face detection(s)</strong>{currentDetection && <><span>{currentDetection.dominant_emotion ?? 'No emotion estimate'}</span><span>{Object.entries(currentDetection.emotions).map(([name, value]) => `${name}: ${value?.toFixed(2) ?? '—'}`).join(' · ')}</span></>}{activeAus.length > 0 && <span>{activeAus.map(([name, value]) => `${name}: ${value.toFixed(2)}`).join(' · ')}</span>}{!current.detections.some(detection => detection.landmarks?.length) && <span>Landmarks unavailable for this frame</span>}</div>}</div>
      <div className="video-frame-details"><strong>Frame {current?.frame ?? currentFrame}</strong><span>Time: {current?.approx_time ?? `${(currentFrame / result.fps).toFixed(3)} s`}</span><span>Detections in frame: {current?.detections.length ?? 0}</span><span>Dominant emotion (first detection): {currentDetection?.dominant_emotion ?? 'No detection'}</span><span>Emotions: {currentDetection ? Object.entries(currentDetection.emotions).map(([name, value]) => `${name} ${value?.toFixed(3) ?? 'N/A'}`).join(' · ') : 'No frame data loaded'}</span><span>Top active AUs (first detection): {activeAus.map(([name, value]) => `${name} ${value.toFixed(3)}`).join(' · ') || 'None'}</span><span>Landmarks: {current?.detections.some(detection => detection.landmarks?.length) ? 'Available for overlay' : 'Unavailable'}</span></div></div>
    <label className="overlay-toggle"><input type="checkbox" checked={overlayEnabled} onChange={event => setOverlayEnabled(event.target.checked)} />Enable video overlay (emotions, AUs and available landmarks)</label>
    <div className="frame-timeline-controls"><label>Filter emotion<select value={emotionFilter} onChange={event => setEmotionFilter(event.target.value)}><option value="all">All emotions</option>{emotionNames.map(name => <option key={name} value={name}>{name}</option>)}</select></label><span>{total.toLocaleString()} analyzed frames · showing {Math.min(offset + 1, total)}–{Math.min(offset + frames.length, total)}</span></div>
    <div className="emotion-timeline" role="list" aria-label="Analyzed video frames">{visibleFrames.map((frame, index) => {const name = frame.dominant_emotion;const selected = frame.frame === currentFrame;return <button key={`${frame.frame}-${index}`} type="button" className={`emotion-frame${selected ? ' selected' : ''}`} style={{background:emotionColor(name)}} title={`Frame ${frame.frame} · ${frame.approx_time ?? frame.time_sec ?? 'time unavailable'} · ${name ?? 'No detection'}`} aria-label={`Frame ${frame.frame}, ${name ?? 'no emotion'}`} onClick={() => seekToFrame(frame)}>{name ? emotionNamesFromLabel(name).length > 1 ? '=' : (EMOTION_LABELS[name] ?? name).slice(0,1) : '?'}</button>})}</div>
    <div className="frame-page-controls"><button type="button" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - FRAME_PAGE_SIZE))}>Previous frames</button><button type="button" disabled={offset + frames.length >= total} onClick={() => setOffset(offset + FRAME_PAGE_SIZE)}>Next frames</button><small>Page {total ? Math.floor(offset / FRAME_PAGE_SIZE) + 1 : 0} of {Math.max(1, Math.ceil(total / FRAME_PAGE_SIZE))}</small></div>
  </section>
}

function emotionNamesFromLabel(label: string | null) { return label ? label.split(' + ') : [] }

const EMOTION_COLORS: Record<string, string> = { anger: '#d95d59', disgust: '#78a66a', fear: '#a78bca', happiness: '#e8bd4a', sadness: '#5794c5', surprise: '#e98b4d', neutral: '#87949a' }
const EMOTION_LABELS: Record<string, string> = { anger: 'Anger', disgust: 'Disgust', fear: 'Fear', happiness: 'Happiness', sadness: 'Sadness', surprise: 'Surprise', neutral: 'Neutral' }
function emotionNames(value: string | null) { return value ? value.split(' + ') : [] }
function emotionLabel(value: string | null) { return emotionNames(value).map(name => EMOTION_LABELS[name] ?? name).join(' + ') }
function emotionColor(value: string | null) {
  const colors = emotionNames(value).map(name => EMOTION_COLORS[name] ?? '#66777d')
  if (!colors.length) return '#34454c'
  if (colors.length === 1) return colors[0]
  const step = 100 / colors.length
  return `linear-gradient(135deg, ${colors.map((color, index) => `${color} ${index * step}% ${(index + 1) * step}%`).join(', ')})`
}

function EmotionTimeline({ frames, emotionNames }: { frames: EmotionFrame[]; emotionNames: string[] }) {
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null)
  const selected = selectedIndex === null ? null : frames[selectedIndex]
  const names = emotionNames.length ? emotionNames : [...new Set(frames.flatMap(frame => Object.keys(frame.emotions)))]
  return <>
    <h3>Dominant emotion by frame</h3>
    <p className="muted">Each block represents one output frame or sample. Scroll horizontally to inspect the full sequence; hover for a quick summary or select a block for all emotion scores.</p>
    <div className="emotion-legend">{names.map(name => <span key={name}><i style={{ background: EMOTION_COLORS[name] ?? '#66777d' }} />{EMOTION_LABELS[name] ?? name}</span>)}</div>
    <div className="emotion-timeline" role="list" aria-label={`${frames.length} frames in emotion timeline`}>
      {frames.map((item, index) => {
        const names = emotionNames(item.dominant_emotion)
        const label = names.length ? `${emotionLabel(item.dominant_emotion)}${names.length > 1 ? ' (tie)' : ''}` : 'No emotion score'
        const tooltip = `Frame ${item.frame ?? index + 1} · ${item.timestamp ?? 'time unavailable'} · ${label}${item.dominant_score == null ? '' : ` (${item.dominant_score.toFixed(4)})`}`
        return <button key={`${item.frame ?? 'frame'}-${index}`} type="button" className={`emotion-frame${selectedIndex === index ? ' selected' : ''}`} style={{ background: emotionColor(item.dominant_emotion) }} title={tooltip} aria-label={tooltip} aria-pressed={selectedIndex === index} onClick={() => setSelectedIndex(index)}>{names.length > 1 ? '=' : names.length ? (EMOTION_LABELS[names[0]] ?? names[0]).slice(0, 1) : '?'}</button>
      })}
    </div>
    {selected && <div className="emotion-frame-details"><strong>Frame {selected.frame ?? (selectedIndex ?? 0) + 1}</strong><span>Timestamp: {selected.timestamp ?? 'N/A'}</span><span>Dominant emotion: {emotionLabel(selected.dominant_emotion) || 'No score'}{emotionNames(selected.dominant_emotion).length > 1 ? ' (tie)' : ''}</span><span>Dominant value: {selected.dominant_score?.toFixed(4) ?? 'N/A'}</span><div className="emotion-scores">{names.map(name => <span key={name}>{EMOTION_LABELS[name] ?? name}: {selected.emotions[name]?.toFixed(4) ?? 'N/A'}</span>)}</div></div>}
  </>
}

function MetricDropdown({options,selected,onChange}:{options:string[];selected:string[];onChange:(metrics:string[])=>void}) {
  const [open,setOpen]=useState(false)
  return <div className="metric-dropdown"><button type="button" aria-expanded={open} onClick={()=>setOpen(value=>!value)}>{selected.length ? `${selected.length} selected` : 'Choose metrics'} ▾</button>{open&&<div className="metric-dropdown-menu">{options.map(option=><label key={option}><input type="checkbox" checked={selected.includes(option)} onChange={event=>onChange(event.target.checked?[...selected,option]:selected.filter(item=>item!==option))}/>{option}</label>)}</div>}</div>
}

function LineChart({points,metrics,timeColumn,scatter=false,threshold,stimuli=[]}:{points:Record<string,number|null>[];metrics:string[];timeColumn:string;scatter?:boolean;threshold?:number;stimuli?:Stimulus[]}) {
  const [hovered,setHovered]=useState<number|null>(null)
  const metric=metrics[0]; if(!metric)return <p className="muted">Choose at least one signal to plot.</p>
  const values=metrics.flatMap(name=>points.map(p=>p[name]).filter((v):v is number=>typeof v==='number'&&Number.isFinite(v))); if(values.length<2)return <p className="muted">Not enough values to plot.</p>
  const min=Math.min(...values,...(threshold===undefined?[]:[threshold])),max=Math.max(...values,...(threshold===undefined?[]:[threshold])),span=max-min||1
  const times=points.map(p=>typeof p[timeColumn]==='number'?p[timeColumn] as number:null), validTimes=times.filter((v):v is number=>v!==null); if(!validTimes.length)return <p className="muted">Time values are not available for this chart.</p>
  const xMin=Math.min(...validTimes),xMax=Math.max(...validTimes),xSpan=xMax-xMin||1
  const series=metrics.map((name,index)=>({name,color:`hsl(${(index*67+24)%360} 72% 65%)`,xy:points.map((p,i)=>{const v=p[name],t=times[i];return typeof v==='number'&&t!==null?{i,x:((t-xMin)/xSpan)*1000,y:180-((v-min)/span)*160,v,t}:null})}))
  const selected=hovered===null?null:series.flatMap(row=>row.xy.map(point=>point?{...point,name:row.name,color:row.color}:null)).find(point=>point?.i===hovered)
  return <div className="line-chart"><svg viewBox="0 0 1000 230" role="img" aria-label={`${metrics.join(', ')} over time`} onMouseLeave={()=>setHovered(null)}><line x1="0" y1="180" x2="1000" y2="180" stroke="#71847f"/><line x1="0" y1="10" x2="0" y2="180" stroke="#71847f"/>{[0,.5,1].map((t,i)=><g key={`axis-${i}`}><line x1="0" x2="1000" y1={10+t*170} y2={10+t*170} stroke="#394951"/><text x="4" y={7+t*170} fill="#aebdb9" fontSize="11">{(max-t*span).toFixed(2)}</text></g>)}{[0,.5,1].map((t,i)=><g key={`x-tick-${i}`}><line x1={t*1000} y1="180" x2={t*1000} y2="186" stroke="#aebdb9"/><text x={t*1000} y="202" textAnchor={i===0?"start":i===2?"end":"middle"} fill="#c2cfcc" fontSize="12">{(xMin+t*xSpan).toFixed(1)}</text></g>)}<text x="500" y="222" textAnchor="middle" fill="#c2cfcc" fontSize="12">{timeColumn}</text>{stimuli.map((item,i)=>{const start=Math.max(xMin,Math.min(item.inicio,item.fim)),end=Math.min(xMax,Math.max(item.inicio,item.fim));return end>start?<rect key={i} x={(start-xMin)/xSpan*1000} y="10" width={(end-start)/xSpan*1000} height="180" fill="#f09a52" opacity=".10"><title>{item.nome}</title></rect>:null})}{series.map(row=>{const runs:{x:number,y:number}[][]=[];let run:{x:number,y:number}[]=[];row.xy.forEach(point=>{if(point)run.push(point);else if(run.length){runs.push(run);run=[]}});if(run.length)runs.push(run);return <g key={row.name}>{!scatter&&runs.map((line,i)=><polyline key={i} points={line.map(point=>`${point.x},${point.y}`).join(' ')} fill="none" stroke={row.color} strokeWidth="2" vectorEffect="non-scaling-stroke"/>)}{row.xy.map(point=>point&&<circle key={point.i} cx={point.x} cy={point.y} r={scatter?4:2.5} fill={row.color} onMouseEnter={()=>setHovered(point.i)}><title>{`${timeColumn}: ${point.t} · ${row.name}: ${point.v.toFixed(4)}`}</title></circle>)}</g>})}{threshold!==undefined&&<line x1="0" x2="1000" y1={180-((threshold-min)/span)*160} y2={180-((threshold-min)/span)*160} stroke="#ef5350" strokeDasharray="7 5"/>}</svg>{selected&&<div className="chart-tooltip">{timeColumn}: {selected.t} · {selected.name}: {selected.v.toFixed(4)}</div>}<div className="chart-legend">{metrics.map((name,index)=><span key={name} style={{color:`hsl(${(index*67+24)%360} 72% 65%)`}}>● {name}</span>)}</div><div className="chart-axis"><span>X: {timeColumn} ({xMin.toFixed(2)}–{xMax.toFixed(2)})</span><span>Y: shared scale ({min.toFixed(3)}–{max.toFixed(3)})</span></div></div>
}

function Heatmap({points,columns,timeColumn,stimuli=[]}:{points:Record<string,number|null>[];columns:string[];timeColumn:string;stimuli?:Stimulus[]}) {
  const sampled=points.filter((_,i)=>i%Math.max(1,Math.ceil(points.length/80))===0)
  const values=columns.flatMap(column=>sampled.map(point=>point[column]).filter((v):v is number=>typeof v==='number'&&Number.isFinite(v)))
  const max=Math.max(1,...values), times=sampled.map(point=>point[timeColumn]).filter((v):v is number=>typeof v==='number')
  function color(value:number|null){if(value===null)return 'transparent';const t=Math.max(0,Math.min(1,value/max));const r=Math.round(35+220*t),g=Math.round(105-75*t),b=Math.round(225-185*t);return `rgb(${r},${g},${b})`}
  return <div className="heatmap-wrap"><div className="heatmap-y-title">Action Unit</div><div className="heatmap"><div className="heatmap-axis-labels"><span/>{sampled.map((point,i)=>{const stride=Math.max(1,Math.floor(sampled.length/5));return i%stride===0?<small key={i}>{String(point[timeColumn]??'')}</small>:null})}</div>{columns.map(column=><div className="heatmap-row" key={column}><span>{column}</span><div>{sampled.map((point,i)=>{const value=point[column],time=point[timeColumn],stimulus=stimuli.find(item=>typeof time==='number'&&time>=Math.min(item.inicio,item.fim)&&time<Math.max(item.inicio,item.fim));return <i key={i} title={`${column} ? ${timeColumn}: ${time??'N/A'}${stimulus?` ? ${stimulus.nome}`:''} ? Value: ${value??'missing'}`} style={{background:typeof value==='number'?color(value):'transparent',outline:stimulus?'1px solid #ef5350':undefined}}/>})}</div></div>)}<div className="heatmap-x-title">X: {timeColumn}</div></div><div className="heatmap-legend"><span>Low activation</span><i/><span>High activation</span><span>Blue ? red</span></div></div>
}

function StimulusAnalysis({ results }: { results: ResultFile[] }) {
  const names = [...new Set(results.flatMap(result => (result.stimuli ?? []).map(stimulus => stimulus.nome)))]
  const metricOptions = [...new Set(results.flatMap(result => {
    const columns = result.preview.chart_columns ?? { action_units: [], emotions: [], head_pose: [] }
    return [...columns.action_units, ...columns.emotions, ...columns.head_pose]
  }))]
  const [stimulusName, setStimulusName] = useState('')
  const [metric, setMetric] = useState('')
  const selectedStimulus = names.includes(stimulusName) ? stimulusName : names[0]
  const selectedMetric = metricOptions.includes(metric) ? metric : metricOptions[0]
  if (!selectedStimulus || !selectedMetric) return null
  const comparisons = results.flatMap(result => {
    const stimulus = (result.stimuli ?? []).find(item => item.nome === selectedStimulus)
    if (!stimulus) return []
    const summary = result.preview.stimulus_statistics?.[selectedStimulus]?.[selectedMetric]
    if (summary) {
      if (summary.before == null || summary.during == null) return []
      return [{ label: `${result.subject} · ${result.name}`, before: summary.before, during: summary.during }]
    }
    const start = Math.min(stimulus.inicio, stimulus.fim), end = Math.max(stimulus.inicio, stimulus.fim), timeColumn = result.preview.time_column ?? 'frame'
    const samples = (result.preview.chart_points ?? []).flatMap(point => { const time = point[timeColumn], value = point[selectedMetric]; return typeof time === 'number' && typeof value === 'number' ? [{ time, value }] : [] })
    const before = samples.filter(point => point.time >= start - 5 && point.time < start).map(point => point.value)
    const during = samples.filter(point => point.time >= start && point.time < end).map(point => point.value)
    if (!before.length || !during.length) return []
    return [{ label: `${result.subject} · ${result.name}`, before: before.reduce((a,b)=>a+b,0)/before.length, during: during.reduce((a,b)=>a+b,0)/during.length }]
  })
  const all = comparisons.flatMap(item => [item.before,item.during])
  const min = all.length ? Math.min(...all) : 0, max = all.length ? Math.max(...all) : 1, span = max-min || 1
  const y = (value:number) => 170-((value-min)/span)*140
  return <section className="panel"><h2>Before/during stimulus comparison</h2><div className="chart-controls"><select value={selectedStimulus} onChange={event=>setStimulusName(event.target.value)}>{names.map(name=><option key={name}>{name}</option>)}</select><select value={selectedMetric} onChange={event=>setMetric(event.target.value)}>{metricOptions.map(name=><option key={name}>{name}</option>)}</select></div>{comparisons.length ? <><div className="line-chart"><svg viewBox="0 0 1000 200" role="img" aria-label={`${selectedMetric} before and during ${selectedStimulus}`}><text x="100" y="195" fill="#c2cfcc">Before</text><text x="870" y="195" fill="#c2cfcc">During</text>{comparisons.map((item,index)=>{const offset=(index-(comparisons.length-1)/2)*12;return <g key={item.label}><line x1="130" y1={y(item.before)+offset} x2="870" y2={y(item.during)+offset} stroke="#f09a52" strokeOpacity=".55"/><circle cx="130" cy={y(item.before)+offset} r="5" fill="#f09a52"><title>{`${item.label} ? Before: ${item.before.toFixed(3)}`}</title></circle><circle cx="870" cy={y(item.during)+offset} r="5" fill="#f09a52"><title>{`${item.label} ? During: ${item.during.toFixed(3)}`}</title></circle></g>})}</svg><div className="chart-axis"><span>{selectedMetric}: {min.toFixed(3)} ? {max.toFixed(3)}</span><span>Descriptive comparison</span></div></div><ChartInfo text="Each pair of points is one recording: the mean for the 5 seconds before onset and the mean during the stimulus. Vertical distance shows the observed difference. Each recording counts once; this is descriptive, not causal evidence." /></> : <p className="muted">There are not enough observations in the selected windows.</p>}</section>
}

function AdvancedStimulusAnalysis({ results }: { results: ResultFile[] }) {
  const [stimulusChoice, setStimulusChoice] = useState('')
  const [metricChoice, setMetricChoice] = useState('')
  const [hovered, setHovered] = useState<string | null>(null)
  const names = [...new Set(results.flatMap(result => (result.stimuli ?? []).map(item => item.nome)))]
  const metrics = [...new Set(results.flatMap(result => { const c=result.preview.chart_columns ?? {action_units:[],emotions:[],head_pose:[]}; return [...c.action_units,...c.emotions,...c.head_pose] }))]
  const stimulus = names.includes(stimulusChoice) ? stimulusChoice : names[0]
  const metric = metrics.includes(metricChoice) ? metricChoice : metrics[0]
  if (!stimulus || !metric) return null
  const series = results.flatMap(result => {
    const window=(result.stimuli ?? []).find(item=>item.nome===stimulus); if(!window) return []
    const start=Math.min(window.inicio,window.fim), end=Math.max(window.inicio,window.fim), timeCol=result.preview.time_column ?? 'frame'
    const points=(result.preview.chart_points ?? []).flatMap(point=>{const time=point[timeCol],value=point[metric]; return typeof time==='number'&&typeof value==='number'&&time>=start-5&&time<=end+5?[{time:time-start,value}]:[]})
    return points.length?[{label:result.subject||result.name,start:0,end:end-start,points}]:[]
  })
  const values=series.flatMap(row=>row.points.map(point=>point.value)); if(!values.length) return <section className="panel"><h2>Event-aligned trajectories</h2><p className="muted">No data around this stimulus.</p></section>
  const minY=Math.min(...values), maxY=Math.max(...values), ySpan=maxY-minY||1
  const minX=Math.min(...series.flatMap(row=>row.points.map(point=>point.time))), maxX=Math.max(...series.flatMap(row=>row.points.map(point=>point.time))), xSpan=maxX-minX||1
  return <section className="panel"><h2>Advanced across-recording analysis</h2><div className="chart-controls"><select value={stimulus} onChange={event=>setStimulusChoice(event.target.value)}>{names.map(name=><option key={name}>{name}</option>)}</select><select value={metric} onChange={event=>setMetricChoice(event.target.value)}>{metrics.map(name=><option key={name}>{name}</option>)}</select></div><h3>Event-aligned trajectories</h3><div className="line-chart"><svg viewBox="0 0 1000 250" role="img" aria-label={`${metric} around ${stimulus}`} onMouseLeave={()=>setHovered(null)}><line x1="20" y1="200" x2="980" y2="200" stroke="#71847f"/><line x1="20" y1="10" x2="20" y2="200" stroke="#71847f"/>{[0,.5,1].map((t,i)=><g key={`event-y-${i}`}><line x1="20" x2="980" y1={10+t*190} y2={10+t*190} stroke="#394951"/><text x="22" y={8+t*190} fill="#aebdb9" fontSize="11">{(maxY-t*ySpan).toFixed(2)}</text></g>)}{[0,.5,1].map((t,i)=><g key={`event-x-${i}`}><line x1={20+t*960} y1="200" x2={20+t*960} y2="206" stroke="#aebdb9"/><text x={20+t*960} y="220" textAnchor={i===0?"start":i===2?"end":"middle"} fill="#c2cfcc" fontSize="12">{(minX+t*xSpan).toFixed(1)}</text></g>)}<text x="500" y="240" textAnchor="middle" fill="#c2cfcc" fontSize="12">Seconds from stimulus onset</text><rect x={20+((0-minX)/xSpan)*960} y="10" width={Math.max(0,Math.min(960,(Math.max(...series.map(row=>row.end))/xSpan)*960))} height="190" fill="#f09a52" opacity=".10"/><line x1={20+((0-minX)/xSpan)*960} y1="10" x2={20+((0-minX)/xSpan)*960} y2="200" stroke="#f09a52" strokeDasharray="5 4"/>{series.map((row,i)=>{const color=`hsl(${i*67%360} 70% 65%)`;const coords=row.points.map(point=>`${20+((point.time-minX)/xSpan)*960},${200-((point.value-minY)/ySpan)*180}`).join(' ');return <g key={row.label}><polyline points={coords} fill="none" stroke={color} strokeWidth="2" vectorEffect="non-scaling-stroke"/>{row.points.map((point,j)=>{const id=`${i}-${j}`;return <circle key={id} cx={20+((point.time-minX)/xSpan)*960} cy={200-((point.value-minY)/ySpan)*180} r="4" fill={color} onMouseEnter={()=>setHovered(id)}><title>{`${row.label} ? ${point.time.toFixed(2)}s ? ${metric}: ${point.value.toFixed(4)}`}</title></circle>})}</g>})}</svg>{hovered && (()=>{const [i,j]=hovered.split('-').map(Number),row=series[i],point=row?.points[j];return point?<div className="chart-tooltip">{row.label} ? {point.time.toFixed(2)}s ? {metric}: {point.value.toFixed(4)}</div>:null})()}<ChartInfo text="The horizontal axis shows seconds from stimulus onset: 0 is onset and the shaded band is its duration. The vertical axis shows the selected signal. Each line is one recording; gaps indicate missing data. Compare shapes and differences across participants."/><div className="chart-axis"><span>{minX.toFixed(1)}s</span><span>Seconds from stimulus onset</span><span>{maxX.toFixed(1)}s</span></div>{series.map((row,i)=><span className="legend-item" key={row.label} style={{color:`hsl(${i*67%360} 70% 65%)`}}>? {row.label}</span>)}</div></section>
}

function ChartInfo({text}:{text:string}) { return <details className="chart-info"><summary>? About this chart</summary><p>{text}</p></details> }

function IntroPage({ onStart }: { onStart: () => void }) {
  const guide = introGuide.replace('data:image/jpeg;base64,AU_REFERENCE_IMAGE', '/Au-Reference.jpg')
  return <main className="intro-shell">
    <section className="intro-hero"><h1>FEDET</h1><p className="hero-subtitle">Facial Expression Data Extraction Tool</p><p className="hero-powered-by">powered by py-feat</p></section>
    <div className="guide-layout"><aside className="guide-sidebar" dangerouslySetInnerHTML={{ __html: guide.split('<div class="guide-content">')[0] }} /><article className="guide-article" dangerouslySetInnerHTML={{ __html: `<div class="guide-content">${guide.split('<div class="guide-content">')[1]}` }} /></div>
    <button className="primary intro-start" onClick={onStart}>Analysis Tool</button>
  </main>
}
