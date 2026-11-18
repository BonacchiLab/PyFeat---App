import { useEffect, useState } from 'react'
import introGuide from './introGuide.html?raw'

const API = 'http://127.0.0.1:8000'
type Stimulus = { nome: string; inicio: number; fim: number }
type VideoEntry = { file: File; subject: string; stimuli: Stimulus[]; fps?: number; frameCount?: number; duration?: number }
type NamedStimulus = { id: number; name: string }
type PreviewData = { columns: string[]; rows: number; preview: Record<string, unknown>[]; face_score_median: number | null; face_score_missing: number | null; face_score_flagged: number | null; detected_frames: number; expected_rows: number; time_column: string | null; chart_points: Record<string, number | null>[]; chart_columns: { action_units: string[]; emotions: string[]; head_pose: string[] } }
type ResultFile = { name: string; url: string; preview: PreviewData; stimuli: Stimulus[]; subject: string; taskId: string }

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

  function addFiles(files: FileList | null) {
    if (!files) return
    const additions = Array.from(files).filter(file => !videos.some(v => v.file.name === file.name)).map(file => ({ file, subject: '', stimuli: stimuli.map((item, i) => ({ nome: item.name, inicio: 30 * (i + 1), fim: 90 * (i + 1) })) }))
    setVideos(current => [...current, ...additions])
    additions.forEach(async video => {
      const form = new FormData(); form.append('file', video.file)
      try { const response = await fetch(`${API}/api/extractions/metadata`, { method: 'POST', body: form }); if (response.ok) { const info = await response.json(); setVideos(current => current.map(item => item.file.name === video.file.name ? { ...item, fps: info.fps, frameCount: info.frame_count, duration: info.duration_seconds } : item)) } } catch { /* metadata remains optional when the API is unavailable */ }
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
    setBusy(true); setDownloads([])
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
          task = await status.json()
          setMessage(`${video.file.name}: ${task.message} (${Math.round(task.progress * 100)}%)`)
          if (task.status === 'failed') throw new Error(task.error || 'Extraction failed')
        } while (task.status !== 'completed')
        const response = await fetch(`${API}/api/extractions/${task_id}/download`)
        if (!response.ok) throw new Error('Could not download the Parquet result.')
        const previewResponse = await fetch(`${API}/api/extractions/${task_id}/preview?limit=50&threshold=0.9`)
        if (!previewResponse.ok) throw new Error(`Could not load data preview (${previewResponse.status}): ${await previewResponse.text()}`)
        const preview: PreviewData = await previewResponse.json()
        results.push({ name: `${video.subject}_${video.file.name.replace(/\.[^.]+$/, '')}_pyfeat.parquet`, url: URL.createObjectURL(await response.blob()), preview, stimuli: videoStimuli, subject: video.subject, taskId: task_id })
        const elapsedSeconds = (performance.now() - videoStartedAt) / 1000
        setRuntimeHistory(current => {
          const previous = current[mode] ?? { rows: 0, seconds: 0 }
          return { ...current, [mode]: { rows: previous.rows + (preview.expected_rows || preview.rows), seconds: previous.seconds + elapsedSeconds } }
        })
      }
      setDownloads(results); setMessage('All videos processed successfully.')
    } catch (error) { setMessage(error instanceof Error ? error.message : 'Unexpected error.') }
    finally { setBusy(false) }
  }

  const estimatedRows = videos.reduce((sum, video) => {
    const frames = video.frameCount ?? 0, fps = video.fps ?? 30
    const skip = mode === 'all_frames' ? 1 : Math.max(1, Math.round(fps / (mode === 'custom_fps' ? targetFps : 1)))
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
    {page === 'introduction' ? <IntroPage onStart={() => setPage('workspace')} /> : <div className="workspace-layout"><aside className="sampling-sidebar"><h2>Extraction controls</h2><label>Frame sampling<select value={mode} onChange={event=>setMode(event.target.value)}><option value="all_frames">Every frame (Maximum resolution)</option><option value="fast_fps">1 FPS (representative frame)</option><option value="custom_fps">Custom sampling</option></select></label>{mode==='custom_fps'&&<label>Target sampling frequency (FPS)<input type="number" min="0.1" step="0.1" value={targetFps} onChange={event=>setTargetFps(Number(event.target.value))}/></label>}<hr/><h3>Processing time estimate</h3>{videos.length?<><div className="sidebar-metric"><span>Frames to analyze</span><strong>{estimatedRows.toLocaleString()}</strong></div><div className="sidebar-metric"><span>Output rows</span><strong>{estimatedRows.toLocaleString()}</strong></div><div className="sidebar-metric"><span>Estimated time</span><strong>{Math.floor(estimatedSeconds/60)}m {Math.round(estimatedSeconds%60)}s</strong></div>{measuredRows===0&&<small>This is an initial estimate; it improves after extraction.</small>}</>:<p className="muted">Upload videos to calculate the estimate.</p>}</aside><main className="shell workspace-main">
    <section className="hero"><p className="eyebrow">LOCAL RESEARCH WORKSPACE</p><h1>Turn video into<br/><em>facial expression data.</em></h1><p>Upload recordings, choose your sampling settings, and extract Py-Feat predictions.</p></section>
    <section className="panel"><h2>1. Videos and subjects</h2><label className="drop">Choose video files<input type="file" accept=".mp4,.avi,.mov" multiple onChange={event => addFiles(event.target.files)}/><small>MP4, AVI, MOV · Videos are processed locally</small></label>
      {videos.map((video, index) => <div className="video" key={`${video.file.name}-${index}`}><div><strong>{video.file.name}</strong><small className="video-meta">{video.fps ? `${video.frameCount?.toLocaleString()} frames · ${video.fps.toFixed(2)} FPS · ${Math.floor((video.duration ?? 0)/60)}:${String(Math.floor((video.duration ?? 0)%60)).padStart(2,'0')}` : 'Reading video metadata…'}</small></div><label>Subject ID<input value={video.subject} onChange={event => updateVideo(index, { subject: event.target.value })} placeholder="e.g. Subject_01"/></label></div>)}
    </section>
    <section className="panel"><h2>2. Experimental design metadata</h2>
      <label className="inline"><input type="checkbox" checked={addMetadata} onChange={event => setAddMetadata(event.target.checked)}/> Add stimulus markers to the dataset</label>
      {addMetadata && <>
        <label>Number of stimuli<input type="number" min="1" max="20" value={stimuli.length} onChange={event => setStimulusCount(Math.max(1, Number(event.target.value) || 1))}/></label>
        {stimuli.map((stimulus, i) => <label key={stimulus.id}>{`Stimulus ${i + 1} name`}<input value={stimulus.name} onChange={event => renameStimulus(i, event.target.value)}/></label>)}
        <label className="inline"><input type="checkbox" checked={shared} onChange={event => setShared(event.target.checked)}/> Use the same stimulus windows for every video</label>
        {shared ? sharedWindows.map((window, i) => <fieldset key={i}><legend>{stimuli[i].name}</legend><div className="window-fields"><label>Start (seconds)<input type="number" min="0" value={window.inicio} onChange={event => updateWindow(0, i, 'inicio', Number(event.target.value), true)}/></label><label>End (seconds)<input type="number" min="0" value={window.fim} onChange={event => updateWindow(0, i, 'fim', Number(event.target.value), true)}/></label></div></fieldset>) : videos.map((video, videoIndex) => <details key={video.file.name}><summary>Stimulus windows — {video.subject || video.file.name}</summary>{video.stimuli.map((window, i) => <fieldset key={i}><legend>{stimuli[i].name}</legend><div className="window-fields"><label>Start (seconds)<input type="number" min="0" value={window.inicio} onChange={event => updateWindow(videoIndex, i, 'inicio', Number(event.target.value), false)}/></label><label>End (seconds)<input type="number" min="0" value={window.fim} onChange={event => updateWindow(videoIndex, i, 'fim', Number(event.target.value), false)}/></label></div></fieldset>)}</details>)}
      </>}
    </section>
    <button className="primary" disabled={busy} onClick={extract}>{busy ? 'Processing…' : 'Extract data'}</button><p className="status" role="status">{message}</p>
    {downloads.length > 0 && <StimulusAnalysis results={downloads} />}
    {downloads.length > 0 && <AdvancedStimulusAnalysis results={downloads} />}
    {downloads.map(result => <ResultCard key={result.name} result={result} />)}
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
  const selectedMetric = activeMetrics[0] ?? available[0]
  useEffect(() => {
    fetch(`${API}/api/extractions/${result.taskId}/preview?limit=1&threshold=${threshold}`).then(response => response.ok ? response.json() : null).then(data => { if (data) setFlaggedScores(data.face_score_flagged) }).catch(() => undefined)
  }, [result.taskId, threshold])
  return <section className="panel results"><h2>Extraction complete — {result.name}</h2><div className="quality-cards"><div><small>Output rows</small><strong>{result.preview.rows.toLocaleString()}</strong></div><div><small>Median FaceScore</small><strong>{result.preview.face_score_median?.toFixed(3) ?? 'N/A'}</strong></div><div><small>Missing FaceScore</small><strong>{result.preview.face_score_missing?.toLocaleString() ?? 'N/A'}</strong></div></div>
    <h3>Data quality</h3><div className="quality-cards"><div><small>Frames with detections</small><strong>{result.preview.detected_frames.toLocaleString()} / {result.preview.expected_rows.toLocaleString()}</strong></div><div><small>Detection coverage</small><strong>{result.preview.expected_rows ? `${Math.min(result.preview.detected_frames/result.preview.expected_rows,1).toLocaleString(undefined,{style:"percent",maximumFractionDigits:1})}` : "N/A"}</strong></div><div><small>FaceScores below limit</small><strong>{flaggedScores?.toLocaleString() ?? "N/A"}</strong></div></div>{chartPoints.some(point => point.FaceScore != null) ? <><label>Flag FaceScore below {threshold.toFixed(2)}<input type="range" min="0" max="1" step="0.05" value={threshold} onChange={event=>setThreshold(Number(event.target.value))}/></label><LineChart points={chartPoints} metric="FaceScore" timeColumn={result.preview.time_column ?? "frame"} scatter threshold={threshold}/><ChartInfo text="Each point is a frame-level detection. The horizontal axis is time or frame; the vertical axis is FaceScore (0?1). The red line is your review threshold. A low score calls for image inspection; it does not tell you whether an AU or emotion is correct."/></> : <p className="muted">FaceScore is not available.</p>}
    <h3>Signals over time</h3><div className="chart-controls"><select value={category} onChange={event=>{setCategory(event.target.value as typeof category);setSelectedMetrics([])}}><option value="action_units">Action Units</option><option value="emotions">Emotions</option><option value="head_pose">Head pose</option></select>{available.length>0&&<select multiple value={activeMetrics.length?activeMetrics:[available[0]]} onChange={event=>setSelectedMetrics(Array.from(event.currentTarget.selectedOptions,option=>option.value))}>{available.map(column=><option key={column}>{column}</option>)}</select>}</div>{available.length?<>{(activeMetrics.length?activeMetrics:[available[0]]).map(name=><LineChart key={name} points={chartPoints} metric={name} timeColumn={result.preview.time_column??"frame"} stimuli={result.stimuli}/>)}<ChartInfo text="The horizontal axis is time or frame; the vertical axis is the model output. Choose AUs, emotions, or pose in the controls. Shaded bands mark stimuli. Compare trends, not unlike scales; these values are model estimates."/></>:<p className="muted">No signals in this category.</p>}
    {chartColumns.action_units.length>0&&<><h3>AU heatmap</h3><Heatmap points={chartPoints} columns={chartColumns.action_units} timeColumn={result.preview.time_column??"frame"} stimuli={result.stimuli}/><ChartInfo text="Each row is an AU and each column is a sampled frame. Color shows the model estimate; blank cells mean missing data, not zero. Blue indicates lower activation and red indicates higher activation. Shaded bands mark stimuli and help locate changes over time."/></>}
    <h3>Data preview (first 50 rows)</h3><div className="table-scroll"><table><thead><tr>{result.preview.columns.map(column => <th key={column}>{column}</th>)}</tr></thead><tbody>{result.preview.preview.map((row, i) => <tr key={i}>{result.preview.columns.map(column => <td key={column}>{row[column] == null ? '' : typeof row[column] === 'object' ? JSON.stringify(row[column]) : String(row[column])}</td>)}</tr>)}</tbody></table></div><a className="download" href={result.url} download={result.name}>Download Parquet</a>
  </section>
}

function LineChart({points,metric,timeColumn,scatter=false,threshold,stimuli=[]}:{points:Record<string,number|null>[];metric:string;timeColumn:string;scatter?:boolean;threshold?:number;stimuli?:Stimulus[]}) {
  const [hovered,setHovered]=useState<number|null>(null)
  const values=points.map(p=>p[metric]).filter((v):v is number=>typeof v==='number'&&Number.isFinite(v)); if(values.length<2)return <p className="muted">Not enough values to plot.</p>
  const min=Math.min(...values,...(threshold===undefined?[]:[threshold])),max=Math.max(...values,...(threshold===undefined?[]:[threshold])),span=max-min||1
  const times=points.map(p=>typeof p[timeColumn]==='number'?p[timeColumn] as number:null), validTimes=times.filter((v):v is number=>v!==null),xMin=Math.min(...validTimes),xMax=Math.max(...validTimes),xSpan=xMax-xMin||1
  const xy=points.map((p,i)=>{const v=p[metric],t=times[i];return typeof v==='number'&&t!==null?{i,x:((t-xMin)/xSpan)*1000,y:180-((v-min)/span)*160,v,t}:null})
  const runs:{i:number,x:number,y:number,v:number,t:number}[][]=[];let run:{i:number,x:number,y:number,v:number,t:number}[]=[];xy.forEach(point=>{if(point){run.push(point)}else if(run.length){runs.push(run);run=[]}});if(run.length)runs.push(run)
  const selected=xy.find(p=>p?.i===hovered)
  return <div className="line-chart"><svg viewBox="0 0 1000 230" role="img" aria-label={`${metric} over time`} onMouseLeave={()=>setHovered(null)}><line x1="0" y1="180" x2="1000" y2="180" stroke="#71847f"/><line x1="0" y1="10" x2="0" y2="180" stroke="#71847f"/>{[0,.5,1].map((t,i)=><g key={`axis-${i}`}><line x1="0" x2="1000" y1={10+t*170} y2={10+t*170} stroke="#394951"/><text x="4" y={7+t*170} fill="#aebdb9" fontSize="11">{(max-t*span).toFixed(2)}</text></g>)}{[0,.5,1].map((t,i)=><g key={`x-tick-${i}`}><line x1={t*1000} y1="180" x2={t*1000} y2="186" stroke="#aebdb9"/><text x={t*1000} y="202" textAnchor={i===0?"start":i===2?"end":"middle"} fill="#c2cfcc" fontSize="12">{(xMin+t*xSpan).toFixed(1)}</text></g>)}<text x="500" y="222" textAnchor="middle" fill="#c2cfcc" fontSize="12">{timeColumn}</text>{stimuli.map((item,i)=>{const start=Math.max(xMin,Math.min(item.inicio,item.fim)),end=Math.min(xMax,Math.max(item.inicio,item.fim));return end>start?<rect key={i} x={(start-xMin)/xSpan*1000} y="10" width={(end-start)/xSpan*1000} height="180" fill="#f09a52" opacity=".10"><title>{item.nome}</title></rect>:null})}{!scatter&&runs.map((r,i)=><polyline key={i} points={r.map(p=>`${p.x},${p.y}`).join(' ')} fill="none" stroke="#f09a52" strokeWidth="2" vectorEffect="non-scaling-stroke"/>)}{threshold!==undefined&&<line x1="0" x2="1000" y1={180-((threshold-min)/span)*160} y2={180-((threshold-min)/span)*160} stroke="#ef5350" strokeDasharray="7 5"/>}{xy.map(p=>p&&<circle key={p.i} cx={p.x} cy={p.y} r={scatter?4:2.5} fill={scatter?'#f09a52':'#edf3f1'} onMouseEnter={()=>setHovered(p.i)}><title>{`${timeColumn}: ${p.t} ? ${metric}: ${p.v.toFixed(4)}`}</title></circle>)}</svg>{selected&&<div className="chart-tooltip">{timeColumn}: {selected.t} ? {metric}: {selected.v.toFixed(4)}</div>}<div className="chart-axis"><span>X: {timeColumn} ({xMin.toFixed(2)}?{xMax.toFixed(2)})</span><span>Y: {metric} ({min.toFixed(3)} ? {max.toFixed(3)})</span></div></div>
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
    const start = Math.min(stimulus.inicio, stimulus.fim), end = Math.max(stimulus.inicio, stimulus.fim)
    const timeColumn = result.preview.time_column ?? 'frame'
    const samples = (result.preview.chart_points ?? []).flatMap(point => {
      const time = point[timeColumn], value = point[selectedMetric]
      return typeof time === 'number' && typeof value === 'number' ? [{ time, value }] : []
    })
    const before = samples.filter(point => point.time >= start - 5 && point.time < start).map(point => point.value)
    const during = samples.filter(point => point.time >= start && point.time < end).map(point => point.value)
    if (!before.length || !during.length) return []
    return [{ label: `${result.subject} ? ${result.name}`, before: before.reduce((a,b)=>a+b,0)/before.length, during: during.reduce((a,b)=>a+b,0)/during.length }]
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
