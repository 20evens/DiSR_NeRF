import { useState, useRef, useEffect, useCallback } from 'react'
import { api } from '../api.js'

/* ── tiny helpers ─────────────────────────────────────────────────────────── */
function cls(...args) { return args.filter(Boolean).join(' ') }

function StatusBadge({ status }) {
  const map = {
    idle:    ['bg-gray-100 text-gray-500',  '待机'],
    running: ['bg-blue-100 text-blue-600',  '运行中'],
    done:    ['bg-green-100 text-green-700','完成'],
    error:   ['bg-red-100 text-red-600',    '错误'],
  }
  const [c, label] = map[status] || map.idle
  return (
    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${c}`}>
      {status === 'running' && <span className="mr-1 animate-pulse">●</span>}
      {label}
    </span>
  )
}

/* ── File Browser Modal ───────────────────────────────────────────────────── */
function FileBrowser({ open, onClose, onSelect }) {
  const [current, setCurrent]   = useState('')
  const [entries, setEntries]   = useState([])
  const [parent, setParent]     = useState(null)
  const [loading, setLoading]   = useState(false)
  const [err, setErr]           = useState('')

  const load = useCallback(async (path = '') => {
    setLoading(true); setErr('')
    try {
      const data = await api.browse(path)
      setCurrent(data.current)
      setParent(data.parent)
      setEntries(data.entries)
    } catch (e) { setErr(e.message) }
    finally { setLoading(false) }
  }, [])

  useEffect(() => { if (open) load('') }, [open, load])

  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
      <div className="bg-white rounded-2xl shadow-2xl w-[560px] max-h-[70vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b">
          <h3 className="font-semibold text-gray-800">📁 浏览文件夹</h3>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600 text-xl leading-none">×</button>
        </div>

        {/* Path bar */}
        <div className="px-5 py-2 bg-gray-50 text-xs text-gray-500 font-mono border-b truncate">
          {current || '根目录'}
        </div>

        {/* Entries */}
        <div className="overflow-y-auto flex-1 px-3 py-2">
          {loading && <p className="text-center py-8 text-gray-400 text-sm">加载中…</p>}
          {err    && <p className="text-center py-4 text-red-500 text-sm">{err}</p>}
          {!loading && !err && (
            <ul className="space-y-0.5">
              {parent && (
                <li>
                  <button
                    onClick={() => load(parent)}
                    className="w-full text-left px-3 py-2 rounded-lg hover:bg-gray-100 text-sm flex items-center gap-2 text-blue-500"
                  >
                    ⬆ 上级目录
                  </button>
                </li>
              )}
              {entries.map(e => (
                <li key={e.path}>
                  <button
                    onClick={() => e.is_dir ? load(e.path) : undefined}
                    className={cls(
                      'w-full text-left px-3 py-2 rounded-lg text-sm flex items-center gap-2 transition',
                      e.is_dir ? 'hover:bg-indigo-50 text-gray-800' : 'text-gray-400 cursor-default'
                    )}
                  >
                    {e.is_dir ? '📁' : '📄'} {e.name}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* Footer */}
        <div className="px-5 py-3 border-t flex items-center gap-3">
          <span className="flex-1 text-xs text-gray-500 font-mono truncate">{current}</span>
          <button
            onClick={() => { onSelect(current); onClose() }}
            disabled={!current}
            className="px-4 py-1.5 bg-indigo-600 text-white text-sm rounded-lg hover:bg-indigo-700 disabled:opacity-40 transition"
          >
            选择此文件夹
          </button>
        </div>
      </div>
    </div>
  )
}

/* ── Progress Bar ─────────────────────────────────────────────────────────── */
function ProgressBar({ value }) {
  return (
    <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
      <div
        className="h-3 rounded-full bg-gradient-to-r from-indigo-500 to-purple-500 transition-all duration-500"
        style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
      />
    </div>
  )
}

/* ── Image Preview Grid ───────────────────────────────────────────────────── */
function PreviewGrid({ path }) {
  const [images, setImages] = useState([])
  const [loading, setLoading] = useState(false)
  const [msg, setMsg] = useState('')

  async function load() {
    if (!path) return setMsg('请先选择数据集目录')
    setLoading(true); setImages([]); setMsg('')
    try {
      const d = await api.preview(path)
      if (d.images.length === 0) setMsg('未找到图像文件')
      else setImages(d.images)
    } catch (e) { setMsg(e.message) }
    finally { setLoading(false) }
  }

  return (
    <div>
      <button
        onClick={load}
        className="text-xs text-indigo-600 hover:underline flex items-center gap-1"
      >
        🖼 预览数据集（前6张）
      </button>
      {loading && <p className="text-xs text-gray-400 mt-1">加载中…</p>}
      {msg     && <p className="text-xs text-gray-400 mt-1">{msg}</p>}
      {images.length > 0 && (
        <div className="mt-2 grid grid-cols-3 gap-1.5">
          {images.map(img => (
            <div key={img.name} className="relative rounded-md overflow-hidden border border-gray-200 bg-gray-50" style={{paddingTop:'75%'}}>
              <img
                src={img.data} alt={img.name}
                className="absolute inset-0 w-full h-full object-cover"
              />
              <span className="absolute bottom-0 left-0 right-0 bg-black/50 text-white text-[10px] px-1 truncate">
                {img.name}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

/* ── Path Input Row ───────────────────────────────────────────────────────── */
function PathRow({ label, value, onChange, placeholder, onBrowse, children }) {
  return (
    <div>
      <label className="block text-xs font-medium text-gray-600 mb-1">{label}</label>
      <div className="flex gap-2">
        <input
          type="text"
          value={value}
          onChange={e => onChange(e.target.value)}
          placeholder={placeholder}
          className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-xs focus:outline-none focus:ring-2 focus:ring-indigo-400"
        />
        <button
          onClick={onBrowse}
          className="px-3 py-2 border border-gray-300 rounded-lg text-xs hover:bg-gray-50 whitespace-nowrap transition"
        >
          📁 浏览
        </button>
      </div>
      {children}
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* Main Dashboard                                                             */
/* ═══════════════════════════════════════════════════════════════════════════ */
export default function Dashboard({ user, onLogout }) {
  /* Config state */
  const [mode,        setMode]        = useState('sr_nerf')  // sr | nerf | sr_nerf
  const [inputDir,    setInputDir]    = useState('')
  const [outputDir,   setOutputDir]   = useState('')
  const [autoOutput,  setAutoOutput]  = useState(true)
  const [advOpen,     setAdvOpen]     = useState(false)
  const [ddimSteps,   setDdimSteps]   = useState(10)
  const [nerfIters,   setNerfIters]   = useState(200000)
  const [srScale,     setSrScale]     = useState('2x')

  /* File browser */
  const [browserOpen,   setBrowserOpen]   = useState(false)
  const [browserTarget, setBrowserTarget] = useState('input')  // 'input' | 'output'

  /* Job state */
  const [jobId,    setJobId]    = useState(null)
  const [status,   setStatus]   = useState('idle')     // idle | running | done | error
  const [progress, setProgress] = useState(0)
  const [logs,     setLogs]     = useState([])
  const [videoPath, setVideoPath] = useState(null)

  const logRef   = useRef(null)
  const esRef    = useRef(null)

  /* Auto-scroll log */
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [logs])

  /* Cleanup SSE on unmount */
  useEffect(() => () => esRef.current?.close(), [])

  /* Auto-generate output path */
  useEffect(() => {
    if (autoOutput && inputDir) {
      const ts   = new Date().toISOString().slice(0,16).replace('T','_').replace(':','-')
      const base = inputDir.replace(/\\/g, '/').split('/').filter(Boolean).pop() || 'output'
      setOutputDir(`./results/${base}_${mode}_${ts}`)
    }
  }, [autoOutput, inputDir, mode])

  /* Open browser for correct target */
  function openBrowser(target) {
    setBrowserTarget(target)
    setBrowserOpen(true)
  }

  function handleBrowserSelect(path) {
    if (browserTarget === 'input') {
      setInputDir(path)
    } else {
      setOutputDir(path)
      setAutoOutput(false)
    }
  }

  /* Start job */
  async function handleStart() {
    if (!inputDir.trim())  return alert('请输入数据集目录')
    if (!outputDir.trim()) return alert('请输入结果目录')

    setLogs([])
    setProgress(0)
    setVideoPath(null)
    setStatus('running')

    try {
      const data = await api.runProcess({
        mode,
        input_dir:  inputDir,
        output_dir: outputDir,
        ddim_steps: ddimSteps,
        nerf_iters: nerfIters,
        sr_scale:   srScale,
      })
      setJobId(data.job_id)
      startSSE(data.job_id)
    } catch (e) {
      setStatus('error')
      setLogs([`[错误] ${e.message}`])
    }
  }

  /* SSE streaming */
  function startSSE(id) {
    esRef.current?.close()
    const es = api.streamLogs(id)
    esRef.current = es

    es.onmessage = (e) => {
      const msg = JSON.parse(e.data)
      if (msg.type === 'log') {
        setLogs(prev => [...prev, msg.data])
        setProgress(msg.progress ?? 0)
      } else if (msg.type === 'done') {
        setProgress(msg.progress ?? 100)
        setStatus(msg.success ? 'done' : 'error')
        es.close()
        if (msg.success && msg.video_path) {
          setVideoPath(msg.video_path)
        } else if (msg.success) {
          // Try to find video in output dir
          api.findVideo(outputDir).then(r => {
            if (r.path) setVideoPath(r.path)
          }).catch(() => {})
        }
      }
    }

    es.onerror = () => {
      setStatus('error')
      setLogs(prev => [...prev, '[连接断开] SSE 连接已中断'])
      es.close()
    }
  }

  /* Re-run: reset to config state */
  function handleRerun() {
    esRef.current?.close()
    setJobId(null)
    setStatus('idle')
    setProgress(0)
    setLogs([])
    setVideoPath(null)
  }

  const isRunning = status === 'running'
  const isDone    = status === 'done' || status === 'error'

  const modeOptions = [
    { value: 'sr',      label: '图像超分辨率',           icon: '🔍' },
    { value: 'nerf',    label: '三维重建',               icon: '🏗' },
    { value: 'sr_nerf', label: '超分辨率 + 三维重建',    icon: '⚡' },
  ]

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col">
      {/* ── Nav ── */}
      <nav className="bg-white border-b border-gray-200 px-6 py-3 flex items-center justify-between shadow-sm">
        <div className="flex items-center gap-3">
          <span className="text-xl">🎬</span>
          <span className="font-bold text-gray-800 text-lg">NeRF SR Studio</span>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-sm text-gray-500">👤 {user.username}</span>
          <StatusBadge status={status} />
          <button
            onClick={onLogout}
            className="text-sm text-gray-500 hover:text-red-500 transition"
          >
            退出
          </button>
        </div>
      </nav>

      {/* ── Main Layout ── */}
      <div className="flex flex-1 overflow-hidden" style={{ height: 'calc(100vh - 57px)' }}>

        {/* ══ LEFT: Config Panel (35%) ══════════════════════════════════════ */}
        <aside className="w-[35%] min-w-[300px] bg-white border-r border-gray-200 overflow-y-auto p-5 flex flex-col gap-5">

          {/* Advanced Settings (collapsible) */}
          <div className="border border-gray-200 rounded-xl overflow-hidden">
            <button
              onClick={() => setAdvOpen(o => !o)}
              className="w-full flex items-center justify-between px-4 py-3 text-sm font-medium text-gray-700 hover:bg-gray-50 transition"
            >
              <span>⚙ 高级参数设置</span>
              <span className="text-gray-400">{advOpen ? '▲' : '▼'}</span>
            </button>
            {advOpen && (
              <div className="px-4 pb-4 pt-2 space-y-3 border-t border-gray-100">
                <div>
                  <label className="block text-xs font-medium text-gray-600 mb-1">
                    超分模型倍率
                    <span className="ml-1 text-gray-400 font-normal">(取决于已训练的模型)</span>
                  </label>
                  <div className="flex gap-2">
                    {['2x', '4x'].map(v => (
                      <button
                        key={v}
                        onClick={() => setSrScale(v)}
                        className={cls(
                          'flex-1 py-1.5 rounded-lg text-xs border font-medium transition',
                          srScale === v
                            ? 'bg-indigo-600 border-indigo-600 text-white'
                            : 'border-gray-300 text-gray-600 hover:border-indigo-400'
                        )}
                      >
                        {v}
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="block text-xs font-medium text-gray-600 mb-1">
                    DDIM 步数
                    <span className="ml-1 text-gray-400 font-normal">(越少越快，质量略降)</span>
                  </label>
                  <input
                    type="number"
                    min={1} max={1000} step={1}
                    value={ddimSteps}
                    onChange={e => setDdimSteps(Number(e.target.value))}
                    className="w-full px-3 py-1.5 border border-gray-300 rounded-lg text-xs focus:outline-none focus:ring-2 focus:ring-indigo-400"
                  />
                </div>

                <div>
                  <label className="block text-xs font-medium text-gray-600 mb-1">
                    NeRF 训练迭代次数
                  </label>
                  <input
                    type="number"
                    min={1000} max={1000000} step={10000}
                    value={nerfIters}
                    onChange={e => setNerfIters(Number(e.target.value))}
                    className="w-full px-3 py-1.5 border border-gray-300 rounded-lg text-xs focus:outline-none focus:ring-2 focus:ring-indigo-400"
                  />
                </div>
              </div>
            )}
          </div>

          {/* Mode selection */}
          <div>
            <p className="text-xs font-medium text-gray-600 mb-2">运行模式</p>
            <div className="space-y-2">
              {modeOptions.map(opt => (
                <label
                  key={opt.value}
                  className={cls(
                    'flex items-center gap-3 px-3 py-2.5 rounded-xl border cursor-pointer transition',
                    mode === opt.value
                      ? 'border-indigo-500 bg-indigo-50'
                      : 'border-gray-200 hover:border-indigo-300'
                  )}
                >
                  <input
                    type="radio"
                    name="mode"
                    value={opt.value}
                    checked={mode === opt.value}
                    onChange={() => setMode(opt.value)}
                    className="accent-indigo-600"
                  />
                  <span className="text-lg">{opt.icon}</span>
                  <span className="text-sm text-gray-700 font-medium">{opt.label}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Dataset input */}
          <PathRow
            label="数据集目录"
            value={inputDir}
            onChange={setInputDir}
            placeholder="选择或输入数据集路径…"
            onBrowse={() => openBrowser('input')}
          >
            <div className="mt-2">
              <PreviewGrid path={inputDir} />
            </div>
          </PathRow>

          {/* Output path */}
          <PathRow
            label="结果存放路径"
            value={outputDir}
            onChange={v => { setOutputDir(v); setAutoOutput(false) }}
            placeholder="选择或输入输出路径…"
            onBrowse={() => openBrowser('output')}
          >
            <label className="flex items-center gap-2 mt-1.5 cursor-pointer">
              <input
                type="checkbox"
                checked={autoOutput}
                onChange={e => setAutoOutput(e.target.checked)}
                className="accent-indigo-600"
              />
              <span className="text-xs text-gray-500">自动生成路径</span>
            </label>
          </PathRow>

          {/* Start button */}
          <button
            onClick={handleStart}
            disabled={isRunning}
            className={cls(
              'w-full py-3 rounded-xl font-semibold text-sm transition focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2',
              isRunning
                ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                : 'bg-indigo-600 hover:bg-indigo-700 text-white shadow-md'
            )}
          >
            {isRunning ? '⏳ 运行中…' : '▶ 开始运行'}
          </button>

        </aside>

        {/* ══ RIGHT: Log & Results (65%) ════════════════════════════════════ */}
        <main className="flex-1 flex flex-col overflow-hidden bg-gray-50 p-4 gap-3">

          {/* Progress bar */}
          {(isRunning || isDone) && (
            <div className="bg-white rounded-xl p-4 shadow-sm border border-gray-200">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-gray-700">
                  {isRunning ? '处理进度' : (status === 'done' ? '✅ 处理完成' : '❌ 处理失败')}
                </span>
                <span className="text-sm font-mono text-indigo-600">
                  {Math.round(progress)}%
                </span>
              </div>
              <ProgressBar value={progress} />
              {isRunning && (
                <p className="text-xs text-gray-400 mt-1.5">
                  {progress < 20 ? '超分辨率阶段…' : progress < 90 ? 'NeRF 三维重建…' : '渲染结果…'}
                </p>
              )}
            </div>
          )}

          {/* Log terminal */}
          <div className="flex-1 flex flex-col bg-[#1e1e2e] rounded-xl shadow-sm border border-gray-700 overflow-hidden min-h-0">
            <div className="flex items-center justify-between px-4 py-2 bg-[#181825] border-b border-gray-700">
              <span className="text-xs text-gray-400 font-mono">📋 实时日志</span>
              {logs.length > 0 && (
                <button
                  onClick={() => setLogs([])}
                  className="text-xs text-gray-500 hover:text-gray-300 transition"
                >
                  清空
                </button>
              )}
            </div>
            <div
              ref={logRef}
              className="flex-1 overflow-y-auto p-4 log-terminal log-scroll"
            >
              {logs.length === 0 ? (
                <p className="text-gray-600 text-xs">等待运行…</p>
              ) : (
                logs.map((line, i) => (
                  <div key={i} className={cls(
                    'text-xs leading-relaxed whitespace-pre-wrap break-all',
                    line.startsWith('[错误]') || line.startsWith('[异常]') || line.startsWith('[OOM]')
                      ? 'text-red-400'
                      : line.startsWith('[完成]')
                        ? 'text-green-400'
                        : line.startsWith('[启动]') || line.startsWith('[SR]') || line.startsWith('[NeRF]')
                          ? 'text-blue-300'
                          : 'text-gray-300'
                  )}>
                    {line}
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Video player */}
          {videoPath && (
            <div className="bg-white rounded-xl p-4 shadow-sm border border-gray-200">
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-semibold text-gray-800 text-sm">🎬 三维重建结果</h3>
                <span className="text-xs text-gray-400 font-mono truncate max-w-[60%]">{videoPath}</span>
              </div>
              <video
                controls
                autoPlay
                loop
                className="w-full rounded-lg bg-black max-h-64"
                src={`/files/video?path=${encodeURIComponent(videoPath)}`}
              >
                您的浏览器不支持 HTML5 视频
              </video>
            </div>
          )}

          {/* Re-run button (shown after done) */}
          {isDone && (
            <div className="flex justify-end">
              <button
                onClick={handleRerun}
                className="px-5 py-2 border-2 border-indigo-500 text-indigo-600 hover:bg-indigo-50 rounded-xl text-sm font-medium transition"
              >
                ↺ 重新运行
              </button>
            </div>
          )}

        </main>
      </div>

      {/* File Browser Modal */}
      <FileBrowser
        open={browserOpen}
        onClose={() => setBrowserOpen(false)}
        onSelect={handleBrowserSelect}
      />
    </div>
  )
}
