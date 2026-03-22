const BASE = ''   // proxied via Vite → http://localhost:8000

function token() {
  return localStorage.getItem('token') || ''
}

function authHeader() {
  return { Authorization: `Bearer ${token()}`, 'Content-Type': 'application/json' }
}

async function request(method, path, body) {
  const res = await fetch(BASE + path, {
    method,
    headers: authHeader(),
    body: body ? JSON.stringify(body) : undefined,
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`)
  return data
}

export const api = {
  register: (username, password) =>
    request('POST', '/auth/register', { username, password }),

  login: (username, password) =>
    request('POST', '/auth/login', { username, password }),

  browse: (path = '') =>
    request('GET', `/files/browse?path=${encodeURIComponent(path)}`),

  preview: (path) =>
    request('GET', `/files/preview?path=${encodeURIComponent(path)}`),

  findVideo: (path) =>
    request('GET', `/files/find_video?path=${encodeURIComponent(path)}`),

  runProcess: (payload) =>
    request('POST', '/process/run', payload),

  jobStatus: (jobId) =>
    request('GET', `/process/status/${jobId}`),

  /** Returns an EventSource for SSE log streaming */
  streamLogs: (jobId) =>
    new EventSource(`/process/logs/${jobId}?token=${encodeURIComponent(token())}`),
}
