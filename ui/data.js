// ---- data: one shared poll of /status for every view in this host --------------------------------------------
const POLL_MS = 2000
const listeners = new Set()
let latest = null
let lastError = null
let timer = null
const seenAlerts = new Set()
let primed = false

async function refresh() {
  try {
    latest = await api.get('/status')
    lastError = null
    const fresh = (latest.alerts || []).filter(a => !seenAlerts.has(a.id))
    fresh.forEach(a => seenAlerts.add(a.id))
    if (primed && fresh.length && env.onNewAlerts) env.onNewAlerts(fresh, latest)
    primed = true
  } catch (err) {
    lastError = String((err && err.message) || err)
  }
  listeners.forEach(fn => fn())
}

function subscribe(fn) {
  listeners.add(fn)
  if (!timer) {
    refresh()
    timer = setInterval(refresh, POLL_MS)
  }
  return () => {
    listeners.delete(fn)
    if (!listeners.size && timer) {
      clearInterval(timer)
      timer = null
    }
  }
}

function useStatus() {
  const [, bump] = useState(0)
  useEffect(() => subscribe(() => bump(n => n + 1)), [])
  return { status: latest, error: lastError }
}

async function act(path) {
  try {
    await api.post(path)
  } finally {
    refresh()
  }
}

const LIVE = ['approved', 'starting', 'running', 'recovering']

// The run a view should open on: one waiting for approval, then one in trouble, then a live one, then the
// most recent one that had an incident.
function pickRun(runs) {
  return runs.find(r => r.status === 'pending') ||
    runs.find(r => r.health === 'critical') ||
    runs.find(r => LIVE.includes(r.status)) ||
    runs.find(r => (r.markers || []).length && (r.recoveries || []).length) ||
    runs.find(r => (r.incidents || []).length) ||
    runs[0] || null
}
