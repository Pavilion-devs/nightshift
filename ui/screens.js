// ---- screens: the full page (sidebar + incidents board), the compact Desktop pane, modals, the status chip ----

const KIND_ICON = { rollback_restart: 'undo', oom_restart: 'undo', stop: 'stop', notify: 'bell' }

function Section({ label, children, open, onToggle }) {
  return h('div', { style: { display: 'flex', flexDirection: 'column', gap: 4 } },
    h('button', { type: 'button', onClick: onToggle, style: { display: 'flex', alignItems: 'center', border: 0,
      background: 'none', cursor: 'pointer', padding: '14px 12px 8px', color: '#6f6f76', fontSize: 12,
      letterSpacing: '0.07em', textTransform: 'uppercase' } },
    label, h('span', { style: { marginLeft: 'auto', transform: open ? 'none' : 'rotate(-90deg)' } },
      h(Icon, { name: 'chevD', size: 15, color: '#6f6f76' }))),
    open ? children : null)
}

function NavItem({ icon, label, on, badge, onClick, trail }) {
  return h('button', { type: 'button', className: `ns-nav${on ? ' on' : ''}`, onClick },
    typeof icon === 'string' ? h(Icon, { name: icon, size: 19, color: on ? '#f0e8ff' : '#bdbdbd' }) : icon,
    h('span', { style: { flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' } }, label),
    badge ? h('span', { style: { minWidth: 22, height: 22, padding: '0 6px', borderRadius: 7, background: C.badge,
      color: '#efe6ff', fontSize: 12.5, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', ...NUM } },
    badge) : null,
    trail || null)
}

function RunGlyph({ run }) {
  const color = run.status === 'pending' ? C.lilac : run.health === 'critical' || run.status === 'failed' ? C.red
    : LIVE.includes(run.status) ? C.purpleHi : run.status === 'finished' ? '#3fbf8f' : '#6b6b6b'
  return h('span', { style: { width: 19, height: 19, flex: 'none', borderRadius: 5, background: color,
    display: 'inline-flex', alignItems: 'center', justifyContent: 'center' } },
  h(Icon, { name: LIVE.includes(run.status) ? 'pulse' : run.status === 'pending' ? 'clock' : 'check', size: 13,
    color: '#151515', sw: 2.4 }))
}

function Sidebar({ st, view, setView, runId, setRunId, openModal }) {
  const [open, setOpen] = useState({ monitor: true, runs: true })
  const runs = st.runs || []
  const live = runs.filter(r => LIVE.includes(r.status)).length
  const toggle = k => setOpen({ ...open, [k]: !open[k] })
  return h('aside', { className: 'ns-side ns-scroll', style: { width: 252, flex: 'none', background: C.side,
    border: `1px solid ${C.line}`, borderRadius: 20, padding: 12, display: 'flex', flexDirection: 'column' } },
  h('div', { style: { display: 'flex', alignItems: 'center', gap: 12, padding: '10px 12px', borderRadius: 13,
    background: '#242424', border: '1px solid #303030', boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.04)' } },
  h(Logo, { size: 32 }),
  h('div', { style: { display: 'flex', flexDirection: 'column', minWidth: 0, flex: 1 } },
    h('span', { style: { fontSize: 14.5, fontWeight: 500 } }, 'Nightshift'),
    h('span', { style: { fontSize: 12.5, color: '#9a9a9a' } }, 'Training runs')),
  h(Icon, { name: 'updown', size: 17, color: '#9a9a9a' })),
  h(Section, { label: 'Monitor', open: open.monitor, onToggle: () => toggle('monitor') },
    h(NavItem, { icon: 'home', label: 'Overview', on: view === 'overview', onClick: () => setView('overview') }),
    h(NavItem, { icon: 'alert', label: 'Incidents', on: view === 'incidents', badge: st.unacknowledged_incidents || null,
      onClick: () => setView('incidents') }),
    h(NavItem, { icon: 'book', label: 'Run log', onClick: () => runId && openModal({ type: 'log', runId }) }),
    h(NavItem, { icon: 'report', label: 'Morning report', onClick: () => openModal({ type: 'report' }) })),
  h(Section, { label: 'Runs', open: open.runs, onToggle: () => toggle('runs') },
    runs.length ? runs.map(r => h(NavItem, { key: r.id, icon: h(RunGlyph, { run: r }), label: r.name,
      on: false, onClick: () => { setRunId(r.id); setView('incidents') },
      trail: h('span', { style: { color: r.id === runId ? C.purpleHi : '#5f5f66', fontSize: 15 } },
        r.id === runId ? '★' : '☆') }))
      : h('div', { style: { padding: '4px 12px', color: C.muted, fontSize: 13 } }, 'No runs yet')),
  h('div', { style: { flex: 1 } }),
  h('div', { style: { borderTop: `1px solid ${C.line}`, margin: '10px 4px 0', padding: '12px 8px 4px',
    display: 'flex', alignItems: 'center', gap: 10, fontSize: 13.5, color: '#cfcfcf' } },
  h('span', { style: { width: 9, height: 9, borderRadius: 5, background: live ? C.green : '#5f5f66',
    animation: live ? 'ns-pulse 1.6s infinite' : 'none' } }),
  h('span', null, live ? `Watching ${live} run${live > 1 ? 's' : ''}` : 'Silent while healthy')))
}

function Header({ title, st, openModal }) {
  const unacked = st.unacknowledged_incidents || 0
  return h('div', { style: { display: 'flex', alignItems: 'center', gap: 12, paddingBottom: 16,
    borderBottom: `1px solid ${C.line}` } },
  h(Icon, { name: 'home', size: 19, color: '#c8c8c8' }),
  h(Icon, { name: 'chevR', size: 14, color: '#777' }),
  h('span', { style: { fontSize: 15 } }, title),
  h('span', { style: { flex: 1 } }),
  h('button', { type: 'button', className: 'ns-ai', onClick: () => openModal({ type: 'report' }) },
    h(Icon, { name: 'spark', size: 16, color: '#fff' }), 'Night report'),
  h('button', { type: 'button', className: 'ns-icon-btn', title: unacked ? `Mark ${unacked} seen` : 'All caught up',
    onClick: () => unacked && act('/ack') },
  h(Icon, { name: 'bell', size: 18 }),
  unacked ? h('span', { style: { position: 'absolute', top: 7, right: 8, width: 8, height: 8, borderRadius: 4,
    background: '#ff3b3b', boxShadow: '0 0 0 2px #212121' } }) : null))
}

function Banner({ st, openModal, compact }) {
  const b = st.banner || {}
  const pending = (st.runs || []).find(r => r.status === 'pending')
  if (pending) {
    return h('div', { style: { display: 'flex', alignItems: 'center', gap: 14, padding: compact ? '12px 14px' : '16px 18px',
      borderRadius: 14, background: '#241f2d', border: '1px solid #3a2f4d' } },
    h(Icon, { name: 'clock', size: 20, color: C.purpleHi }),
    h('div', { style: { flex: 1, minWidth: 0 } },
      h('div', { style: { fontSize: compact ? 14 : 15.5 } }, `${pending.name} is waiting for your approval`),
      h('div', { style: { fontSize: 13.5, color: C.sub, marginTop: 3 } }, 'Nothing runs until you approve the exact command')),
    h('button', { type: 'button', className: 'ns-btn', onClick: () => openModal({ type: 'approve', runId: pending.id }) },
      'Review'))
  }
  if (!b.incidents || !st.unacknowledged_incidents) return null
  const sub = b.attention ? `${b.attention} still need${b.attention > 1 ? '' : 's'} your attention`
    : b.recovered ? `${b.recovered === b.incidents ? 'All' : b.recovered} recovered automatically. No action needed`
      : 'Review what happened'
  return h('div', { style: { display: 'flex', alignItems: 'center', gap: 14, padding: compact ? '12px 14px' : '16px 18px',
    borderRadius: 14, background: C.bannerBg, border: `1px solid ${C.bannerLine}` } },
  h(Icon, { name: 'alert', size: 21, color: C.amber }),
  h('div', { style: { flex: 1, minWidth: 0 } },
    h('div', { style: { fontSize: compact ? 14 : 15.5 } },
      `Nightshift caught ${b.incidents} incident${b.incidents > 1 ? 's' : ''} in the last ${b.window_h || 24} hours`),
    h('div', { style: { fontSize: 13.5, color: C.sub, marginTop: 3 } }, sub)),
  compact ? null : h('button', { type: 'button', className: 'ns-btn', style: { background: '#2f2a22', borderColor: '#463a28' },
    onClick: () => openModal({ type: 'log', runId: (st.table[0] || {}).run_id }) }, 'View all incidents'),
  h('button', { type: 'button', title: 'Mark seen', onClick: () => act('/ack'), style: { border: 0, background: 'none',
    cursor: 'pointer', color: '#9a9a9a', padding: 4 } }, h(Icon, { name: 'x', size: 18 })))
}

function IncidentRow({ inc, onOpen, compact }) {
  const crit = inc.severity === 'critical'
  const span = inc.onset_step !== null && inc.onset_step !== undefined && inc.step !== null && inc.onset_step !== inc.step
    ? `steps ${inc.onset_step}–${inc.step}` : inc.step !== null && inc.step !== undefined ? `step ${inc.step}` : ''
  const msg = inc.message.charAt(0).toUpperCase() + inc.message.slice(1)
  return h('div', { style: { display: 'flex', alignItems: 'center', gap: 14, padding: compact ? '10px 12px' : '14px 16px',
    borderRadius: 14, border: `1px solid ${C.line}`, background: C.inner } },
  h(IconBox, { name: crit ? 'octagon' : 'alert', color: crit ? C.red : C.yellow, size: compact ? 32 : 38 }),
  h('div', { style: { flex: 1, minWidth: 0 } },
    h('div', { style: { fontSize: compact ? 14 : 15, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' } }, msg),
    h('div', { style: { fontSize: 13.5, color: C.muted, marginTop: 3, ...NUM } },
      [`${fmtDay(inc.at)}, ${fmtClock(inc.at)}`, span, inc.state === 'recovered' ? 'recovered' : ''].filter(Boolean).join(' · '))),
  h('button', { type: 'button', className: 'ns-link', onClick: onOpen },
    crit ? 'Investigate' : 'Review', h(Icon, { name: 'chevR', size: 15 })))
}

function ChartCard({ run, openModal, compact }) {
  const [range, setRange] = useState('all')
  const seg = (k, label) => h('button', { type: 'button', className: `ns-seg${range === k ? ' on' : ''}`,
    onClick: () => setRange(k) }, label)
  const stat = (k, v) => h('span', { style: { fontSize: 13, color: C.muted } }, `${k} `,
    h('span', { style: { color: C.text, ...NUM } }, v))
  return h(Card, { pad: compact ? 14 : 20, style: { display: 'flex', flexDirection: 'column', gap: compact ? 12 : 16 } },
    h('div', { style: { display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' } },
      h('div', { style: { display: 'flex', flexDirection: 'column', gap: 4, minWidth: 0 } },
        h('span', { style: { fontSize: compact ? 15 : 16.5, fontWeight: 600 } }, `${run.name}`),
        h('div', { style: { display: 'flex', gap: 14, flexWrap: 'wrap' } },
          stat('step', run.step === null || run.step === undefined ? '—' : run.step),
          stat('loss', fmtNum(run.train_loss && run.train_loss.value)),
          run.val_loss ? stat('val', fmtNum(run.val_loss.value)) : null,
          stat('fixes', `${run.interventions} of ${(run.playbook || {}).max_interventions || 3}`))),
      h('span', { style: { flex: 1 } }),
      h(Pill, { kind: run.status === 'pending' ? 'pending' : LIVE.includes(run.status) ? (run.health === 'critical' ? 'critical' : 'running')
        : run.status === 'finished' ? 'success' : 'failed', label: run.status === 'finished' ? 'Finished' : undefined }),
      compact ? null : h('div', { style: { display: 'inline-flex', gap: 2, padding: 3, borderRadius: 9, background: C.inner,
        border: `1px solid ${C.line}` } }, seg('all', 'All'), seg('recent', 'Last 1k'),
      (run.markers || []).length ? seg('incident', 'Incident') : null)),
    h(LossChart, { run, range, height: compact ? 150 : 250 }),
    h(ChartLegend, { run }),
    ...(run.incidents || []).slice(0, compact ? 2 : 3).map(inc => h(IncidentRow, { key: inc.id, inc, compact,
      onOpen: () => openModal({ type: 'log', runId: run.id }) })))
}

function Diagnostics({ run, openModal, compact }) {
  const sig = run.signals || []
  const did = (run.did || []).filter(a => a.kind !== 'notify' || a.message.indexOf('notify only') !== 0)
  const live = LIVE.includes(run.status) && run.mode === 'launch'
  const todo = run.status === 'pending'
    ? [h('button', { key: 'a', type: 'button', className: 'ns-todo', onClick: () => openModal({ type: 'approve', runId: run.id }) },
      h(Icon, { name: 'play', size: 16, color: C.purpleHi }), h('span', { style: { flex: 1 } }, 'Review & approve'),
      h(Icon, { name: 'chevR', size: 16 }))]
    : did.slice(-3).reverse().map((a, i) => h('button', { key: i, type: 'button', className: 'ns-todo',
      onClick: () => openModal({ type: 'log', runId: run.id }) },
    h(Icon, { name: KIND_ICON[a.kind] || 'check', size: 16, color: a.ok ? '#bdbdbd' : C.red }),
    h('span', { style: { flex: 1, minWidth: 0 } }, a.message.charAt(0).toUpperCase() + a.message.slice(1)),
    h(Icon, { name: 'chevR', size: 16 })))
  if (live) todo.push(h(StopButton, { key: 'stop', run }))
  return h(Card, { pad: compact ? 14 : 20, style: { display: 'flex', flexDirection: 'column', gap: 16 } },
    h('span', { style: { fontSize: compact ? 15 : 16.5, fontWeight: 600 } }, 'Diagnostics'),
    sig.length ? h('div', { style: { border: `1px solid ${C.line}`, borderRadius: 14, padding: 16, display: 'flex',
      flexDirection: 'column', gap: 16 } },
    ...sig.map((s, i) => h('div', { key: i, style: { display: 'flex', flexDirection: 'column', gap: 9 } },
      h('div', { style: { display: 'flex', fontSize: 14, color: '#d0d0d0' } },
        h('span', { style: { flex: 1 } }, `${s.metric} vs its recent median`),
        h('span', { style: { color: C.sub, ...NUM } }, s.text)),
      h('div', { style: { height: 7, borderRadius: 4, background: C.track, overflow: 'hidden' } },
        h('div', { style: { width: `${Math.round((s.frac || 0) * 100)}%`, height: '100%', borderRadius: 4,
          background: `linear-gradient(90deg, ${C.purple} 0%, ${C.purpleHi} 100%)` } }))))) : null,
    h('div', { style: { display: 'flex', gap: 12, padding: 16, borderRadius: 14, background: C.ai,
      border: `1px solid ${C.aiLine}` } },
    h(Icon, { name: 'spark', size: 18, color: C.purpleHi }),
    h('div', { style: { display: 'flex', flexDirection: 'column', gap: 6 } },
      h('span', { style: { fontSize: 14, lineHeight: 1.5, color: '#e8e4ee' } }, run.diagnosis || run.summary || ''),
      h('span', { style: { fontSize: 12, color: C.muted } },
        run.diagnosis ? 'Root cause from your model, after recovery' : 'From Nightshift’s own records'))),
    todo.length ? h('div', { style: { border: `1px solid ${C.line}`, borderRadius: 14, padding: 16, display: 'flex',
      flexDirection: 'column', gap: 10 } },
    h('span', { style: { fontSize: 15, fontWeight: 600, marginBottom: 4 } },
      run.status === 'pending' ? 'What to do' : 'What Nightshift did'), ...todo) : null)
}

function StopButton({ run }) {
  const [armed, setArmed] = useState(false)
  useEffect(() => {
    if (!armed) return undefined
    const id = setTimeout(() => setArmed(false), 4000)
    return () => clearTimeout(id)
  }, [armed])
  return h('button', { type: 'button', className: 'ns-todo', style: armed ? { color: C.red } : null,
    onClick: () => { if (armed) { setArmed(false); act(`/stop/${run.id}`) } else setArmed(true) } },
  h(Icon, { name: 'stop', size: 16, color: C.red }),
  h('span', { style: { flex: 1 } }, armed ? 'Click again to stop the run' : 'Stop this run'),
  h(Icon, { name: 'chevR', size: 16 }))
}

function IncidentTable({ st, onView }) {
  const rows = st.table || []
  const cols = 'minmax(120px,1.1fr) minmax(200px,2.4fr) 104px 104px 124px 70px 76px'
  const head = ['Run', 'Description', 'Severity', 'Time', 'Status', 'Step', 'Action']
  return h(Card, { pad: 0, style: { overflow: 'hidden' } },
    h('div', { style: { display: 'grid', gridTemplateColumns: cols, gap: 12, padding: '22px 20px', color: C.muted,
      fontSize: 14.5, borderBottom: `1px solid ${C.line}` } },
    ...head.map((t, i) => h('span', { key: i, style: { textAlign: i === 6 ? 'right' : 'left' } }, t))),
    rows.length ? rows.map(r => h('div', { key: r.id, className: 'ns-row', style: { display: 'grid',
      gridTemplateColumns: cols, gap: 12, alignItems: 'center', padding: '16px 20px', borderBottom: `1px solid ${C.line}` } },
    h('span', { style: { fontSize: 15, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' } }, r.run),
    h('span', { style: { fontSize: 14, color: C.sub, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' } },
      r.message),
    h('span', null, h(Pill, { kind: r.severity })),
    h('span', { style: { fontSize: 14, color: C.sub, ...NUM } }, fmtClock(r.at)),
    h('span', null, h(Pill, { kind: r.state })),
    h('span', { style: { fontSize: 14, color: C.sub, ...NUM } }, r.step === null || r.step === undefined ? '—' : r.step),
    h('span', { style: { textAlign: 'right' } },
      h('button', { type: 'button', className: 'ns-btn', onClick: () => onView(r) }, 'View'))))
      : h('div', { style: { padding: '28px 20px', color: C.muted, fontSize: 14 } },
        'No incidents in the last 24 hours. Nightshift stays silent while runs are healthy.'))
}

function Modal({ title, onClose, children, width }) {
  return h('div', { onClick: onClose, style: { position: 'fixed', inset: 0, zIndex: 50, background: 'rgba(8,8,8,0.62)',
    backdropFilter: 'blur(6px)', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 16 } },
  h('div', { onClick: e => e.stopPropagation(), className: 'ns-root', style: { width: '100%', maxWidth: width || 760,
    maxHeight: '86%', display: 'flex', flexDirection: 'column', background: C.card, border: `1px solid ${C.line}`,
    borderRadius: 18, overflow: 'hidden', boxShadow: '0 30px 80px rgba(0,0,0,0.55)' } },
  h('div', { style: { display: 'flex', alignItems: 'center', padding: '18px 22px', background: '#232323',
    borderBottom: `1px solid ${C.line}` } },
  h('span', { style: { fontSize: 15.5, fontWeight: 600, flex: 1 } }, title),
  h('button', { type: 'button', onClick: onClose, style: { border: 0, background: 'none', color: '#bbb', cursor: 'pointer' } },
    h(Icon, { name: 'x', size: 18 }))),
  h('div', { className: 'ns-scroll', style: { overflowY: 'auto' } }, children)))
}

function LogModal({ run, onClose }) {
  const cols = '130px minmax(150px,1.2fr) 110px minmax(160px,1.6fr)'
  return h(Modal, { title: `Run log · ${run.name}`, onClose },
    h('div', { style: { display: 'grid', gridTemplateColumns: cols, gap: 12, padding: '20px 22px', color: C.muted,
      fontSize: 14.5, borderBottom: `1px solid ${C.line}` } },
    ...['Timestamp', 'Event', 'Status', 'Details'].map(t => h('span', { key: t }, t))),
    ...(run.log || []).map((e, i) => h('div', { key: i, style: { display: 'grid', gridTemplateColumns: cols, gap: 12,
      alignItems: 'center', padding: '20px 22px', borderBottom: `1px solid ${C.line}` } },
    h('span', { style: { fontSize: 13.5, color: C.sub, ...NUM } }, fmtClock(e.at, true)),
    h('span', { style: { fontSize: 15 } }, e.event),
    h('span', null, h(Pill, { kind: e.status })),
    h('span', { style: { fontSize: 13.5, color: C.sub, lineHeight: 1.45 } }, e.detail))))
}

function ApproveModal({ run, onClose }) {
  const [ok, setOk] = useState(false)
  const card = run.card || {}
  const row = (k, v) => h('div', { style: { display: 'grid', gridTemplateColumns: '120px 1fr', gap: 14,
    padding: '12px 0', borderBottom: `1px solid ${C.line}` } },
  h('span', { style: { fontSize: 13.5, color: C.muted } }, k), v)
  return h(Modal, { title: `Approve run · ${run.name}`, onClose, width: 640 },
    h('div', { style: { padding: '20px 22px', display: 'flex', flexDirection: 'column', gap: 4 } },
      h('div', { style: { padding: 16, borderRadius: 14, background: C.ai, border: `1px solid ${C.aiLine}`, marginBottom: 8 } },
        h('div', { style: { fontSize: 12.5, color: C.muted, marginBottom: 6 } }, 'Command'),
        h('div', { style: { fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace', fontSize: 13.5,
          lineHeight: 1.5, wordBreak: 'break-all', color: '#efe9f7' } }, card.command || '')),
      row('Directory', h('span', { style: { fontSize: 14, wordBreak: 'break-all' } }, card.cwd || '')),
      row('Playbook', h('div', { style: { display: 'flex', flexDirection: 'column', gap: 6, fontSize: 14 } },
        ...(card.playbook || []).map((ln, i) => h('span', { key: i }, ln)))),
      h('label', { style: { display: 'flex', alignItems: 'center', gap: 10, marginTop: 14, padding: '12px 14px',
        borderRadius: 12, background: C.inner, fontSize: 14, cursor: 'pointer' } },
      h('input', { type: 'checkbox', checked: ok, onChange: e => setOk(e.target.checked),
        style: { width: 17, height: 17, accentColor: C.purple } }),
      'I reviewed the command, the directory and the playbook'),
      h('div', { style: { display: 'flex', alignItems: 'center', gap: 10, marginTop: 16 } },
        h('button', { type: 'button', className: 'ns-link', style: { color: C.red },
          onClick: () => { act(`/deny/${run.id}`); onClose() } }, 'Decline'),
        h('span', { style: { flex: 1 } }),
        h('button', { type: 'button', className: 'ns-btn', onClick: onClose }, 'Not now'),
        h('button', { type: 'button', className: 'ns-ai', disabled: !ok, style: ok ? null : { opacity: 0.45, cursor: 'not-allowed' },
          onClick: () => { if (ok) { act(`/approve/${run.id}`); onClose() } } }, 'Approve & launch'))))
}

function ReportModal({ st, onClose }) {
  const r = st.report || {}
  const big = (n, label) => h('div', { style: { flex: 1, minWidth: 120, padding: '18px 18px', borderRadius: 14,
    border: `1px solid ${C.line}`, background: C.inner } },
  h('div', { style: { fontSize: 12, letterSpacing: '0.07em', textTransform: 'uppercase', color: C.muted } }, label),
  h('div', { style: { fontSize: 30, fontWeight: 500, marginTop: 8, ...NUM } }, n))
  return h(Modal, { title: 'While you slept · last 24 hours', onClose, width: 680 },
    h('div', { style: { padding: 22, display: 'flex', flexDirection: 'column', gap: 16 } },
      h('div', { style: { display: 'flex', gap: 12, flexWrap: 'wrap' } },
        big(r.runs || 0, 'Runs'), big(r.incidents || 0, 'Incidents'), big(r.recoveries || 0, 'Recovered'),
        big(`${(r.gpu_hours_rescued_estimate || 0).toFixed(1)} h`, 'GPU time rescued')),
      h('div', { style: { fontSize: 13, color: C.muted } },
        'GPU time rescued is an estimate: how long each recovered attempt went on to train.'),
      ...(st.runs || []).filter(x => (x.did || []).length).map(x => h('div', { key: x.id, style: { display: 'flex',
        flexDirection: 'column', gap: 6 } },
      h('span', { style: { fontSize: 15, fontWeight: 600 } }, x.name),
      ...x.did.filter(a => a.kind !== 'notify').map((a, i) => h('span', { key: i, style: { fontSize: 14, color: C.sub } },
        `${fmtClock(a.at)} · ${a.message}`))))))
}

function Modals({ st, modal, close }) {
  if (!modal) return null
  const run = (st.runs || []).find(r => r.id === modal.runId)
  if (modal.type === 'report') return h(ReportModal, { st, onClose: close })
  if (!run) return null
  if (modal.type === 'approve' && run.status === 'pending') return h(ApproveModal, { run, onClose: close })
  if (modal.type === 'log') return h(LogModal, { run, onClose: close })
  return null
}

function RunTile({ run, onOpen }) {
  return h('button', { type: 'button', onClick: onOpen, className: 'ns-row', style: { textAlign: 'left', cursor: 'pointer',
    border: `1px solid ${C.line}`, background: C.card, borderRadius: 18, padding: 18, color: C.text,
    display: 'flex', flexDirection: 'column', gap: 12 } },
  h('div', { style: { display: 'flex', alignItems: 'center', gap: 10 } }, h(RunGlyph, { run }),
    h('span', { style: { fontSize: 15.5, fontWeight: 600, flex: 1 } }, run.name),
    h(Pill, { kind: run.status === 'pending' ? 'pending' : LIVE.includes(run.status) ? 'running'
      : run.status === 'finished' ? 'success' : 'failed', label: run.status === 'finished' ? 'Finished' : undefined })),
  h(LossChart, { run, range: 'all', height: 90 }),
  h('span', { style: { fontSize: 13, color: C.muted, ...NUM } },
    `step ${run.step === null || run.step === undefined ? '—' : run.step} · loss ${fmtNum(run.train_loss && run.train_loss.value)} · ${run.interventions} fix${run.interventions === 1 ? '' : 'es'}`))
}

function Page() {
  const { status, error } = useStatus()
  const st = status || { runs: [], table: [] }
  const [view, setView] = useState('incidents')
  const [runId, setRunId] = useState(null)
  const [modal, openModal] = useState(null)
  const runs = st.runs || []
  const run = runs.find(r => r.id === runId) || pickRun(runs)
  const body = !status && !error ? h('div', { style: { color: C.muted, padding: 20 } }, 'Loading…')
    : error ? h(Card, null, h('span', { style: { color: C.sub } }, `Can't reach Nightshift: ${error}`))
      : !runs.length ? h(Card, null, h('div', { style: { fontSize: 15, marginBottom: 8 } }, 'No training runs yet'),
        h('div', { style: { color: C.sub, fontSize: 14 } },
          'Ask Hermes: "train model.py with lr 3e-4 and watch it overnight". You approve the exact command before anything starts.'))
        : view === 'overview'
          ? h('div', { style: { display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 16 } },
            ...runs.map(r => h(RunTile, { key: r.id, run: r, onOpen: () => { setRunId(r.id); setView('incidents') } })))
          : h('div', { style: { display: 'flex', flexDirection: 'column', gap: 18 } },
            h('div', { className: 'ns-grid' },
              run ? h(ChartCard, { key: run.id, run, openModal }) : null,
              run ? h(Diagnostics, { run, openModal }) : null),
            h('div', { className: 'ns-scroll', style: { overflowX: 'auto' } },
              h('div', { style: { minWidth: 760 } },
                h(IncidentTable, { st, onView: r => { setRunId(r.run_id); openModal({ type: 'log', runId: r.run_id }) } }))))
  return h('div', { className: 'ns-root', style: { display: 'flex', gap: 14, padding: 14, height: '100%', minHeight: 560,
    background: C.page, borderRadius: 0 } },
  h('style', null, CSS),
  h(Sidebar, { st, view, setView, runId: run && run.id, setRunId, openModal }),
  h('main', { className: 'ns-scroll', style: { flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 18,
    padding: '10px 6px 24px 8px' } },
  h(Header, { title: view === 'overview' ? 'Overview' : 'Incidents', st, openModal }),
  h(Banner, { st, openModal }),
  body),
  h(Modals, { st, modal, close: () => openModal(null) }))
}

function Pane() {
  const { status, error } = useStatus()
  const st = status || { runs: [] }
  const [runId, setRunId] = useState(null)
  const [modal, openModal] = useState(null)
  const runs = st.runs || []
  const run = runs.find(r => r.id === runId) || pickRun(runs)
  return h('div', { className: 'ns-root ns-scroll', style: { display: 'flex', flexDirection: 'column', gap: 12,
    padding: 12, height: '100%', background: C.page } },
  h('style', null, CSS),
  h('div', { style: { display: 'flex', alignItems: 'center', gap: 10 } },
    h(Logo, { size: 26 }), h('span', { style: { fontSize: 15, fontWeight: 600 } }, 'Nightshift'),
    h('span', { style: { flex: 1 } }),
    (st.unacknowledged_incidents || 0) ? h('button', { type: 'button', className: 'ns-icon-btn', title: 'Mark seen',
      onClick: () => act('/ack') }, h(Icon, { name: 'bell', size: 17 }),
    h('span', { style: { position: 'absolute', top: 7, right: 8, width: 7, height: 7, borderRadius: 4, background: '#ff3b3b' } }))
      : null),
  runs.length > 1 ? h('div', { style: { display: 'flex', gap: 6, flexWrap: 'wrap' } },
    ...runs.slice(0, 6).map(r => h('button', { key: r.id, type: 'button', className: `ns-seg${run && r.id === run.id ? ' on' : ''}`,
      style: { display: 'inline-flex', alignItems: 'center', gap: 6, border: `1px solid ${C.line}` }, onClick: () => setRunId(r.id) },
    h(RunGlyph, { run: r }), r.name))) : null,
  error ? h(Card, { pad: 14 }, h('span', { style: { color: C.sub } }, `Can't reach Nightshift: ${error}`)) : null,
  h(Banner, { st, openModal, compact: true }),
  run ? h(ChartCard, { key: run.id, run, openModal, compact: true }) : null,
  run ? h(Diagnostics, { run, openModal, compact: true }) : null,
  !error && status && !runs.length ? h(Card, { pad: 14 }, h('span', { style: { color: C.sub, fontSize: 14 } },
    'No runs yet. Ask Hermes: "train model.py with lr 3e-4 and watch it overnight".')) : null,
  h(Modals, { st, modal, close: () => openModal(null) }))
}

function Chip() {
  const { status, error } = useStatus()
  const runs = (status && status.runs) || []
  const running = runs.filter(r => ['running', 'recovering'].includes(r.status))
  const critical = runs.some(r => r.health === 'critical')
  const pending = runs.some(r => r.status === 'pending')
  const color = error ? C.muted : critical ? C.red : pending ? C.lilac : C.purpleHi
  const label = error ? 'Nightshift offline'
    : pending ? 'A run is waiting for your approval'
      : running.length ? running.map(r => `${r.name}: ${r.health}`).join(' · ') : 'Nightshift: no runs'
  return { color, text: `☾ ${running.length}${pending ? ' ·!' : ''}`, label }
}
