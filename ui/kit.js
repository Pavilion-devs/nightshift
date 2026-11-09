// ---- kit: palette, scoped CSS, icons, formatting, small primitives -----------------------------------------
// Dark, purple-accented look: near-black surfaces, a lilac chart line, an amber attention banner,
// outlined status pills. Fixed colours on purpose: the screen looks the same in every host theme.

const C = {
  page: '#1c1c1c',
  side: '#212121',
  line: '#2a2a2a',
  card: '#1e1e1e',
  inner: '#212121',
  raised: '#262626',
  raisedHi: '#2c2c2c',
  divider: '#2b2b2b',
  text: '#f2f2f2',
  sub: '#a6a6a6',
  muted: '#7e7c87',
  faint: '#55555c',
  purple: '#914cfd',
  purpleHi: '#ae84ff',
  lilac: '#cfa8ff',
  badge: '#2e2638',
  ai: '#26232c',
  aiLine: '#353040',
  bannerBg: '#2b2219',
  bannerLine: '#3d301c',
  amber: '#ff9100',
  amberLine: '#f5a524',
  red: '#ff577f',
  redLine: '#e5484d',
  yellow: '#fdee76',
  green: '#91fdd9',
  track: '#3d3d3d'
}

const FONT = '"Inter", "SF Pro Text", -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif'
const NUM = { fontVariantNumeric: 'tabular-nums' }

// Hover states, scrollbars and the responsive grid need real CSS; every class is prefixed `ns-` and lives
// inside our own root, so nothing leaks into the host app.
const CSS = `
.ns-root { font-family: ${FONT}; color: ${C.text}; -webkit-font-smoothing: antialiased; letter-spacing: -0.005em; }
.ns-root *, .ns-root *::before, .ns-root *::after { box-sizing: border-box; }
.ns-root button { font-family: inherit; }
.ns-scroll { overflow-y: auto; scrollbar-width: thin; scrollbar-color: #333 transparent; }
.ns-nav { display: flex; align-items: center; gap: 12px; height: 40px; padding: 0 12px; border-radius: 10px;
  color: #e6e6e6; font-size: 14.5px; cursor: pointer; position: relative; border: 0; background: transparent;
  width: 100%; text-align: left; }
.ns-nav:hover { background: #262626; }
.ns-nav.on { background: linear-gradient(90deg, #262429 0%, #2f2a38 55%, #3a2b52 100%); color: #fff; }
.ns-nav.on::after { content: ''; position: absolute; right: -1px; top: 8px; bottom: 8px; width: 3px;
  border-radius: 3px; background: ${C.purple}; box-shadow: 0 0 12px ${C.purple}; }
.ns-btn { display: inline-flex; align-items: center; gap: 6px; border-radius: 10px; border: 1px solid #333;
  background: ${C.raised}; color: ${C.text}; font-size: 14px; padding: 8px 14px; cursor: pointer; }
.ns-btn:hover { background: ${C.raisedHi}; }
.ns-link { display: inline-flex; align-items: center; gap: 4px; background: none; border: 0; color: ${C.text};
  font-size: 14px; cursor: pointer; padding: 4px 0; }
.ns-link:hover { color: #fff; text-decoration: underline; text-underline-offset: 3px; }
.ns-ai { display: inline-flex; align-items: center; gap: 6px; border: 0; border-radius: 9px; color: #fff;
  font-size: 14px; font-weight: 500; padding: 8px 14px; cursor: pointer;
  background: linear-gradient(180deg, #b683ff 0%, #9c52ff 100%);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.25), 0 4px 16px rgba(145,76,253,0.25); }
.ns-ai:hover { filter: brightness(1.08); }
.ns-icon-btn { width: 38px; height: 38px; display: inline-flex; align-items: center; justify-content: center;
  border-radius: 10px; border: 1px solid #2e2e2e; background: ${C.side}; color: #d0d0d0; cursor: pointer;
  position: relative; }
.ns-icon-btn:hover { background: ${C.raised}; }
.ns-todo { display: flex; align-items: center; gap: 10px; width: 100%; border: 0; border-radius: 10px;
  background: ${C.raised}; color: #ececec; font-size: 14.5px; padding: 10px 14px; cursor: pointer; text-align: left; }
.ns-todo:hover { background: ${C.raisedHi}; }
.ns-seg { border: 0; background: transparent; color: ${C.muted}; font-size: 12.5px; padding: 4px 8px;
  border-radius: 7px; cursor: pointer; }
.ns-seg.on { background: #2a2a2a; color: ${C.text}; }
.ns-row:hover { background: #222222; }
.ns-grid { display: grid; grid-template-columns: minmax(0, 1fr) 340px; gap: 18px; align-items: start; }
@media (max-width: 1180px) { .ns-grid { grid-template-columns: minmax(0, 1fr); } }
@media (max-width: 900px) { .ns-side { display: none !important; } }
@keyframes ns-pulse { 0% { box-shadow: 0 0 0 0 rgba(145,215,190,0.5); } 100% { box-shadow: 0 0 0 7px rgba(145,215,190,0); } }
`

// 24×24 line icons, drawn for Nightshift. Shapes: ['p', d] path, ['c', cx, cy, r] circle, ['r', x, y, w, h, rx].
const ICONS = {
  home: [['p', 'M4 10.6 12 4l8 6.6V19a1 1 0 0 1-1 1h-4.5v-5.5h-5V20H5a1 1 0 0 1-1-1z']],
  pulse: [['p', 'M3 12h4l2.5-6 5 12 2.5-6h4']],
  runs: [['r', 4, 4, 16, 16, 3], ['p', 'M8 15l3-3 2 2 3-4']],
  spark: [['p', 'M10 3.5l1.6 4.4 4.4 1.6-4.4 1.6L10 15.5l-1.6-4.4L4 9.5l4.4-1.6z'],
    ['p', 'M17.5 13.5l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8z']],
  report: [['p', 'M14 3.5H7a1.5 1.5 0 0 0-1.5 1.5v14A1.5 1.5 0 0 0 7 20.5h10a1.5 1.5 0 0 0 1.5-1.5V8z'],
    ['p', 'M14 3.5V8h4.5M9.5 17v-2.5M12 17v-5M14.5 17v-3.5']],
  alert: [['p', 'M12 4 3 19.5h18z'], ['p', 'M12 10v4.2M12 16.8v.2']],
  octagon: [['p', 'M8.4 3.5h7.2l4.9 4.9v7.2l-4.9 4.9H8.4l-4.9-4.9V8.4z'], ['p', 'M12 8v5M12 15.9v.2']],
  check: [['c', 12, 12, 8.5], ['p', 'M8.3 12.3l2.5 2.5 5-5.2']],
  book: [['p', 'M5 4.5h10.5A2.5 2.5 0 0 1 18 7v12.5H7.5A2.5 2.5 0 0 1 5 17z'], ['p', 'M5 17a2.5 2.5 0 0 1 2.5-2.5H18']],
  clock: [['c', 12, 12, 8.5], ['p', 'M12 7.5V12l3 2']],
  calClock: [['r', 3.5, 5, 13, 12.5, 2], ['p', 'M3.5 9h13M7 3.5V6M13 3.5V6'], ['c', 17, 16.5, 4], ['p', 'M17 14.8v1.9l1.2.8']],
  db: [['p', 'M5 6c0-1.4 3.1-2.5 7-2.5s7 1.1 7 2.5-3.1 2.5-7 2.5S5 7.4 5 6z'],
    ['p', 'M5 6v12c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5V6M5 12c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5']],
  gear: [['c', 12, 12, 3], ['p', 'M12 3.5v2.2M12 18.3v2.2M3.5 12h2.2M18.3 12h2.2M6 6l1.6 1.6M16.4 16.4 18 18M6 18l1.6-1.6M16.4 7.6 18 6']],
  help: [['c', 12, 12, 8.5], ['p', 'M9.6 9.6a2.5 2.5 0 1 1 3.4 2.3c-.6.3-1 .8-1 1.5v.4M12 16.6v.2']],
  chevR: [['p', 'M9.5 6l6 6-6 6']],
  chevD: [['p', 'M6.5 9.5l5.5 5.5 5.5-5.5']],
  updown: [['p', 'M8 9.5l4-4 4 4M8 14.5l4 4 4-4']],
  bell: [['p', 'M6.5 16.5v-5a5.5 5.5 0 0 1 11 0v5l1.5 2h-14z'], ['p', 'M10 20.5a2 2 0 0 0 4 0']],
  x: [['p', 'M6.5 6.5l11 11M17.5 6.5l-11 11']],
  moon: [['p', 'M19.5 14.6A7.8 7.8 0 1 1 9.4 4.5a6.3 6.3 0 0 0 10.1 10.1z']],
  undo: [['p', 'M4 9h10.5a5 5 0 0 1 0 10H8'], ['p', 'M7.5 5.5 4 9l3.5 3.5']],
  stop: [['r', 6, 6, 12, 12, 2.5]],
  play: [['p', 'M8 5.5v13l10-6.5z']]
}

function Icon({ name, size, color, sw, fill }) {
  const shapes = ICONS[name] || []
  return h('svg', { width: size || 18, height: size || 18, viewBox: '0 0 24 24', fill: 'none',
    stroke: color || 'currentColor', strokeWidth: sw || 1.6, strokeLinecap: 'round', strokeLinejoin: 'round',
    style: { flex: 'none', display: 'block' } },
  ...shapes.map((s, i) => s[0] === 'p'
    ? h('path', { key: i, d: s[1], fill: fill && i === 0 ? fill : 'none' })
    : s[0] === 'c'
      ? h('circle', { key: i, cx: s[1], cy: s[2], r: s[3] })
      : h('rect', { key: i, x: s[1], y: s[2], width: s[3], height: s[4], rx: s[5] })))
}

// ---- formatting -------------------------------------------------------------------------------------------
function fmtNum(v) {
  if (v === null || v === undefined) return '—'
  if (typeof v === 'string') return v.toUpperCase() === 'NAN' ? 'NaN' : v
  const a = Math.abs(v)
  if (a === 0) return '0'
  if (a >= 10000 || a < 0.001) return v.toExponential(1).replace('e-', 'e−')
  if (a >= 100) return v.toFixed(0)
  return String(Number(v.toPrecision(3)))
}

function fmtClock(t, seconds) {
  const d = new Date(t * 1000)
  let hh = d.getHours()
  const ap = hh >= 12 ? 'PM' : 'AM'
  hh = hh % 12 || 12
  const mm = String(d.getMinutes()).padStart(2, '0')
  const ss = seconds ? `:${String(d.getSeconds()).padStart(2, '0')}` : ''
  return `${String(hh).padStart(2, '0')}:${mm}${ss} ${ap}`
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
function fmtDay(t) {
  const d = new Date(t * 1000)
  return `${MONTHS[d.getMonth()]} ${d.getDate()}`
}

function fmtDur(s) {
  if (s === null || s === undefined) return ''
  if (s < 90) return `${Math.round(s)}s`
  if (s < 5400) return `${Math.round(s / 60)} min`
  return `${(s / 3600).toFixed(1)} h`
}

function ago(t, now) {
  const s = Math.max(0, (now || Date.now() / 1000) - t)
  if (s < 60) return 'just now'
  if (s < 3600) return `${Math.round(s / 60)} min ago`
  if (s < 86400) return `${Math.round(s / 3600)} h ago`
  return fmtDay(t)
}

// ---- primitives -------------------------------------------------------------------------------------------
const PILL = {
  failed: [C.red, 'Failed'],
  critical: [C.red, 'Critical'],
  warning: [C.yellow, 'Warning'],
  warn: [C.yellow, 'Warning'],
  success: [C.green, 'Success'],
  recovered: [C.green, 'Recovered'],
  info: ['#d4d4d4', 'Info'],
  new: ['#e6e6e6', 'New'],
  investigating: ['#e6e6e6', 'Investigating'],
  running: [C.green, 'Running'],
  pending: [C.lilac, 'Awaiting approval']
}

function Pill({ kind, label }) {
  const [color, text] = PILL[kind] || PILL.info
  const tinted = color !== '#e6e6e6' && color !== '#d4d4d4'
  return h('span', { style: { display: 'inline-flex', alignItems: 'center', height: 24, padding: '0 8px',
    borderRadius: 7, fontSize: 13, color, whiteSpace: 'nowrap',
    border: `1px solid ${tinted ? color + '66' : '#4a4a4a'}`, background: tinted ? color + '12' : '#232323' } },
  label || text)
}

function IconBox({ name, color, size }) {
  const s = size || 36
  return h('span', { style: { width: s, height: s, flex: 'none', display: 'inline-flex', alignItems: 'center',
    justifyContent: 'center', borderRadius: 10, border: '1px solid #333',
    background: 'linear-gradient(180deg, #2b2b2b 0%, #1f1f1f 100%)', boxShadow: '0 1px 2px rgba(0,0,0,0.4)' } },
  h(Icon, { name, size: s * 0.5, color }))
}

function Card({ children, style, pad }) {
  return h('div', { style: { background: C.card, border: `1px solid ${C.line}`, borderRadius: 18,
    padding: pad === undefined ? 20 : pad, ...style } }, children)
}

function Logo({ size }) {
  const s = size || 30
  return h('span', { style: { width: s, height: s, flex: 'none', borderRadius: 8, background: '#fafafa',
    display: 'inline-flex', alignItems: 'center', justifyContent: 'center' } },
  h(Icon, { name: 'moon', size: s * 0.62, color: '#141414', sw: 2.2, fill: '#141414' }))
}
