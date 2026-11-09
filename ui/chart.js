// ---- chart: loss against training step, with the learning rate on the right axis ------------------------------
// After a rollback the new attempt restarts at the checkpoint's step and is drawn over the branch that diverged,
// so the chart shows the fork. Lines live in a stretched SVG (non-scaling strokes); labels, dots and callouts are
// HTML placed by percentage, so they stay crisp at any width.

function finite(p) { return typeof p.value === 'number' && isFinite(p.value) }

function niceTicks(lo, hi, n) {
  const span = hi - lo || 1
  const raw = span / (n - 1)
  const mag = Math.pow(10, Math.floor(Math.log10(raw)))
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw) || raw
  const start = Math.floor(lo / step) * step
  const out = []
  for (let v = start; v <= hi + step * 0.5 && out.length < n + 2; v += step) out.push(Number(v.toPrecision(6)))
  return out
}

function LossChart({ run, range, height }) {
  const H = height || 250
  const train = (run.series && run.series.train) || []
  const val = (run.series && run.series.val) || []
  const lr = (run.series && run.series.lr) || []
  if (train.filter(finite).length < 2) {
    return h('div', { style: { height: H, display: 'flex', alignItems: 'center', justifyContent: 'center',
      color: C.muted, fontSize: 14 } }, run.status === 'pending' ? 'Starts after you approve it.' : 'Waiting for metrics…')
  }
  const markers = (run.markers || []).filter(m => typeof m.step === 'number')
  const restores = (run.recoveries || []).filter(r => typeof r.step === 'number')
  let x0 = Math.min(...train.map(p => p.step))
  let x1 = Math.max(...train.map(p => p.step), ...markers.map(m => m.step))
  if (range === 'recent' && x1 - x0 > 1000) x0 = x1 - 1000
  if (range === 'incident' && markers.length) {
    const m = markers[markers.length - 1].step
    x0 = Math.max(x0, m - 160)
    x1 = Math.min(x1, m + 320)
  }
  if (x1 <= x0) x1 = x0 + 1
  const inX = p => p.step >= x0 && p.step <= x1
  const tr = train.filter(inX)
  const lastAttempt = Math.max(0, ...train.map(p => p.attempt || 0))

  // Fit the axes to the healthy curve: after a recovery, the branch that diverged (an earlier attempt, past the
  // anomaly's onset) is left out of the fit and runs off the top with a callout, instead of flattening the rest.
  const onset = markers.length ? Math.min(...markers.map(m => m.step)) : Infinity
  const healthy = p => finite(p) && (lastAttempt === 0 || (p.attempt || 0) === lastAttempt || p.step < onset)
  const vals = tr.filter(healthy).map(p => p.value).concat(val.filter(inX).filter(healthy).map(p => p.value))
  const sorted = (vals.length ? vals : tr.filter(finite).map(p => p.value)).slice().sort((a, b) => a - b)
  const q = f => sorted[Math.min(sorted.length - 1, Math.floor(f * (sorted.length - 1)))]
  const cap = lastAttempt === 0 && sorted.length > 10 && q(1) > 3 * q(0.9) ? q(0.9) * 1.5 : q(1)
  const ticks = niceTicks(Math.min(...sorted), cap, 5)
  const y0 = ticks[0]
  const y1 = ticks[ticks.length - 1]
  const X = s => ((s - x0) / (x1 - x0)) * 1000
  const Y = v => (1 - (Math.min(v, y1 * 3) - y0) / (y1 - y0)) * 100
  const pct = s => `${(X(s) / 10).toFixed(2)}%`

  const lrIn = lr.filter(inX).filter(finite).filter(p => p.value < 1e3)
  const lrFit = lrIn.filter(healthy).map(p => p.value)
  const lrTicks = lrFit.length ? niceTicks(0, Math.max(...lrFit), 4).filter(t => t >= 0).reverse() : []
  const lrCap = lrTicks.length ? lrTicks[0] : 0
  const LY = v => (1 - Math.min(v, lrCap * 3) / lrCap) * 100

  const segments = (points, fy) => {
    const out = []
    let seg = []
    let att = null
    points.forEach(p => {
      if (!finite(p) || p.attempt !== att) {
        if (seg.length > 1) out.push({ att, pts: seg })
        seg = []
      }
      att = p.attempt
      if (finite(p)) seg.push(`${X(p.step).toFixed(1)},${fy(p.value).toFixed(2)}`)
    })
    if (seg.length > 1) out.push({ att, pts: seg })
    return out
  }
  const trSegs = segments(tr, Y)
  const current = trSegs.filter(s => (s.att || 0) === lastAttempt)
  const area = current.length
    ? current.map(s => `M${s.pts[0].split(',')[0]},100 L${s.pts.join(' L')} L${s.pts[s.pts.length - 1].split(',')[0]},100 Z`).join(' ')
    : ''

  // where an earlier attempt shot past the top of the axis, say how far (once, at its peak)
  const over = tr.filter(p => (p.attempt || 0) < lastAttempt && finite(p) && p.value > y1 * 1.04)
  const peaks = over.length ? [over.reduce((a, b) => (b.value > a.value ? b : a))] : []
  const nanAt = tr.filter(p => !finite(p) && (p.attempt || 0) < lastAttempt).slice(0, 1)

  const dots = restores.filter(r => r.step >= x0 && r.step <= x1).map(r => {
    const after = train.filter(p => (p.attempt || 0) > 0 && p.step >= r.step && finite(p)).sort((a, b) => a.step - b.step)[0]
    return after ? { step: after.step, value: after.value } : null
  }).filter(Boolean)

  const xt = niceTicks(x0, x1, 12).filter(s => s >= x0 && s <= x1)
  const label = { position: 'absolute', fontSize: 12.5, color: C.muted, ...NUM, whiteSpace: 'nowrap' }

  return h('div', { style: { position: 'relative', height: H + 30, paddingLeft: 46, paddingRight: lrTicks.length ? 52 : 8 } },
    // y labels
    ...ticks.map((t, i) => h('span', { key: `y${i}`, style: { ...label, left: 0, width: 40, textAlign: 'right',
      top: `calc(${Y(t) * H / 100}px - 8px)` } }, fmtNum(t))),
    ...lrTicks.map((t, i) => h('span', { key: `l${i}`, style: { ...label, right: 0, width: 46, textAlign: 'right',
      top: `calc(${LY(t) * H / 100}px - 8px)` } }, fmtNum(t))),
    h('div', { style: { position: 'relative', height: H } },
      h('svg', { viewBox: '0 0 1000 100', preserveAspectRatio: 'none', style: { position: 'absolute', inset: 0,
        width: '100%', height: '100%', overflow: 'hidden' } },
      h('defs', null, h('linearGradient', { id: `nsfill-${run.id}`, x1: 0, y1: 0, x2: 0, y2: 1 },
        h('stop', { offset: '0%', stopColor: C.lilac, stopOpacity: 0.14 }),
        h('stop', { offset: '100%', stopColor: C.lilac, stopOpacity: 0 }))),
      ...xt.map((s, i) => h('line', { key: `g${i}`, x1: X(s), x2: X(s), y1: 0, y2: 100, stroke: '#262626',
        strokeWidth: 1, vectorEffect: 'non-scaling-stroke' })),
      area ? h('path', { d: area, fill: `url(#nsfill-${run.id})`, stroke: 'none' }) : null,
      ...segments(lrIn, LY).map((s, i) => h('polyline', { key: `lr${i}`, points: s.pts.join(' '), fill: 'none',
        stroke: '#45434d', strokeWidth: 1.1, vectorEffect: 'non-scaling-stroke' })),
      ...segments(val.filter(inX), Y).map((s, i) => h('polyline', { key: `v${i}`, points: s.pts.join(' '), fill: 'none',
        stroke: '#8d7bb0', strokeWidth: 1.3, strokeDasharray: '3 4', vectorEffect: 'non-scaling-stroke' })),
      ...trSegs.map((s, i) => h('polyline', { key: `t${i}`, points: s.pts.join(' '), fill: 'none', stroke: C.lilac,
        strokeOpacity: (s.att || 0) < lastAttempt ? 0.38 : 1, strokeWidth: 1.7, strokeLinejoin: 'round',
        vectorEffect: 'non-scaling-stroke' })),
      ...markers.filter(m => m.step >= x0 && m.step <= x1).map((m, i) => h('line', { key: `m${i}`, x1: X(m.step),
        x2: X(m.step), y1: 0, y2: 100, stroke: C.redLine, strokeWidth: 1.3, strokeDasharray: '2 4',
        vectorEffect: 'non-scaling-stroke' })),
      ...restores.filter(r => r.step >= x0 && r.step <= x1).map((r, i) => h('line', { key: `r${i}`, x1: X(r.step),
        x2: X(r.step), y1: 0, y2: 100, stroke: C.amberLine, strokeWidth: 1.3, strokeDasharray: '2 4',
        vectorEffect: 'non-scaling-stroke' }))),
      ...dots.map((d, i) => h('span', { key: `d${i}`, style: { position: 'absolute', left: pct(d.step),
        top: `${Y(d.value)}%`, width: 14, height: 14, marginLeft: -7, marginTop: -7, borderRadius: 7,
        background: '#fff', border: '3px solid #151515', boxShadow: `0 0 0 1px ${C.amberLine}66` } })),
      ...peaks.map((c, i) => h('span', { key: `c${i}`, style: { position: 'absolute', left: pct(c.step), top: 6,
        marginLeft: 8, padding: '2px 7px', borderRadius: 6, fontSize: 12, color: C.red, background: '#2a1a1f',
        border: `1px solid ${C.red}55`, ...NUM, whiteSpace: 'nowrap' } }, `↑ ${fmtNum(c.value)} off the chart`)),
      ...nanAt.map((p, i) => h('span', { key: `n${i}`, style: { position: 'absolute', left: pct(p.step), top: -2,
        transform: 'translate(-50%, -100%)', fontSize: 11.5, color: C.red } }, 'NaN'))),
    // x labels
    h('div', { style: { position: 'relative', height: 30 } },
      ...xt.map((s, i) => h('span', { key: `x${i}`, style: { ...label, top: 12, left: pct(s),
        transform: 'translateX(-50%)' } }, s >= 10000 ? `${(s / 1000).toFixed(0)}k` : String(s)))))
}

function ChartLegend({ run }) {
  const has = k => ((run.series && run.series[k]) || []).length > 0
  const key = (color, text, dash) => h('span', { style: { display: 'inline-flex', alignItems: 'center', gap: 6 } },
    h('span', { style: { width: 14, borderTop: `2px ${dash ? 'dashed' : 'solid'} ${color}` } }), text)
  return h('div', { style: { display: 'flex', flexWrap: 'wrap', gap: 14, fontSize: 12.5, color: C.muted } },
    key(C.lilac, (run.series_names && run.series_names.train) || 'loss'),
    has('val') ? key('#8d7bb0', (run.series_names && run.series_names.val) || 'val', true) : null,
    has('lr') ? key('#4d4b57', 'learning rate (right)') : null,
    (run.markers || []).length ? key(C.redLine, 'anomaly began', true) : null,
    (run.recoveries || []).some(r => typeof r.step === 'number') ? key(C.amberLine, 'restored from', true) : null)
}
