/**
 * Nightshift for Hermes Desktop: a live pane for watched training runs and a status-bar chip.
 *
 * Disk plugins load uncompiled: no JSX, imports limited to @hermes/plugin-sdk, react, react/jsx-runtime.
 * Data comes from the plugin's backend routes (dashboard/plugin_api.py) via ctx.rest('/status').
 */

import { STATUSBAR_AREAS, Tip } from '@hermes/plugin-sdk'
import { jsx, jsxs } from 'react/jsx-runtime'
import { useEffect, useState } from 'react'

const POLL_MS = 2000

function useStatus(rest) {
  const [status, setStatus] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    let alive = true
    const tick = async () => {
      try {
        const res = await rest('/status')
        if (alive) {
          setStatus(res)
          setError(null)
        }
      } catch (err) {
        if (alive) setError(String(err && err.message ? err.message : err))
      }
    }
    tick()
    const id = setInterval(tick, POLL_MS)
    return () => {
      alive = false
      clearInterval(id)
    }
  }, [rest])

  return { status, error }
}

function Pane({ rest }) {
  const { status, error } = useStatus(rest)
  const runs = (status && status.runs) || []
  return jsxs('div', {
    className: 'flex h-full flex-col gap-2 p-3 text-sm',
    style: { color: 'var(--ui-text-secondary)' },
    children: [
      jsx('div', {
        className: 'text-xs uppercase tracking-wide',
        style: { color: 'var(--ui-text-quaternary)' },
        children: `Nightshift ${status ? 'v' + status.version : ''}`
      }),
      error
        ? jsx('div', { style: { color: 'var(--ui-text-tertiary)' }, children: `Not connected: ${error}` })
        : jsx('div', { children: runs.length ? `${runs.length} run(s) watched` : 'No runs watched yet.' })
    ]
  })
}

function Chip({ rest }) {
  const { status, error } = useStatus(rest)
  const n = (status && status.runs && status.runs.length) || 0
  const label = error ? 'Nightshift offline' : n ? `${n} run(s) watched` : 'Nightshift idle'
  return jsx(Tip, {
    label,
    children: jsx('span', {
      className: 'inline-flex h-full items-center gap-1 px-1.5 text-[0.6875rem]',
      style: { color: error ? 'var(--ui-text-quaternary)' : 'var(--ui-accent)' },
      children: `☾ ${n}`
    })
  })
}

export default {
  id: 'nightshift',
  name: 'Nightshift',
  register(ctx) {
    ctx.register({
      id: 'pane',
      area: 'panes',
      title: 'Nightshift',
      data: { placement: 'right', width: '320px' },
      render: () => jsx(Pane, { rest: ctx.rest })
    })
    ctx.register({
      id: 'chip',
      area: STATUSBAR_AREAS.right,
      order: 125,
      render: () => jsx(Chip, { rest: ctx.rest })
    })
  }
}
