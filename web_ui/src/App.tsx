import { lazy, Suspense, useEffect } from 'react'
import { api } from './lib/api'
import { DEFAULT_OPTIONS } from './lib/defaults'
import { useStore } from './lib/store'
import BootSequence from './screens/BootSequence'
import PreLaunch from './screens/PreLaunch'

// three.js is only needed once a mission is running — keep the config screen light.
const MissionControl = lazy(() => import('./screens/MissionControl'))

export default function App() {
  const screen = useStore((s) => s.screen)

  useEffect(() => {
    const ctrl = new AbortController()
    api
      .options(ctrl.signal)
      .then((o) => useStore.getState().setOptions({ ...DEFAULT_OPTIONS, ...o }, true))
      .catch(() => useStore.getState().setOptions(DEFAULT_OPTIONS, false))
    // pick up a mission that is already running (page reload mid-demo)
    api
      .status(ctrl.signal)
      .then((st) => {
        const live = st.status === 'launching' || st.status === 'running' || st.status === 'degraded'
        if (live && useStore.getState().screen === 'prelaunch') {
          useStore.setState({ lastConfig: st.config })
          useStore.getState().setScreen('boot')
        }
      })
      .catch(() => {})
    return () => ctrl.abort()
  }, [])

  // warm the mission bundle while the operator configures the fleet
  useEffect(() => {
    const t = setTimeout(() => void import('./screens/MissionControl'), 1500)
    return () => clearTimeout(t)
  }, [])

  if (screen === 'prelaunch') return <PreLaunch />
  if (screen === 'boot') return <BootSequence />
  return (
    <Suspense fallback={<div className="grid h-full place-items-center text-haze">Loading mission control…</div>}>
      <MissionControl />
    </Suspense>
  )
}
