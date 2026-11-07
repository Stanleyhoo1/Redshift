import { useEffect, useState } from 'react'

// Wall-clock seconds (performance.now() based), refreshed every `ms` while enabled.
export function useNow(ms, enabled = true) {
  const [now, setNow] = useState(() => performance.now() / 1000)
  useEffect(() => {
    if (!enabled) return
    setNow(performance.now() / 1000)
    const id = setInterval(() => setNow(performance.now() / 1000), ms)
    return () => clearInterval(id)
  }, [ms, enabled])
  return now
}
