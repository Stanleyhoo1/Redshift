import { useViewMode, viewMode } from './viewMode.js'

// View: Sector | Planet, laid out like the Graphics row below it (ViewControls puts it
// first in the Graphics group, so the Follow buttons keep their places). G does the same.
export function ViewModeControl() {
  const view = useViewMode((s) => s.view)
  return (
    <div className="gfx-row">
      <div className="hud-label" id="viewmode-label">
        View
      </div>
      <div className="seg gfx-seg" role="group" aria-labelledby="viewmode-label">
        <button type="button" aria-pressed={view === 'plane'} onClick={() => viewMode.request('plane')} title="The landing sector the rovers are mapping (G)">
          Sector
        </button>
        <button type="button" aria-pressed={view === 'globe'} onClick={() => viewMode.request('globe')} title="The whole planet, with the landing site marked (G)">
          Planet
        </button>
      </div>
    </div>
  )
}
