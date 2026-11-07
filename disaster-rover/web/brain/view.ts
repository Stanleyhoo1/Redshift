/** Read-only view of a brain's belief, for the renderer / minimap. Sim-grid (100x100) cell indices map onto it via cellOfWorld. */
export interface BeliefView {
  n: number; res: number;
  /** max current risk over all hazard channels, 0..1 */
  risk: Float32Array;
  /** predicted (30 s ahead) risk, 0..1 */
  predicted: Float32Array;
  /** per-channel current risk: water, heat, rubble, wind, slope */
  channels: Float32Array[];
  /** 0 = never sensed */
  conf: Float32Array;
  obst: Float32Array;
  cellOfWorld(simIndex: number): number;
}
