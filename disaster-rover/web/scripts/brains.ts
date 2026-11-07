import type { SensorFrame, BrainOutput } from '../brain/msgs';
import { WanderBrain } from '../brain/types';
import type { Sim } from '../sim/sim';

export interface RunnerBrain {
  name: string;
  tick(f: SensorFrame): BrainOutput | Promise<BrainOutput>;
  close?(): Promise<void> | void;
  stats?(): Record<string, unknown>;
}

export async function makeBrain(kind: string, _args: Record<string, string>, _sim: Sim): Promise<RunnerBrain> {
  if (kind === 'wander') return new WanderBrain();
  throw new Error(`unknown brain ${kind}`);
}
