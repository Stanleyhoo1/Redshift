import { ATMOSPHERE, REGOLITH, SIGNAL, VOID } from '../lib/theme.js'

// The planet view's colours (sRGB hex). The ground, the air and the light are the sector's
// own theme colours, so from orbit it reads as the same world; only the ice, the night fill
// and the star tints are new.
export const GLOBE = {
  void: VOID,
  dust: REGOLITH.dust,
  dustLight: REGOLITH.dustLight,
  rust: REGOLITH.strataA,
  mare: REGOLITH.soil,
  mareDark: REGOLITH.strataB,
  ice: '#eef1f2',
  sunLight: ATMOSPHERE.sunLight,
  twilight: ATMOSPHERE.sunGlow,
  haze: ATMOSPHERE.haze,
  airLow: ATMOSPHERE.horizon,
  airHigh: ATMOSPHERE.sky,
  airDusk: ATMOSPHERE.zenith,
  night: '#2a3346', // starlight on the night side, cool and very dim
  signal: SIGNAL,
  starWarm: '#ffd9b0',
  starWhite: '#f3f5ff',
  starCool: '#bcd0ff',
}
