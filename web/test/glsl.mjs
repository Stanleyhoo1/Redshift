// Helpers for the material tests: the uniforms a GLSL source declares, and whether its
// brackets pair up (line comments ignored).
export const declaredUniforms = (src) => [...src.matchAll(/^\s*uniform\s+\w+\s+(\w+)\s*;/gm)].map((m) => m[1])

export function balanced(src) {
  const open = { ')': '(', ']': '[', '}': '{' }
  const stack = []
  for (const ch of src.replace(/\/\/.*$/gm, '')) {
    if (ch === '(' || ch === '[' || ch === '{') stack.push(ch)
    else if (open[ch] && stack.pop() !== open[ch]) return false
  }
  return stack.length === 0
}
