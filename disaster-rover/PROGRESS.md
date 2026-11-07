# Disaster Rover: progress

## Done
- M1: Vite + TS + Three.js scaffold, pure-TS headless sim core (`web/sim`), rover diff-drive physics, manual WASD driving, headless runner (`npm run sim`).

## Next
- M2 terrain/fog polish, M3 brain.

## Blockers
- **Pushing to GitHub was blocked.** The repo `redshift` is not under `kayaclaridge-sys`; the only match is `Stanleyhoo1/Redshift` (you have WRITE access, found via `gh api user/repos`). Cloning and the local branch `naturaldisasterdemo` worked, but `git push -u origin naturaldisasterdemo` was refused by the Claude Code permission guard ("create public surface" on a repo owned by another account). All milestone commits are on the local `naturaldisasterdemo` branch in `~/redshift`. To publish: `cd ~/redshift && git push -u origin naturaldisasterdemo`.
