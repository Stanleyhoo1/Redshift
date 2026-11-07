/**
 * Headless sim driven by the ROS 2 brain over rosbridge (lockstep). Start the ROS side first:
 *   ros2 launch disaster_brain rover.launch.py use_agent:=false
 *   timeout 300 npm run sim:ros -- --disasters flood,fire --seed 3 --duration 120
 */
if (!process.argv.includes('--brain')) process.argv.push('--brain', 'ros');
await import('./run-sim');
export {};
