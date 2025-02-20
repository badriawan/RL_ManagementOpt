# Register environment
import gymnasium as gym
from construction_env import ConstructionEnv  # Ensure correct module name
gym.register(id='ConstructionEnv-v0', entry_point=ConstructionEnv)

# Run until all activities finish
env = gym.make("ConstructionEnv-v0")

state, _ = env.reset()
done = False
step_count = 0

print("Episode 1:")
while not done:
    step_count += 1
    action = (env.action_space.sample()[0], 0)  # Selecting only activity index
    state, reward, done, _, _ = env.step(action)
print("All activities completed.")