import os
import gymnasium as gym
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from construction_env import ConstructionEnv  # Import your custom environment

# Register environment
gym.register(id="ConstructionEnv-v0", entry_point="construction_env:ConstructionEnv")

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create vectorized environment
env = make_vec_env("ConstructionEnv-v0", n_envs=1, monitor_dir="./tensorboard_logs/")

# Set up logging and model saving
log_dir = "./tensorboard_logs/"
os.makedirs(log_dir, exist_ok=True)

save_dir = "./models/"
os.makedirs(save_dir, exist_ok=True)

# **Check if there is an existing model to continue training**
model_path = os.path.join(save_dir, "dqn_construction_final.zip")
resumed_training = False  # Flag to track if training was resumed

if os.path.exists(model_path):
    print("🔄 Training continues from previous model...")
    model = DQN.load(model_path, env=env, tensorboard_log=log_dir, device=device)
    resumed_training = True  # Mark that training is resuming
else:
    print("🆕 Starting new training session...")
    model = DQN(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=5e-4,
        buffer_size=10_000,  # Increased buffer size to allow more training data
        learning_starts=500,  # Starts training earlier to avoid slow progress
        batch_size=32,  # Standard batch size for stability
        gamma=0.95,
        exploration_fraction=0.1,
        exploration_final_eps=0.1,
        target_update_interval=500,
        train_freq=1,  # Update every step to ensure continuous learning
        gradient_steps=4,  # More frequent updates
        verbose=1,
        tensorboard_log=log_dir,
        device=device,
    )

# Set up callbacks
checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=save_dir, name_prefix="dqn_construction")
eval_callback = EvalCallback(env, best_model_save_path=save_dir, log_path=log_dir, eval_freq=5000, deterministic=True, render=False)
callbacks = CallbackList([checkpoint_callback, eval_callback])

# **Ensure training continues for exactly 10000 more steps**
initial_steps = model.num_timesteps  # Get current step count
target_steps = initial_steps + 6000  # Ensure 10000 more steps

print(f"🚀 Continuing training from {initial_steps} steps until {target_steps} steps...")

while model.num_timesteps < target_steps:
    model.learn(total_timesteps=target_steps, callback=callbacks)

# Save updated model
model.save(model_path)

# **Print message at the END of training**
if resumed_training:
    print(f"✅ Training continues from previous model. Total steps: {model.num_timesteps}")
else:
    print(f"✅ Starting new training session. Total steps: {model.num_timesteps}")

print(f"🏁 Training complete. Final total steps: {model.num_timesteps}")
