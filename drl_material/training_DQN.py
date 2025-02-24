import os
import gymnasium as gym
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from construction_env import ConstructionEnv  # Import your custom environment



# Otomatis deteksi GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Buat direktori untuk menyimpan log TensorBoard
log_dir = "dqn_construction_v0.1"
os.makedirs(log_dir, exist_ok=True)

# Buat lingkungan
env = ConstructionEnv()

# Inisialisasi model DQN
model = DQN(
    "MultiInputPolicy",  # Policy untuk observasi dictionary
    env,
    verbose=1,
    device=device,
    tensorboard_log=log_dir,  # Simpan log untuk TensorBoard
    learning_rate=0.001,
    buffer_size=10000,
    batch_size=64,
    gamma=0.99,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
)

# Callback untuk evaluasi selama pelatihan
eval_callback = EvalCallback(
    env,
    best_model_save_path=log_dir,
    log_path=log_dir,
    eval_freq=1000,
    deterministic=True,
    render=False,
)

# Latih model
model.learn(total_timesteps=10000, callback=eval_callback)

# Simpan model
model.save("dqn_construction_v0.1")

