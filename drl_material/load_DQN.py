import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random
from datetime import datetime
from stable_baselines3 import DQN
from construction_env import ConstructionEnv  # Import your custom environment




# Load model yang telah dilatih
model = DQN.load("dqn_construction_model")

# Inisialisasi lingkungan
env = ConstructionEnv()

#training 
model.learn(total_timesteps=target_steps,    verbose=1,, tensorboard= , callback=callbacks)

# Save updated model
model.save("dqn_construction_10000.zip")




# # Jalankan simulasi dengan render
# obs, _ = env.reset()
# for _ in range(20):  # Simulasi 20 langkah
# 	action,state = model.predict(obs, deterministic=True)
# 	obs, reward, done, truncated, info = env.step(action)
    
# 	# Render lingkungan
# 	env.render()
    
# 	print(f"Aksi yang diambil: Aktivitas {action[0]}, Jumlah {action[1]}")
# 	print(f"Reward: {reward}")
# 	print("-" * 50)