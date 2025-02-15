import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random
from datetime import datetime

class ConstructionEnv(gym.Env):
    def __init__(self):
        super(ConstructionEnv, self).__init__()

        # Define state space
        self.observation_space = spaces.Dict({
            "activity_status": spaces.Box(low=0, high=2, shape=(6,), dtype=np.int32),
            "inventory": spaces.Box(low=0, high=1000, shape=(3,), dtype=np.int32),
            "installed_material": spaces.Box(low=0, high=1000, shape=(3,), dtype=np.int32),
            "productivity": spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32),
        })

        # Define action space
        self.action_space = spaces.Tuple((
            spaces.Discrete(6),  # Choose one of 6 activities
            spaces.Discrete(101)  # Produce between 0-100 units
        ))

        # Define dependencies
        self.activities = [
            {"id": 0, "type": "production", "material": 0, "requirement": 15, "planned_duration": 3, "planned_start": "5-Aug-24", "planned_end": "7-Aug-24", "planned_productivity": 5.00, "predecessor": [], "successor": [1]},
            {"id": 1, "type": "installation", "material": 0, "requirement": 15, "planned_duration": 5, "planned_start": "8-Aug-24", "planned_end": "12-Aug-24", "planned_productivity": 3.00, "predecessor": [0], "successor": [3]},
            {"id": 2, "type": "production", "material": 1, "requirement": 25, "planned_duration": 3, "planned_start": "5-Aug-24", "planned_end": "7-Aug-24", "planned_productivity": 8.33, "predecessor": [], "successor": [3]},
            {"id": 3, "type": "installation", "material": 1, "requirement": 25, "planned_duration": 6, "planned_start": "13-Aug-24", "planned_end": "18-Aug-24", "planned_productivity": 4.17, "predecessor": [1, 2], "successor": [5]},
            {"id": 4, "type": "production", "material": 2, "requirement": 20, "planned_duration": 5, "planned_start": "5-Aug-24", "planned_end": "9-Aug-24", "planned_productivity": 4.00, "predecessor": [], "successor": [5]},
            {"id": 5, "type": "installation", "material": 2, "requirement": 20, "planned_duration": 8, "planned_start": "19-Aug-24", "planned_end": "26-Aug-24", "planned_productivity": 2.50, "predecessor": [3, 4], "successor": []}
        ]

        # Initialize state
        self.state = {
            "activity_status": np.zeros(6, dtype=np.int32),
            "inventory": np.zeros(3, dtype=np.int32),
            "installed_material": np.zeros(3, dtype=np.int32),
            "productivity": np.array([1.0, 1.0, 1.0])
        }

        self.cumulative_reward = 0  # Track cumulative reward
        self.excess_inventory_penalty = np.zeros(3, dtype=np.int32)  # Track excess inventory penalty per material

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = {
            "activity_status": np.zeros(6, dtype=np.int32),
            "inventory": np.zeros(3, dtype=np.int32),
            "installed_material": np.zeros(3, dtype=np.int32),
            "productivity": np.array([1.0, 1.0, 1.0])
        }
        self.cumulative_reward = 0  # Reset cumulative reward
        self.excess_inventory_penalty = np.zeros(3, dtype=np.int32)  # Reset penalty tracking

        # Extract the earliest planned start date from activities
        planned_start_dates = [datetime.strptime(act["planned_start"], "%d-%b-%y") for act in self.activities]
        self.simulation_date = min(planned_start_dates)  # Set as the earliest date

        return self.state, {}

    def step(self, action):
        activity_idx, _ = action
        activity = self.activities[activity_idx]
        reward, done = 0, False
        reward_reason = ""

        # Ensure the activity is not already completed
        if self.state["activity_status"][activity_idx] == 2:
            reward = -5
            reward_reason = "Penalty for repeating completed activity"
            return self.state, reward, False, False, {}

        # Choose random quantity between 1 and the max requirement
        quantity = random.randint(1, activity["requirement"])

        if activity["type"] == "production":
            material_idx = activity["material"]
            self.state["inventory"][material_idx] += quantity
            reward = 5 * quantity
            reward_reason = f"Production reward: 5 * {quantity}"

            # Apply penalty for excess inventory and persist it
            excess_inventory = max(0, self.state["inventory"][material_idx] - activity["requirement"])
            if excess_inventory > 0:
                self.excess_inventory_penalty[material_idx] = 7 * excess_inventory  # Store penalty for future steps

            # Mark production as in-progress if not enough material is produced yet
            if self.state["inventory"][material_idx] < activity["requirement"]:
                self.state["activity_status"][activity_idx] = 1  # In progress
            else:
                self.state["activity_status"][activity_idx] = 2  # Completed

        elif activity["type"] == "installation":
            material_idx = activity["material"]

            # Ensure installation can be conducted as long as any material is available
            if self.state["inventory"][material_idx] > 0:
                install_qty = min(quantity, self.state["inventory"][material_idx])  # Ensure we don't install more than available
                self.state["inventory"][material_idx] -= install_qty
                self.state["installed_material"][material_idx] += install_qty
                reward = 10 * install_qty
                reward_reason = f"Installation reward: 10 * {install_qty}"

                # Check if installed material exceeds requirement
                excess_material = max(0, self.state["installed_material"][material_idx] - activity["requirement"])
                if excess_material > 0:
                    reward -= 7 * excess_material  # Apply penalty
                    reward_reason += f", Excess installation penalty: -7 * {excess_material}"
                    self.state["installed_material"][material_idx] = activity["requirement"]  # Limit installed material

                # Mark installation as in-progress if not enough material is installed yet
                if self.state["installed_material"][material_idx] < activity["requirement"]:
                    self.state["activity_status"][activity_idx] = 1  # In progress
                else:
                    self.state["activity_status"][activity_idx] = 2  # Completed
            else:
                reward = -10  # Penalty for insufficient material
                reward_reason = "Penalty for insufficient material"

        # Apply excess inventory penalty from previous steps (persisting penalty)
        penalty_sum = np.sum(self.excess_inventory_penalty)
        if penalty_sum > 0:
            reward -= penalty_sum
            reward_reason += f", Recurring excess inventory penalty: -{penalty_sum}"

        self.cumulative_reward += reward  # Update cumulative reward
        done = all(status == 2 for status in self.state["activity_status"])

        print(f"Step: {step_count}, ({activity_idx}, {quantity}), Type: {activity['type']}, Status: {self.state['activity_status']}, Inventory: {self.state['inventory']}, Installed Material: {self.state['installed_material']}, Reward: {reward}, Reason: {reward_reason}, Cumulative Reward: {self.cumulative_reward}")

        return self.state, reward, done, False, {}

    def render(self):
        print(f"Activity Status: {self.state['activity_status']}")
        print(f"Inventory: {self.state['inventory']}")
        print(f"Installed Material: {self.state['installed_material']}")
        print(f"Productivity: {self.state['productivity']}")
        print(f"Cumulative Reward: {self.cumulative_reward}")

# Register environment
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