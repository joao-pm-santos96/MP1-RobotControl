import gym
from stable_baselines3 import SAC, HerReplayBuffer

# Init env
env = gym.make('FetchSlide-v1')

# Init model
model = SAC(
    "MultiInputPolicy",
    env,
    replay_buffer_class = HerReplayBuffer,
    learning_rate = 0.001,
    tau = 0.005,
    gamma = 0.99,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy='future',
        online_sampling=True
    ),
    verbose=1,
)

# Train the model
model.learn(50000, log_interval=1)

# Save the model
model.save("./her")


