import gym
from stable_baselines3 import SAC, HerReplayBuffer

# Init env
env = gym.make('FetchSlide-v1')

# Init model
model = SAC(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy='future',
        online_sampling=True
    ),
    verbose=1,
)

# Train the model
model.learn(10000)

# Save the model
model.save("./her")

