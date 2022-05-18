# Import the RL algorithm (Trainer) we would like to use.
from ray.rllib.agents.a3c import A3CTrainer
from ray.rllib.agents.trainer import with_common_config
import argparse

ENVS = ['FetchReach-v1', 'FetchSlide-v1', 'FetchPush-v1', 'FetchPickAndPlace-v1']


def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-env', '--environment', help='Environment name', choices=ENVS, required=True)

    args = parser.parse_args()    

    # Configure the algorithm.
    config = with_common_config({
        # Environment (RLlib understands openAI gym registered strings).
        "env": args.environment,
        # Should use a critic as a baseline (otherwise don't use value baseline; required for using GAE).
        "use_critic": True,
        # If true, use the Generalized Advantage Estimator (GAE) with a value function.
        "use_gae": True,
        # Size of rollout batch
        "rollout_fragment_length": 256,
        "train_batch_size": 256,
        # GAE(gamma) parameter
        "lambda": 0.99,
        # Max global norm for each gradient calculated by worker
        "grad_clip": 40.0,
        # Learning rate
        "lr": 0.001,
        # Learning rate schedule
        "lr_schedule": None,
        # Value Function Loss coefficient
        "vf_loss_coeff": 5.0,
        # Entropy coefficient
        "entropy_coeff": 0.1,
        # Entropy coefficient schedule
        "entropy_coeff_schedule": None,
        
        # Workers sample async. Note that this increases the effective
        # rollout_fragment_length by up to 5x due to async buffering of batches.
        #"sample_async": True, # gym raises an error with this line
        
        # Use the Trainer's `training_iteration` function instead of `execution_plan`.
        # Fixes a severe performance problem with A3C. Setting this to True leads to a
        # speedup of up to 3x for a large number of workers and heavier
        # gradient computations (e.g. ray/rllib/tuned_examples/a3c/pong-a3c.yaml)).
        "_disable_execution_plan_api": True,
        # Use 4 environment workers (aka "rollout workers") that parallelly
        # collect samples from their own environment clone(s).
        "num_workers": 4,
        # Using "framework: tf2" for tf2.x eager execution.
        "framework": "tf2",
        "evaluation_num_workers": 0,
        # dont render env while training
        "render_env": False,
        "evaluation_interval": 100000,
        # Only for evaluation runs, render the env.
        "evaluation_config": {
            "render_env": True,
        },
        # if gym fails because of joint 0 this line is the best fix found
        #"horizon": 15,
        # uncomment to record training
        #"record_env": True
    })

    # Create our RLlib Trainer.
    trainer = A3CTrainer(config=config)

    # Run it for n training iterations. A training iteration includes
    # parallel sample collection by the environment workers as well as
    # loss calculation on the collected batch and a model update.
    for _ in range(10):
        print("Iteration:", _)
        stats = trainer.train()
        print("Episode Reward Mean:", stats["sampler_results"]["episode_reward_mean"])
        print("Episode Reward Min:", stats["sampler_results"]["episode_reward_min"])
        print("Episode Reward Max:", stats["sampler_results"]["episode_reward_max"])

    input("Finished training... (Press a key to render)")

    # Evaluate the trained Trainer (and render each timestep).
    trainer.evaluate()


"""
MAIN
"""
if __name__ == '__main__':
    main()
