# MP1-RobotControl
Robot control (Gym)

## Usage

### Train an Environment

This code will start training the agent for the selected environment. It will use the \<EnvName\>.yaml file as the hyper-parameters.

It will, periodically and at the end, save the model in the *models* folder and generate logs on the *logs* folder.

```
python her.py -env <EnvName> -t
```

### Run a model

This command will run a selected model on the choice environment. By default, they are inside the *models* folder.

```
python her.py -env <EnvName> -m <ModelPath>
```

### Check the training progress

```
tensorboard --logdir=logs
```

## TODO

- [ ] Add algorithm to hyperparams.yaml

## Links

- [Docs](https://gym.openai.com/docs/)
- [Environment](https://gym.openai.com/envs/FetchSlide-v1/)

https://openai.com/blog/ingredients-for-robotics-research/

https://github.com/openai/baselines/tree/master/baselines/her

https://stable-baselines3.readthedocs.io/en/master/modules/her.html

## Papers

https://arxiv.org/pdf/1802.09464.pdf


