# MP1-RobotControl
Robot control (Gym)

## Models and videos

https://drive.google.com/drive/folders/1C2MNLSTiROR85qicB8NQyKLJjCt2M76V?usp=sharing

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


## Links

- [Docs](https://gym.openai.com/docs/)
- [Environment](https://gym.openai.com/envs/FetchSlide-v1/)
- [Multiple Environments (more detailed)](https://www.cnblogs.com/siahekai/p/14161023.html)

https://openai.com/blog/ingredients-for-robotics-research/

https://github.com/openai/baselines/tree/master/baselines/her

https://stable-baselines3.readthedocs.io/en/master/modules/her.html

## Papers

https://arxiv.org/pdf/1802.09464.pdf


## Presentation (Access Required)
(may contain slides from previous work related to box2d environment)

https://docs.google.com/presentation/d/16Df9HLZiZi2XrAyTAo7vK3Ek_FmHXVT-aNOLCnBCMYA/edit?usp=sharing


