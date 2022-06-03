# Distributional Reinforcement Learning with Unconstrained Monotonic Neural Networks
Experimental code supporting the results presented in the scientific research paper:
> Thibaut Théate, Antoine Wehenkel, Adrien Bolland, Gilles Louppe and Damien Ernst. "Distributional Reinforcement Learning with Unconstrained Monotonic Neural Networks." (2021).
> [[arxiv]](https://arxiv.org/abs/)



# Dependencies

The dependencies are listed in the text file "requirements.txt":
* Python 3.7.7
* Pytorch
* Tensorboard
* Gym
* Opencv-python
* Atari-py
* MinAtar
* Umnn
* Numpy
* Pandas
* Matplotlib
* Scipy
* Tqdm



# Usage

Training and testing a chosen distributional RL algorithm for the control problem of a chosen environment is performed by running the following command:

```bash
python main.py -algorithm ALGORITHM -environment ENVIRONMENT
```

with:
* ALGORITHM being the name of the algorithm (by default UMDQN_C),
* ENVIRONMENT being the name of the environment (by default StochasticGridWorld).

The (distributional) RL algorithms supported are:
* DQN,
* CDQN,
* QR_DQN,
* IQN,
* FQF,
* UMDQN_KL,
* UMDQN_C,
* UMDQN_W.

The benchmark environments supported are:
* StochasticGridWorld,
* CartPole-v0,
* Acrobot-v1,
* LunarLander-v2,
* MountainCar-v0,
* MinAtar/Asterix-v0,
* MinAtar/Breakout-v0,
* MinAtar/Freeway-v0,
* MinAtar/Seaquest-v0,
* MinAtar/SpaceInvaders-v0,
* PongNoFrameskip-v4,
* BoxingNoFrameskip-v4,
* FreewayNoFrameskip-v4.

The number of episodes for training the DRL algorithm may also be specified by the user through the argument "-episodes". The parameters of the DRL algorithms can be set with the argument "-parameters" and by providing the name of the .txt file containing these parameters within the "Parameters" folder.

For more advanced tests and manipulations, please directly refer to the code.



# Citation

If you make use of this experimental code, please cite the associated research paper:

```
@inproceedings{Théate2021,
  title={Distributional Reinforcement Learning with Unconstrained Monotonic Neural Networks},
  author={Thibaut Théate, Antoine Wehenkel, Adrien Bolland, Gilles Louppe and Damien Ernst},
  year={2021}
}
```
