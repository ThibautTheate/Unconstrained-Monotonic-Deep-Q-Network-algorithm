# Distributional Reinforcement Learning with Unconstrained Monotonic Neural Networks
Official implementation of the UMDQN algorithm presented in the scientific research paper:
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
* Umnn
* Numpy
* Pandas
* Matplotlib
* Scipy
* TQDM



# Usage

Training and testing a chosen distributional RL algorithm for the control problem of a chosen environment is performed by running the following command:

```bash
python main.py -algorithm ALGORITHM -environment ENVIRONMENT
```

with:
* ALGORITHM being the name of the algorithm (by default UMDQN_C),
* ENVIRONMENT being the name of the environment (by default StochasticGridWorld).

The algorithms supported are DQN, CDQN, QR_DQN, IQN, FQF, UMDQN_KL, UMDQN_C and UMDQN_KL.
The environments supported are "StochasticGridWorld", "PongNoFrameskip-v4", "BoxingNoFrameskip-v4" and "FreewayNoFrameskip-v4".

The number of episodes to be used for training the DRL algorithm may be specified by the user through the argument "-episodes". The parameters of the DRL algorithm can be set with the argument "-parameters" and by providing the name of the .json file containing these parameters.

For more advanced tests and manipulations, please directly refer to the code.



# Citation

If you make use of this experimental code, please cite the associated research paper:

```
@inproceedings{Theate2021,
  title={Distributional Reinforcement Learning with Unconstrained Monotonic Neural Networks},
  author={Thibaut Theate, Antoine Wehenkel, Adrien Bolland, Gilles Louppe and Damien Ernst},
  year={2021}
}
```
