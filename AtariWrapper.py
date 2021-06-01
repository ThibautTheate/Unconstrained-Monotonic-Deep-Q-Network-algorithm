# coding=utf-8

###############################################################################
################################### Imports ###################################
###############################################################################

import random
import gym
import cv2

import numpy as np

from collections import deque

cv2.ocl.setUseOpenCL(False)



###############################################################################
############################## Class NoopWrapper ##############################
###############################################################################

class NoopWrapper(gym.Wrapper):
    """
    GOAL: Wrapper for the executation of a random of "No Operation" actions
          when resetting the Atari gym environment.
    
    VARIABLES: - maxNoop: Maximum number of Noop actions to execute at reset.
                         
    METHODS: - __init__: Initialization of the wrapped environment.
             - reset: Reset of the environment, with the execution of Noop actions.
             - step: Execute a specified action in the wrapped environment.
    """

    def __init__(self, env, maxNoop=30):
        """
        GOAL: Initialization of the wrapped environment.
        
        INPUTS: - env: Environment to wrap.
                - maxNoop: Maximum number of Noop actions to execute at reset.
        
        OUTPUTS: /
        """

        gym.Wrapper.__init__(self, env)

        # Initialization of important variables
        self.maxNoop = maxNoop


    def reset(self, **kwargs):
        """
        GOAL: Reset the environment, with the execution of Noop actions.
        
        INPUTS: - kwargs: Parameters for the resetting of the wrapped environment.
        
        OUTPUTS: - state: RL state.
        """

        # Resetting of the wrapped environment
        self.env.reset(**kwargs)

        # Execution of a random number of Noop actions
        numberNoop = np.random.randint(1, self.maxNoop+1)
        for _ in range(numberNoop):
            state, _, done, _ = self.env.step(0)
            if done:
                state = self.env.reset(**kwargs)
        
        return state


    def step(self, action):
        """
        GOAL: Execute a specified action in the wrapped environment.
        
        INPUTS: - action: Action to be executed.
        
        OUTPUTS: - state: RL state.
                 - reward: RL reward.
                 - done: RL episode termination signal.
                 - info: Additional information (optional).
        """

        return self.env.step(action)



###############################################################################
############################## Class SkipWrapper ##############################
###############################################################################

class SkipWrapper(gym.Wrapper):
    """
    GOAL: Wrapper for the skipping of frames, with support for sticky actions.
    
    VARIABLES: - skip: Number of frames to skip.
               - stickyActionsProba: Probability associated with sticky actions.
                         
    METHODS: - __init__: Initialization of the wrapped environment.
             - reset: Reset of the environment.
             - step: Execute the specified action in the wrapped environment.
    """

    def __init__(self, env, skip=4, stickyActionsProba=0):
        """
        GOAL: Initialization of the wrapped environment.
        
        INPUTS: - env: Environment to wrap.
                - skip: Number of frames to skip.
                - stickyActionsProba: Probability associated with sticky actions.
        
        OUTPUTS: /
        """

        gym.Wrapper.__init__(self, env)

        # Initialization of important variables
        self.skip = skip
        self.buffer = np.zeros((2,) + env.observation_space.shape, dtype=env.observation_space.dtype)
        self.stickyActionsProba = stickyActionsProba
        self.previousAction = 1


    def step(self, action):
        """
        GOAL: Execute a specified action in the wrapped environment, taking into
              account the skipped frames and the sticky actions.
        
        INPUTS: - action: Action to be executed.
        
        OUTPUTS: - state: RL state.
                 - reward: RL reward.
                 - done: RL episode termination signal.
                 - info: Additional information (optional).
        """

        # Initialization of variables
        totalReward = 0.0
        
        # Loop through the frames to skip
        for i in range(self.skip):
            
            # Sticky actions
            if random.random() > self.stickyActionsProba:
                state, reward, done, info = self.env.step(action)
                self.previousAction = action
            else:
                state, reward, done, info = self.env.step(self.previousAction)

            # Storing of the two last frames into the buffer
            if i == self.skip-2: self.buffer[0] = state
            if i == self.skip-1: self.buffer[1] = state
            
            # Aggregation of the RL variables
            totalReward += reward

            # Early breaking if termination signal
            if done:
                break

        # Pixel-wise maximum of the last two frames
        newState = self.buffer.max(axis=0)
        
        return newState, totalReward, done, info


    def reset(self, **kwargs):
        """
        GOAL: Reset the environment.
        
        INPUTS: - kwargs: Parameters for the resetting of the wrapped environment.
        
        OUTPUTS: - state: RL state.
        """
        # Resetting of the wrapped environment
        state = self.env.reset(**kwargs)

        # Resetting of the previous action (sticky actions technique)
        self.previousAction = 1

        return state



###############################################################################
######################### Class FireResetWrapper ##############################
###############################################################################

class FireResetWrapper(gym.Wrapper):
    """
    GOAL: Wrapper for the executation of the "Fire" action when resetting the
          Atari gym environment (required to start certain games).
    
    VARIABLES: /
                         
    METHODS: - __init__: Initialization of the wrapped environment.
             - reset: Reset of the environment, with the execution of "Fire" action.
             - step: Execute a specified action in the wrapped environment.
    """

    def __init__(self, env):
        """
        GOAL: Initialization of the wrapped environment.
        
        INPUTS: - env: Environment to wrap.
        
        OUTPUTS: /
        """

        gym.Wrapper.__init__(self, env)


    def reset(self, **kwargs):
        """
        GOAL: Reset the environment, with the execution of "Fire" action.
        
        INPUTS: - kwargs: Parameters for the resetting of the wrapped environment.
        
        OUTPUTS: - state: RL state.
        """

        # Resetting of the wrapped environment
        self.env.reset(**kwargs)

        # Execution of the "Fire" action
        state, _, done, _ = self.env.step(1)
        if done:
            state = self.env.reset(**kwargs)
        
        return state


    def step(self, action):
        """
        GOAL: Execute a specified action in the wrapped environment.
        
        INPUTS: - action: Action to be executed.
        
        OUTPUTS: - state: RL state.
                 - reward: RL reward.
                 - done: RL episode termination signal.
                 - info: Additional information (optional).
        """

        # Normal step function with the chosen action
        state, reward, done, info = self.env.step(action)

        # Execution of the "FIRE" action if a loss of life happens
        if not done and info['lossOfLife'] == True:
            for _ in range(3): # For countering sticky actions
                state, _, done, _ = self.env.step(1)
                if done:
                    return state, reward, done, info

        return state, reward, done, info
            


###############################################################################
######################### Class LossOfLifeWrapper #############################
###############################################################################

class LossOfLifeWrapper(gym.Wrapper):
    """
    GOAL: Wrapper for making available the knowledge regarding the loss of life
          in certain Atari games.
    
    VARIABLES: numberOfLives: Current number of lives for the agent.
                         
    METHODS: - __init__: Initialization of the wrapped environment.
             - reset: Reset of the wrapped environment.
             - step: Execute a specified action in the wrapped environment.
    """

    def __init__(self, env):
        """
        GOAL: Initialization of the wrapped environment.
        
        INPUTS: - env: Environment to wrap.
        
        OUTPUTS: /
        """

        gym.Wrapper.__init__(self, env)

        # Initialization of important variables
        self.numberOfLives = 0


    def reset(self, **kwargs):
        """
        GOAL: Reset the environment, with the execution of Noop actions.
        
        INPUTS: - kwargs: Parameters for the resetting of the wrapped environment.
        
        OUTPUTS: - state: RL state.
        """

        # Reset of the wrapped environment
        state = self.env.reset(**kwargs)

        # Update of the number of lives for the agent
        self.numberOfLives = self.env.unwrapped.ale.lives()

        return state


    def step(self, action):
        """
        GOAL: Execute a specified action in the wrapped environment.
        
        INPUTS: - action: Action to be executed.
        
        OUTPUTS: - state: RL state.
                 - reward: RL reward.
                 - done: RL episode termination signal.
                 - info: Additional information (optional).
        """

        # Step for the wrapped environment
        state, reward, done, info = self.env.step(action)

        # Check for the loss of life
        lives = self.env.unwrapped.ale.lives()
        lossOfLife = False
        if lives < self.numberOfLives:
            lossOfLife = True
        
        # Update of the number of lives for the agent
        self.numberOfLives = lives

        # Include the "loss of life" information into the info variable
        info['lossOfLife'] = lossOfLife

        return state, reward, done, info



###############################################################################
######################### Class ClipRewardWrapper #############################
###############################################################################

class ClipRewardWrapper(gym.RewardWrapper):
    """
    GOAL: Wrapper for the clipping of the RL reward to {+1, 0, -1}.
    
    VARIABLES: /
                         
    METHODS: - __init__: Initialization of the wrapped environment.
             - reward: Assign the appropriate RL reward.
    """

    def __init__(self, env):
        """
        GOAL: Initialization of the wrapped environment.
        
        INPUTS: - env: Environment to wrap.
        
        OUTPUTS: /
        """

        gym.RewardWrapper.__init__(self, env)


    def reward(self, reward):
        """
        GOAL: Process the RL reward.
        
        INPUTS: - reward: RL reward to be processed.
        
        OUTPUTS: - reward: RL reward processed.
        """
        
        return np.sign(reward)



###############################################################################
############################# Class FrameWrapper ##############################
###############################################################################

class FrameWrapper(gym.ObservationWrapper):
    """
    GOAL: Wrapper for changing the format of a frame to 84x84.
    
    VARIABLES: /
                         
    METHODS: - __init__: Initialization of the wrapped environment.
             - observation: RL state (observation).
    """

    def __init__(self, env):
        """
        GOAL: Initialization of the wrapped environment.
        
        INPUTS: - env: Environment to wrap.
        
        OUTPUTS: /
        """

        gym.ObservationWrapper.__init__(self, env)

        # Initialization of the new observation space (state space)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=env.observation_space.dtype)


    def observation(self, frame):
        """
        GOAL: Process the frame.
        
        INPUTS: - frame: Frame to be processed.
        
        OUTPUTS: - state: RL state after processing.
        """

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]



###############################################################################
######################### Class NormalizationWrapper ##########################
###############################################################################

class NormalizationWrapper(gym.ObservationWrapper):
    """
    GOAL: Wrapper for normalizing the state (pixel range [0, 255] -> [0.0, 1.0]).
    
    VARIABLES: /
                         
    METHODS: - __init__: Initialization of the wrapped environment.
             - observation: RL state (observation).
    """

    def __init__(self, env):
        """
        GOAL: Initialization of the wrapped environment.
        
        INPUTS: - env: Environment to wrap.
        
        OUTPUTS: /
        """

        gym.ObservationWrapper.__init__(self, env)

        # Initialization of the new observation space (state space)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=env.observation_space.shape, dtype=np.float32)


    def observation(self, frame):
        """
        GOAL: Process the frame.
        
        INPUTS: - frame: Frame to be processed.
        
        OUTPUTS: - state: RL state after processing.
        """

        return np.array(frame).astype(np.float32) / 255.0
        
        

###############################################################################
########################## Class StackingWrapper ##############################
###############################################################################

class StackingWrapper(gym.Wrapper):
    """
    GOAL: Wrapper for stacking the n last frames, considering this set of
          frames as the RL state (observation).
    
    VARIABLES: - numberOfFrames: Number of frames to be stacked.
               - frames: Data structure storing the last n frames.
                         
    METHODS: - __init__: Initialization of the wrapped environment.
             - reset: Reset of the environment, with the execution of "Fire" action.
             - step: Execute a specified action in the wrapped environment.
    """

    def __init__(self, env, numberOfFrames=4):
        """
        GOAL: Initialization of the wrapped environment.
        
        INPUTS: - env: Environment to wrap.
                - numberOfFrames: Number of frames to be stacked.
        
        OUTPUTS: /
        """

        gym.Wrapper.__init__(self, env)

        # Initialization of important variables
        self.numberOfFrames = numberOfFrames
        self.frames = deque([], maxlen=numberOfFrames)

        # Setting of the new observation space (state space)
        space = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(space[0], space[1], space[2]*numberOfFrames), dtype=env.observation_space.dtype)


    def reset(self, **kwargs):
        """
        GOAL: Reset the wrapped environment.
        
        INPUTS: - kwargs: Parameters for the resetting of the wrapped environment.
        
        OUTPUTS: - state: RL state.
        """

        # Resetting of the wrapped environment
        state = self.env.reset(**kwargs)

        # Storing of the first frame n times
        for _ in range(self.numberOfFrames):
            self.frames.append(state)
        
        return LazyFrames(list(self.frames))


    def step(self, action):
        """
        GOAL: Execute a specified action in the wrapped environment.
        
        INPUTS: - action: Action to be executed.
        
        OUTPUTS: - state: RL state.
                 - reward: RL reward.
                 - done: RL episode termination signal.
                 - info: Additional information (optional).
        """

        # Execution of the action specified
        state, reward, done, info = self.env.step(action)

        # Appending of the new frame resulting from this action
        self.frames.append(state)

        return LazyFrames(list(self.frames)), reward, done, info



###############################################################################
############################## Class LazyFrames ###############################
###############################################################################

class LazyFrames():
    """
    GOAL: Data structure obtimizing memory usage by ensuring that common frames
          between observations are only stored once. The following code is
          from the "OpenAI gym baselines" repository.
    """

    def __init__(self, frames):

        self._frames = frames
        self._out = None
    

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]



###############################################################################
############################ Class PytorchWrapper #############################
###############################################################################

class PytorchWrapper(gym.ObservationWrapper):
    """
    GOAL: Wrapper for changing the shape of the RL observations (states) for
          Pytorch.
    
    VARIABLES: /
                         
    METHODS: - __init__: Initialization of the wrapped environment.
             - observation: RL state (observation).
    """

    def __init__(self, env):
        """
        GOAL: Initialization of the wrapped environment.
        
        INPUTS: - env: Environment to wrap.
        
        OUTPUTS: /
        """

        gym.ObservationWrapper.__init__(self, env)

        # Initialization of the new observation space (state space)
        space = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(space[-1], space[0], space[1]), dtype=np.float32)


    def observation(self, observation):
        """
        GOAL: Process the observation.
        
        INPUTS: - observation: Observation to be processed.
        
        OUTPUTS: - state: RL state after processing.
        """

        return np.moveaxis(observation, 2, 0)



###############################################################################
############################# Class AtariWrapper ##############################
###############################################################################

class AtariWrapper():
    """
    GOAL: Custom wrapper for the OpenAI gym {}NoFramekip-v4 environments
          (Atari games).
    
    VARIABLES: /
                         
    METHODS: - wrapper: Wrap the OpenAI gym {}NoFramekip-v4 environment.
    """
    
    def wrapper(self, environment, maxNoop=30, skip=4, numberOfFramesStacked=4, stickyActionsProba=0):
        """
        GOAL: Wrapping of the OpenAI gym {}NoFramekip-v4 environment.
        
        INPUTS: - environment: Name of the environment ({}NoFramekip-v4).
                - maxNoop: Maximum number of Noop actions to execute at reset.
                - skip: Number of frames to skip.
                - numberOfFramesStacked: Number of frames to be stacked.
                - stickyActionsProba: Probability associated with sticky actions.
        
        OUTPUTS: env: Wrapped environment.
        """

        # Creation of the OpenAI gym environment.
        env = gym.make(environment)

        # Application of the necessary wrappers.
        env = NoopWrapper(env, maxNoop=maxNoop)
        env = SkipWrapper(env, skip=skip, stickyActionsProba=stickyActionsProba)
        env = LossOfLifeWrapper(env)
        env = FireResetWrapper(env)
        env = ClipRewardWrapper(env)
        env = FrameWrapper(env)
        env = NormalizationWrapper(env)
        env = StackingWrapper(env, numberOfFrames=numberOfFramesStacked)
        env = PytorchWrapper(env)
        return env
