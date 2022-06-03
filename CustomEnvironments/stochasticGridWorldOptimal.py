# coding=utf-8

###############################################################################
##################### Class StochasticGridWorldOptimal ########################
###############################################################################

class StochasticGridWorldOptimal():
    """
    GOAL: Implementing the optimal policy associated with the stochastic grid
          world environment.
    
    VARIABLES: - environment: Stochastic grid world environment.
                                
    METHODS: - __init__: Initialization of the class.
    """

    def __init__(self, environment):
        """
        GOAL: Perform the initialization of the class.
        
        INPUTS: - environment: Stochastic grid world environment considered.
        
        OUTPUTS: - processState: Preprocessing of the RL state.
                 - chooseAction: Choose the optimal RL action.
        """

        # Initialization of important variables
        self.environment = environment
        self.size = self.environment.size
        self.trapPosition = self.environment.trapPosition
        self.targetPosition = self.environment.targetPosition

    
    def processState(self, state):
        """
        GOAL: Potentially process the RL state returned by the environment.
        
        INPUTS: - state: RL state returned by the environment.
        
        OUTPUTS: - state: RL state processed.
        """

        return state

    
    def chooseAction(self, state, plot=False):
        """
        GOAL: Choose the optimal RL action.
        
        INPUTS: - state: RL state returned by the environment.
                - plot: False, because not supported.
        
        OUTPUTS: - action: RL action selected.
        """

        # Retrieve the coordinates of the agent
        x = state[0]
        y = state[1]

        # Implementation of the optimal policy
        if x == self.targetPosition[0] and y < self.trapPosition[1]:
            action = 0
        elif x == self.targetPosition[0] and y > self.trapPosition[1]:
            action = 3
        elif y == self.targetPosition[1] and x < self.targetPosition[0]:
            action = 0
        elif y == self.targetPosition[1] and x > self.targetPosition[0]:
            action = 2
        elif (x < self.targetPosition[0] or x > self.targetPosition[0]) and y < (self.targetPosition[1]-1):
            action = 3
        elif y == (self.targetPosition[1]-1) and y > self.trapPosition[1] and x < self.targetPosition[0]:
            action = 0
        elif y == (self.targetPosition[1]-1) and y > self.trapPosition[1] and x > self.targetPosition[0]:
            action = 2
        else:
            action = 3

        # Return of the RL action selected
        return action
