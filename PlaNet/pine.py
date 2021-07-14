import models

class Pine():

    def __init__(self, config, pine_model=None, saved_dict=None):
        self.pine_model = models.PineModel(config, saved_dict) if pine_model is None else pine_model


    def new_environment(self, env, desc):
        initial_observation = env.reset()




    def solve(self):


    def train(self):

