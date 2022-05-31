from rlig import agent, base, buffer

class ImaginedGoals(agent.Agent):

    def __init__(self, q, bvae, buffer):
        super().__init__(locals())

    def get_action(self, state):
        return self.q.select_action(state)

    def store(self, *args):
        self.buffer.store(*args)