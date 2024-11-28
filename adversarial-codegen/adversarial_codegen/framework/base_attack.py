class BaseAttack:
    def __init__(self, params=None):
        self.params = params or {}

    def generate(self, input_text):
        raise NotImplementedError