class BaseAgent:
    def __init__(self, name):
        self.name = name

    async def act(self, observation):
        raise NotImplementedError("Subclasses must implement this method")

    async def think(self, data):
        raise NotImplementedError("Subclasses must implement this method")