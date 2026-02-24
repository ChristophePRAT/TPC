class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        self.position = 0

    def add(self, element):
        if len(self.buffer) < self.max_size:
            self.buffer.append(element)
        else:
            self.buffer[self.position] = element
            self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size):
        import random
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

    def __str__(self):
        return f"ReplayBuffer(max_size={self.max_size}, current_size={len(self.buffer)}): {self.buffer}"
