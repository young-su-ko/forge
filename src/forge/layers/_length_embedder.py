from forge.layers._time_embedder import TimestepEmbedder


class LengthEmbedder(TimestepEmbedder):
    def __init__(self, hidden_size):
        super().__init__(hidden_size)

    # def forward(self, length, mean, std):
    #     normalized_length = (length - mean) / std
    #     length_freq = self.timestep_embedding(normalized_length, self.frequency_embedding_size)
    #     return self.mlp(length_freq)

    def forward(self, length):
        length_freq = self.timestep_embedding(length, self.frequency_embedding_size)
        return self.mlp(length_freq)
