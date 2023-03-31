import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


class BetterAutoEncoder(nn.Module):
    def __init__(self, num_question, meta_features, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(BetterAutoEncoder, self).__init__()

        # Define linear functions.
        self.question_encoder = nn.Linear(num_question, k)
        self.meta_data_encoder = nn.Linear(meta_features, k)
        self.question_encoder_hidden = nn.Linear(2*k, k)
        self.question_decoder_hidden = nn.Linear(k, k)
        self.question_decoder = nn.Linear(k, num_question)
        
        self.acts1 = nn.PReLU()
        self.acts2 = nn.PReLU()
        self.acts3 = nn.PReLU()
        self.acts4 = nn.PReLU()
        self.acts5 = nn.PReLU()

    def forward(self, inputs, user_data):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :param user_data: user data vector
        :return: user vector.
        """
        encoded_questions = self.question_encoder(inputs)
        encoded_questions = self.acts1(encoded_questions)

        encoded_meta = self.meta_data_encoder(user_data)
        encoded_meta = self.acts2(encoded_meta)

        concatenated_vector = torch.concatenate([encoded_questions, encoded_meta], dim=-1)
        latent_embedding = self.question_encoder_hidden(concatenated_vector)
        latent_embedding = self.acts3(latent_embedding)

        out = self.question_decoder_hidden(latent_embedding)
        out = self.acts4(out)

        out = self.question_decoder(out)
        out = self.acts5(out)

        return out