import torch
import torch.nn as nn
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer


class TransformerRegressor(nn.Module):
    '''
    TransformerRegressor
    @ Description: TransformerRegressor is designed with sequences of encoder unit followed by 
    one decoder unit and fully connected neural network to predict one float value at the end.
    This class was particularly designed to learn transcription acitivity from the transcription factor binding profiles 
    
    @ key parameters
        - input_dim: the dimension of raw input information (251 in my case because I had 251 types of transcription factor)
        - model_dim: the dimension of the input information into the encoder unit. input_dim will be converted into model_dim through NN
            [_NOTE] model_dim should be divisible by num_heads. Default: 64
        - num_heads: the number of attention heads in each transformer unit. Default: 8
        - num_layers: the number of transformer layers running on parallel. Default: 3
        - ff_dim: the dimension of the last feed forward fully connected neural network. Default: 256

    @ Note 1: The single encoder unit input is the sequence of EOS
    @ Note 2: relu function was taken for the last ff fully connected neural network, but you can test out different functions
    '''
    def __init__(self, input_dim, model_dim, num_heads, num_layers, ff_dim, output_dim=1):
        super(TransformerRegressor, self).__init__()
        # transforming the initial input
        self.encoder_embedding = nn.Linear(input_dim, model_dim)
        self.decoder_embedding = nn.Linear(input_dim, model_dim)
        
        encoder_layers = TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        
        decoder_layers = TransformerDecoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=ff_dim)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_layers)

        self.fc1 = nn.Linear(model_dim, ff_dim) # The last layer of fully conencted neural network from the encoder unit
        self.fc2 = nn.Linear(ff_dim, output_dim)
        
        self.eos_token = nn.Parameter(torch.randn(1, model_dim))
        
    def forward(self, src):
        src = self.encoder_embedding(src) # shape: (batch_size, seq_length, input_dim)
        src = src.permute(1, 0, 2)  # shape: (seq_length, batch_size, model_dim)
        
        memory = self.transformer_encoder(src)
        batch_size = memory.size(1)

        # Using the EOS token as the input to the decoder
        eos_token_expanded = self.eos_token.expand(1, batch_size, -1)  # Shape: (1, batch_size, model_dim)
        output = self.transformer_decoder(eos_token_expanded, memory)
        output = output.squeeze(0)  # Shape: (batch_size, model_dim)
        output = torch.relu(self.fc1(output))
        output = self.fc2(output)
        
        return output