
import torch.nn as nn
from transformer_improved import TransformerEncoderLayer

class SingleTransformer(nn.Module):

    def __init__(self, input_size):
        super(SingleTransformer, self).__init__()

        self.transformer = TransformerEncoderLayer(d_model=input_size, nhead=4, dim_feedforward=256, dropout=0)

    def forward(self, input):
        # input shape: batch, seq, dim
        output = input
        output = self.transformer(output.permute(1, 0, 2).contiguous()).permute(1, 0, 2).contiguous()
        return output
