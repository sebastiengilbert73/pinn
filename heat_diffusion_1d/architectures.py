import torch

class ResidualBlock(torch.nn.Module):
    def __init__(self, number_of_inputs, number_of_outputs):
        super().__init__()
        self.linear1 = torch.nn.Linear(number_of_inputs, number_of_outputs)
        self.tanh = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(number_of_outputs, number_of_outputs)
        self.passthrough = torch.nn.Identity()
        if number_of_inputs != number_of_outputs:
            self.passthrough = torch.nn.Sequential(
                torch.nn.Linear(number_of_inputs, number_of_outputs)
            )

    def forward(self, input_tsr):  # input_tsr.shape = (B, Ni)
        act1 = self.linear1(input_tsr)  # (B, No)
        act2 = self.tanh(act1)  # (B, No)
        act3 = self.linear2(act2)  # (B, No)
        act4 = act3 + self.passthrough(input_tsr)  # (B, No)
        return act4


class ResidualNet(torch.nn.Module):
    def __init__(self, number_of_inputs, number_of_blocks, block_width, number_of_outputs):
        super().__init__()
        self.blocks_dict = torch.nn.ModuleDict()
        self.blocks_dict['block1'] = ResidualBlock(number_of_inputs, block_width)
        for block_ndx in range(2, number_of_blocks + 1):
            self.blocks_dict[f'block{block_ndx}'] = ResidualBlock(block_width, block_width)
        self.linear1 = torch.nn.Linear(block_width, number_of_outputs)

    def forward(self, input_tsr):  # input_tsr.shape = (B, Ni)
        act = self.blocks_dict['block1'](input_tsr)  # (B, W)
        for block_ndx in range(2, len(self.blocks_dict) + 1):
            act = self.blocks_dict[f'block{block_ndx}'](act)  # (B, W)
        output_tsr = self.linear1(act)  # (B, No)
        return output_tsr