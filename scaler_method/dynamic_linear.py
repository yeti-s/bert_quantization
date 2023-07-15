import torch
import torch.nn as nn

class Int8Linear(nn.Module):
    def __init__(self, linear, name):
        super(Int8Linear, self).__init__()

        self.name = name

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bit_width = 8
        
        weight, scale, zero_point = self.quantize(linear.weight.detach())
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point)
        self.register_buffer("weight", weight)
        self.bias = linear.bias

    def quantize(self, x):
        x_max = x.max()
        x_min = x.min()
        x_scale = (x_max - x_min) / (2 ** 8 - 1)
        x_zero_point = torch.round((x_max*-128 - x_min*127)/(x_max - x_min))
        x_q = (torch.round(x/x_scale) + x_zero_point).to(torch.int8)
        return x_q, x_scale, x_zero_point

    def forward(self, x):
        x_q, x_scale, x_zero_point = self.quantize(x)
        output = torch.matmul(x_q.to(torch.int32) - x_zero_point, self.weight.T.to(torch.int32) - self.zero_point)
        output = output.to(torch.float32) * (x_scale * self.scale)
        output = output + self.bias
        return output
