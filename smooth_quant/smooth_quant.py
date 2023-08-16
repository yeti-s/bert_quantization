import torch
from torch import nn
import functools
from functools import partial


def quantize_per_tensor_asymmetric(x, n_bits = 8):
    x_max = x.max()
    x_min = x.min()
    q_max = 2**(n_bits-1)-1
    q_min = -(q_max + 1)
    scale = (x_max - x_min) / (2 ** n_bits - 1)
    zero_point = torch.round((x_max * q_min - x_min * q_max) / (x_max - x_min))
    q_x = torch.round(x/scale) + zero_point
    return q_x, scale, zero_point

def quantize_per_tensor_symmetric(x, n_bits = 8):
    scale = x.view(-1, x.shape[-1]).abs().max()
    q_max = 2**(n_bits - 1) - 1
    scale = scale / q_max
    x = torch.round(x / scale)
    return x, scale, torch.tensor(0)

class FakeQuantLinear(nn.Module):
    def __init__(self, linear, quantization_method = quantize_per_tensor_symmetric, n_bits=8):
        super(FakeQuantLinear, self).__init__()

        self.quantize = partial(quantization_method, n_bits=n_bits)
        q_w, s_w, z_w = self.quantize(linear.weight.detach())

        self.register_buffer("q_w", q_w)
        self.register_buffer("s_w", s_w)
        self.register_buffer("z_w", z_w)
        self.register_buffer("bias", linear.bias.detach())


    def forward(self, x):
        q_x, s_x, z_x = self.quantize(x)
        scale = s_x * self.s_w
        output = torch.functional.F.linear(q_x.sub_(z_x), self.q_w.sub(self.z_w), self.bias.div(scale).round_())
        return output.mul_(scale)
    


class SmoothQuantLinear(nn.Module):
    def __init__(self, linear, act_scale, quantization_method = quantize_per_tensor_symmetric, n_bits=8, alpha = 0.5, observe_func = None):
        super(SmoothQuantLinear, self).__init__()

        self.quantize = partial(quantization_method, n_bits=n_bits)
        self.observe_func = observe_func
        
        weight = linear.weight.detach()
        w_abs_max = weight.abs().max(dim=0)[0].clamp_(min=1e-5)
        scale = act_scale.pow(alpha).div_(w_abs_max.pow(1 - alpha)).clamp_(min=1e-5)
        q_w, s_w, z_w = self.quantize(weight.mul_(scale.view(1, -1)))

        self.register_buffer("scale", scale)
        self.register_buffer("q_w", q_w)
        self.register_buffer("s_w", s_w)
        self.register_buffer("z_w", z_w)
        self.register_buffer("bias", linear.bias.detach())

    def smooth(self, x):
        return x.div(self.scale)
    
    def forward(self, x):
        smooth_x = self.smooth(x)
        if self.observe_func != None:
            self.observe_func(smooth_x)
        q_x, s_x, z_x = self.quantize(smooth_x)
        scale = s_x * self.s_w
        output = torch.functional.F.linear(q_x.sub_(z_x), self.q_w.sub(self.z_w), self.bias.div(scale).round_())
        return output.mul_(scale)



def get_act_scales(model, data_loader):
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}

    # get maximum channel values
    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
            stat_tensor(name, x)
        
    # register hook on every Linear layer
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name))
            )
    
    # get act scales with validation dataset
    for batch in data_loader:
        batch_inputs = tuple(t.to(device) for t in batch)
        inputs = {
            'input_ids': batch_inputs[0],
            'attention_mask': batch_inputs[1],
        }
        model(**inputs)
    
    # remove hooks
    for h in hooks:
        h.remove()
    
    return act_scales