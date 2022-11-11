from collections import OrderedDict

import torch
from torch import nn

import numpy as np

"""
The code in this file has been mainly derived from the paper on SIRENs.
The source is: https://github.com/vsitzmann/siren
"""


class SineLayer(nn.Module):
    # Section references below are section of the original SIREN paper

    # See paper sec. 3.2, final paragraph,
    # and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies
    # the activations before the nonlinearity. 
    # Different signals may require different omega_0 in the first layer -
    # this is a hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0
    # so as to keep the magnitude of activations constant, but boost
    # gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input, gamma, beta, i):
        return torch.sin(gamma[i] * self.omega_0 * self.linear(input) + beta[i])

    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    
    
class SirenDecoder(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers,
                 out_features, outermost_linear=False, first_omega_0=30,
                 hidden_omega_0=30.):
        super().__init__()
        self.in_features = in_features
        self.outermost_linear = outermost_linear
        self.net = []

        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers - 1):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.ModuleList(self.net)

    def forward(self, x, gamma, beta, mode):
        out = []
        if mode == 'film':
            for xi, gamma, beta in zip(x, gamma, beta):
                for i, layer in enumerate(self.net[:-1]):
                    xi = layer(xi, gamma, beta, i)
                out.append(xi)
            x = torch.cat(out)
        else:
            for layer in self.net[:-1]:
                x = layer(x, gamma, beta, 0)

        if self.outermost_linear:
            x = self.net[-1](x)
        else:
            x = self.net[-1](x, [1.], [0.], 0)

        return x

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()
 
        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else: 
                x = layer(x)

                if retain_grad:
                    x.retain_grad()
                    
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations
