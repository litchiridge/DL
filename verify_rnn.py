import torch
import torch.nn as nn
import torch.nn.functional as F


def equal(torch_a, torch_b):
    if torch_a.shape != torch_b.shape:
        return False
    return ((torch_a-torch_b).abs() < 1e-6).all().item()

step = 3
input_size = 100
hidden_size = 5
num_layers = 3
bidirectional = True
nonlinearity = 'tanh'

# initalize a test data x
x = torch.normal(0,1,(step,input_size))
# initialize a rnn
rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, nonlinearity=nonlinearity, bias=True, bidirectional=bidirectional)
# get the parameters of the rnn, and then manully write the forward pass
# in each layer and each direction, there are 4 parameter matrixes, w_xh, w_hh, b_xh, b_hh since the bias is True
num_layer_params = (2 if bidirectional else 1) * 4
# get the parameters of rnn
param_list = list(rnn.parameters())

# store the hidden_state in each layer, and the is updated in each layer
layer_hidden_state = torch.zeros((step, hidden_size*2 if bidirectional else hidden_size)) 
last_step_state = [] # the last step of the hidden state 

for j in range(num_layers):
    layer_params = param_list[j*num_layer_params:(j+1)*num_layer_params]
    w_ih, w_hh, b_ih, b_hh = layer_params[:4]
    if bidirectional: 
        w_ih_r, w_hh_r, b_ih_r, b_hh_r = layer_params[4:]
    # initalize the hidden_state to 0
    hidden_state = torch.zeros(1, hidden_size)
    hidden_state_r = torch.zeros(1, hidden_size) # for bidirectional rnn
    hs_list, hs_r_list = [], []
    # in the first layer, the input is x, else the input is set to the hidden_state of previous layer
    input = x if j==0 else layer_hidden_state 
    for i in range(step):
        hidden_state = F.tanh(input[i:i+1]@w_ih.T + b_ih+ hidden_state @ w_hh.T + b_hh)
        hs_list.append(hidden_state)
        if bidirectional:
            hidden_state_r=F.tanh(input[step-i-1:step-i]@w_ih_r.T + b_ih_r+ hidden_state_r @ w_hh_r.T + b_hh_r)
            hs_r_list.append(hidden_state_r)
    layer_hidden_state = torch.concat(hs_list)
    last_step_state.append(hs_list[-1])
    if bidirectional:
        layer_hidden_state = torch.concat([layer_hidden_state, torch.concat(hs_r_list[::-1])], dim=1)
        last_step_state.append(hs_r_list[-1])
last_step_state = torch.concat(last_step_state)
h,c = rnn.forward(x)
print( equal(h, layer_hidden_state) )
print( equal(c, last_step_state))