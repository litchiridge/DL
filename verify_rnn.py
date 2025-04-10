import torch
import torch.nn as nn

step = 4
x = torch.normal(0,1,(step,100))
rnn = nn.RNN(input_size=100, hidden_size=5, num_layers=2, nonlinearity='tanh', bias=True, bidirectional=True)
h,c = rnn.forward(x)
print(h.shape)
print(c.shape)

list(rnn.state_dict().keys())
w_xh_l0, w_hh_l0, b_xh_l0, b_hh_l0, w_xh_l0_reverse, w_hh_l0_reverse, b_xh_l0_reverse, b_hh_l0_reverse, w_xh_l1, w_hh_l1, b_xh_l1, b_hh_l1, w_xh_l1_reverse, w_hh_l1_reverse, b_xh_l1_reverse, b_hh_l1_reverse = rnn.parameters()

hidden_state_l0 = torch.zeros(1, 5)
hidden_state_l0_reverse = torch.zeros(1, 5)
hidden_state_l1 = torch.zeros(1, 5)
hidden_state_l1_reverse = torch.zeros(1, 5)

h0_list, h0_r_list = [], []
h1_list, h1_r_list = [], []
for i in range(step):
    hidden_state_l0 = F.tanh(x[i:i+1]@w_xh_l0.T + b_xh_l0+ hidden_state_l0 @ w_hh_l0.T + b_hh_l0)
    hidden_state_l0_reverse = F.tanh(x[step-i-1:step-i]@w_xh_l0_reverse.T + b_xh_l0_reverse+ hidden_state_l0_reverse @ w_hh_l0_reverse.T + b_hh_l0_reverse)
    h0_list.append(hidden_state_l0)
    h0_r_list.append(hidden_state_l0_reverse)
h0_state = torch.concat([torch.concat(h0_list), torch.concat(h0_r_list[::-1])], dim=1)
for i in range(step):
    hidden_state_l1 = F.tanh(h0_state[i:i+1]@w_xh_l1.T + b_xh_l1 + hidden_state_l1@w_hh_l1.T + b_hh_l1)
    hidden_state_l1_reverse = F.tanh( h0_state[step-i-1:step-i]@w_xh_l1_reverse.T + b_xh_l1_reverse + hidden_state_l1_reverse@w_hh_l1_reverse.T + b_hh_l1_reverse)
    h1_list.append(hidden_state_l1)
    h1_r_list.append(hidden_state_l1_reverse)
h1_state = torch.concat([torch.concat(h1_list), torch.concat(h1_r_list[::-1])], dim=1)
