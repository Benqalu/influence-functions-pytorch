import torch
import numpy as np
import pandas as pd

# A sample torch model with layer attribute
# inps: Number of input features
# hiddens: Number of neurons on each layer, 
#     e.g., [] means no hidden layer, 
#     [128] means one hidden layer with 128 neurons
# bias: Decide if there is a bias on each layer, must be true in the example
# seed: Reproductivity, None means random seed, otherwise specifiy a integer
# hidden_activation: Activation function after each hidden layer

class TorchNNCore(torch.nn.Module):
    def __init__(
        self, inps, hiddens=[], bias=True, seed=None, hidden_activation=torch.nn.ReLU
    ):
        super(TorchNNCore, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        struct = [inps] + hiddens + [1]
        self.layers = [] # This layer attribute is required under 
        for i in range(1, len(struct)):
            self.layers.append(
                torch.nn.Linear(
                    in_features=struct[i - 1], out_features=struct[i], bias=bias
                )
            )
            if i == len(struct) - 1:
                self.layers.append(torch.nn.Sigmoid())
            else:
                self.layers.append(hidden_activation())
        self.model = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        output = self.model(x)
        return output

# Prepare training & testing dataset
data = pd.read_csv('./adult.csv').to_numpy()
X_train = torch.tensor(data[:,:-1], dtype=torch.float)
y_train = torch.tensor(data[:,-1].reshape(-1,1), dtype=torch.float)
print(X_train.shape, y_train.shape)

# Specify loss function, define model and optimizer
loss_func = torch.nn.BCELoss()
model = TorchNNCore(inps=X_train.shape[1], hiddens=[128], hidden_activation=torch.nn.LeakyReLU)
optim = torch.optim.Adam(model.parameters(),lr=0.001)

y_train_np = y_train.detach().numpy()
for epoch in range(0,300):
    optim.zero_grad()
    y_pred = model(X_train)
    loss = loss_func(y_pred, y_train)
    loss.backward()
    optim.step()
    if epoch%10==0:
        y_pred_np = (y_pred.detach().numpy()) > 0.5
        accuracy = sum(y_pred_np == y_train_np)/y_train_np.shape[0]
        print('Epoch = %d, loss = %.4f, accuracy=%.4f'%(epoch, loss.tolist(), accuracy))
optim.zero_grad()

from InfluenceFunction import InfluenceFunction

infl = InfluenceFunction(
    model = model, 
    X_train = X_train, 
    y_train = y_train, 
    loss_func = loss_func, # In this example, it's BCELoss
    layer_index = -2, # In this example, as shown in the model structure, we use the second last layer 
)

for i in range(0,100):
    print(i, y_train[i][0].tolist(), '%.4f'%y_pred[i][0].tolist(), '%.4f'%infl.influence_remove_single(i))