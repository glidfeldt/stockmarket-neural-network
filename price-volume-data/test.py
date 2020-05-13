## Import the Adjoint Method (ODE Solver)
from torchdiffeq import odeint_adjoint as odeint


## Normal Residual Block Example

class ResBlock(nn.Module):

    # init a block - Convolve, pool, activate, repeat
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    # Forward pass - pass output of one layer to the input of the next
    def forward(self, x):
        shortcut = x
        out = self.relu(self.norm1(x))
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut


## Ordinary Differential Equation Definition

class ODEfunc(nn.Module):

    # init ODE variables
    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(dim, dim)
        self.norm2 = norm(dim)
        self.conv2 = conv3x3(dim, dim)
        self.norm3 = norm(dim)
        self.nfe = 0

    # init ODE operations
    def forward(self, t, x):
        # nfe = number of function evaluations per timestep
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm3(out)
        return out


## ODE block
class ODEBlock(nn.Module):

    # initialized as an ODE Function
    # count the time
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    # foorward pass
    # input the ODE function and input data into the ODE Solver (adjoint method)
    # to compute a forward pass
    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=args.tol, atol=args.tol)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


## Main Method

if __name__ == '__main__':

    # Add Pooling
    downsampling_layers = [
        nn.Conv2d(1, 64, 3, 1),
        ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
        ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
    ]

    # Initialize the network as 1 ODE Block
    feature_layers = [ODEBlock(ODEfunc(64))]
    # Fully connected Layer at the end
    fc_layers = [norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(64, 10)]

    # The Model consists of an ODE Block, pooling, and a fully connected block at the end
    model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers).to(device)

    # Declare Gradient Descent Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # Training Loop
    for itr in range(args.nepochs * batches_per_epoch):
        # init the optimizer
        optimizer.zero_grad()

        # Generate training data
        x, y = data_gen()
        # Input Training data to model, get Prediction
        logits = model(x)
        # Compute Error using Prediction vs Actual Label
        loss = CrossEntropyLoss(logits, y)

        # Backpropagate
        loss.backward()
        optimizer.step()