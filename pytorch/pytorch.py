from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from torchvision import datasets, transforms

class LinearStochasticFunction(Function):
    BIT_SIZE = 512;

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):

        # Normalize inputs and weights

        #weight += weight.min(dim=1)[0][:, None].abs()
        #weight /= weight.max(dim=1)[0][:, None]
        #weight = weight.clamp(0,1)

        ctx.save_for_backward(input, weight, bias)

        # Test values
        input = torch.tensor([[1.0,0.5,0.6,0.7],[0.1,0.2,0.3,0.8]])
        weight = torch.tensor([[0.5 for _ in range(4)] for _ in range(500)])
        #weight = weight.t()

        input /= input.max(dim=1)[0][:, None]
        #input /= 2.0

        # Ensure the addition of the inputs does not exceed 1
        input /= input.shape[1]

        #print(input.min())
        #print(input.max())


        weight = weight.t()

        ## Convert input and weight to bitstreams
        input_expand = input[:, :, None]
        input_tiled = input_expand.repeat(1,1,LinearStochasticFunction.BIT_SIZE)
        input_bitstream = torch.rand_like(input_tiled) <= (input_tiled) #+1)/2

        weight_expand = weight[:, :, None]
        weight_tiled = weight_expand.repeat(1,1,LinearStochasticFunction.BIT_SIZE)
        weight_bitstream = torch.rand_like(weight_tiled) <= (weight_tiled) #+1)/2

        # Multiplication
        multiplication = (input_bitstream[:, :, None, :] & weight_bitstream)

        # Addition

        # Saturating addition
        final_row = multiplication.sum(dim=1) > 0

        # Non-saturating addition
        #final_row = multiplication[:, 0, :, :]
        #exceeding = torch.Tensor(final_row.shape)

        #for row_i in range(1, multiplication.shape[1]):
        #    row = multiplication[:, row_i, :, :]
        #    select_line = torch.rand_like(final_row) <= 0.5
        #    final_row = (select_line*final_row) + ((1-select_line) * row)


        # Convert bitstream to floating point
        output = torch.mean(final_row.float(), 2)

        #output = input.mm(weight)

        print(output)
        output *= input.shape[1]
        print(output)

        #print(output, output.shape)

        #if bias is not None:
        #    output += bias.unsqueeze(0).expand_as(output)

        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias

class StochasticLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(StochasticLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        # Not a very smart way to initialize weights
        self.weight.data.uniform_(-0.1, 0.1)
        if self.bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return LinearStochasticFunction.apply(input, self.weight, self.bias)

class StochasticNet(nn.Module):
    def __init__(self):
        super(StochasticNet, self).__init__()
        self.d1 = nn.Linear(28*28, 512, False)
        #self.do1 = nn.Dropout(0.2)
        self.d3 = StochasticLinear(512, 10, False)
        #self.do2 = nn.Dropout(0.2)
        #self.d3 = nn.Linear(500, 10, False)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.d1(x))
        #x = self.do1(x)
        #x = F.relu(self.d2(x))
        #x = self.do2(x)
        x = F.relu(self.d3(x))
        return F.log_softmax(x, dim=1)

class NormalNet(nn.Module):
    def __init__(self):
        super(StochasticNet, self).__init__()
        self.d1 = nn.Linear(28*28, 512, False)
        self.do1 = nn.Dropout(0.2)
        self.d2 = nn.Linear(512, 500, False)
        self.do2 = nn.Dropout(0.2)
        self.d3 = nn.Linear(500, 10, False)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.d1(x))
        x = self.do1(x)
        x = F.relu(self.d2(x))
        x = self.do2(x)
        x = F.relu(self.d3(x))
        return F.log_softmax(x, dim=1)
    
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # Ensure weights are positive
        #for p in model.parameters():
        #    p.data.clamp_(0)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    #torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = StochasticNet().to(device)
    #model.load_state_dict(torch.load('mnist_cnn.pt'))

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
        
if __name__ == '__main__':
    main()
