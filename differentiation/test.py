import torch
import logging
import differentiate as diff

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s %(message)s')

class Squarer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        act = x**2
        return act

class Siner(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)

class Combiner(torch.nn.Module):
    def __init__(self, a, b, c):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c

    def forward(self, x):
        output = self.a * x[:, 0]**2 + self.b * x[:, 0] + self.c * x[:, 1]
        output = output.unsqueeze(1)
        return output

def main():
    logging.info(f"test.main()")

    x = torch.tensor([[1.0], [2.0], [3.0]])
    net = Squarer()
    u = net(x)
    logging.info(f"u = {u}")

    du_dx = diff.first_derivative(net, x)
    logging.info(f"du_dx = {du_dx}")

    d2u_dx2 = diff.second_derivative(net, x, 0)
    logging.info(f"d2u_dx2 = {d2u_dx2}")

    siner = Siner()
    u = siner(x)
    logging.info(f"Siner:\n\tu = {u}")

    du_dx = diff.first_derivative(siner, x)
    logging.info(f"du_dx = {du_dx}")

    d2u_dx2 = diff.second_derivative(siner, x, 0)
    logging.info(f"d2u_dx2 = {d2u_dx2}")

    combiner = Combiner(1, 2, 3)
    x = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    u = combiner(x)
    logging.info(f"Combiner\n\tu = {u}")
    du_dx = diff.first_derivative(combiner, x)
    logging.info(f"du_dx = {du_dx}")

    d2u_dx02__d2u_dx0dx1 = diff.second_derivative(combiner, x, 0)
    logging.info(f"d2u_dx02__d2u_dx0dx1 = {d2u_dx02__d2u_dx0dx1}")

    d2u_dx1dx0__d2u_dx12 = diff.second_derivative(combiner, x, 1)
    logging.info(f"d2u_dx1dx0__d2u_dx12 = {d2u_dx1dx0__d2u_dx12}")


if __name__ == '__main__':
    main()