import torch
import torch.autograd as autograd


class L1_Charbonnier_loss(torch.nn.Module):
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


def gradient_penalty(Discriminator, data, real_data, generated_data):
    eta = torch.FloatTensor(real_data.size(0), 1, 1, 1).uniform_(0, 1).cuda()
    eta = eta.expand(real_data.size(0), real_data.size(1), real_data.size(2), real_data.size(3)).cuda()

    interpolated = eta * real_data + ((1 - eta) * generated_data)
    interpolated.cuda()

    # define it to calculate gradient
    interpolated = torch.autograd.Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = Discriminator(data, interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10

    return gradients_penalty

