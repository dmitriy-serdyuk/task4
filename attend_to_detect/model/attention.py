import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.functional import tanh, softmax
from torch.nn.init import xavier_normal, constant


class ContentAttention(nn.Module):
    def __init__(self, dim):
        super(ContentAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.mask = None

    def apply_mask(self, mask):
        self.mask = mask

    def forward(self, input, context):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        targetT = self.linear_in(input).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, targetT).squeeze(2)  # batch x sourceL
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = softmax(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        context_combined = torch.cat((weighted_context, input), 1)

        context_output = tanh(self.linear_out(context_combined))

        return context_output, attn


class GaussianAttention(nn.Module):
    def __init__(self, dim, monotonic=False, bias=True, normalize=False):
        super(GaussianAttention, self).__init__()
        self.linear_in = nn.Linear(dim, 2, bias=bias)
        self.mask = None
        self.monotonic = monotonic
        self.normalize = normalize

        self.initialize()

    def initialize(self):
        xavier_normal(self.linear_in.weight.data)
        constant(self.linear_in.bias.data, 0)

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def apply_mask(self, mask):
        self.mask = mask

    def get_initial_kappa(self, context):
        batch_size = context.size(0)
        kappa = Variable(torch.zeros(batch_size))
        if torch.has_cudnn:
            kappa = kappa.cuda()
        return kappa

    def get_indexes(self, context):
        indexes = Variable(torch.arange(0, context.size(1)) / context.size(1),
                           requires_grad=False).unsqueeze(0)
        if self.is_cuda:
            indexes.cuda()
        return indexes

    def forward(self, state, context, kappa_prev):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        beta_kappa = torch.exp(self.linear_in(state))
        beta = beta_kappa[:, 0]
        if self.monotonic:
            kappa = kappa_prev + beta_kappa[:, 1]
        else:
            kappa = beta_kappa[:, 1]

        # Get attention
        if self.mask is not None:
            # TODO: do we need mask?
            pass
        indexes = self.get_indexes(context)
        indexes_exp = indexes.expand(kappa.size(0), indexes.size(1))
        if torch.has_cudnn:
            indexes_exp = indexes_exp.cuda()
        beta_exp = beta.contiguous().view(beta.size(0), 1).expand_as(indexes_exp)
        kappa_exp = kappa.contiguous().view(kappa.size(0), 1).expand_as(indexes_exp)
        attn = torch.exp(-beta_exp * (kappa_exp - indexes_exp)**2)
        if self.normalize:
            attn = attn / attn.sum(1).expand_as(attn)
        attn3 = attn.unsqueeze(2)

        weighted_context = attn3.expand_as(context) * context
        context_combined = weighted_context.sum(dim=1).squeeze(1)

        return context_combined, attn, kappa
