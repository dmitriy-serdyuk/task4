# -*- coding: utf-8 -*-

# imports
from operator import mul
from functools import reduce
import torch
from torch.autograd import Variable
from torch.nn.init import xavier_normal, constant

from attention import GaussianAttention

__author__ = 'Konstantinos Drossos - TUT'
__docformat__ = 'reStructuredText'


def make_me_a_list(with_that, and_this_long=1):
    if type(with_that) is not list:
        with_that = [with_that]

    if len(with_that) == 1:
        with_that = with_that * and_this_long

    return with_that


class CategoryBranch2(torch.nn.Module):

    def __init__(self,
                 cnn_channels_in, cnn_channels_out,
                 cnn_kernel_sizes, cnn_strides, cnn_paddings, cnn_activations,
                 max_pool_kernels, max_pool_strides, max_pool_paddings,
                 rnn_input_size, rnn_out_dims, rnn_activations,
                 dropout_cnn, dropout_rnn_input, dropout_rnn_recurrent,
                 rnn_subsamplings, decoder_dim, output_classes,
                 monotonic_attention=False, attention_bias=True):

        super(CategoryBranch2, self).__init__()

        if type(cnn_channels_out) is not list:
            raise AttributeError('Channels out must be a list of len '
                                 'equal to the desired amount of conv '
                                 'layers')

        self.cnn_channels_out = [cnn_channels_in] + cnn_channels_out

        cnn_len = len(self.cnn_channels_out) - 1

        self.cnn_kernel_sizes = make_me_a_list(cnn_kernel_sizes, cnn_len)
        self.cnn_strides = make_me_a_list(cnn_strides, cnn_len)
        self.cnn_paddings = make_me_a_list(cnn_paddings, cnn_len)
        self.cnn_activations = make_me_a_list(cnn_activations, cnn_len)

        self.max_pool_kernels = make_me_a_list(max_pool_kernels, cnn_len)
        self.max_pool_strides = make_me_a_list(max_pool_strides, cnn_len)
        self.max_pool_paddings = make_me_a_list(max_pool_paddings, cnn_len)

        self.rnn_out_dims = [rnn_input_size] + rnn_out_dims
        rnn_len = len(self.rnn_out_dims) - 1
        self.rnn_activations_f = make_me_a_list(rnn_activations, rnn_len)
        self.rnn_activations_b = make_me_a_list(rnn_activations, rnn_len)

        self.rnn_subsamplings = make_me_a_list(rnn_subsamplings, rnn_len - 1) + [1]

        self.dropout_cnn = dropout_cnn
        self.dropout_rnn_input_f = dropout_rnn_input
        self.dropout_rnn_input_b = dropout_rnn_input
        self.dropout_rnn_recurrent_f = dropout_rnn_recurrent
        self.dropout_rnn_recurrent_b = dropout_rnn_recurrent

        self.decoder_dim = decoder_dim
        self.output_classes = output_classes

        all_cnn_have_proper_len = all([
            len(a) == cnn_len
            for a in [
                self.cnn_kernel_sizes, self.cnn_strides,
                self.cnn_paddings, self.cnn_activations
            ]
        ])

        if not all_cnn_have_proper_len:
            raise AttributeError('Either provide arguments for all layers or '
                                 'just one (except of channels out)')

        if not len(self.rnn_activations_f) == len(self.rnn_out_dims) - 1:
            raise AttributeError('Amount of rnn activations should be equal to '
                                 'the specified output dims of rnns')

        self.cnn_layers = []
        self.cnn_dropout_layers = []
        self.bn_layers = []
        self.pooling_layers = []
        self.rnn_layers_f = []
        self.rnn_layers_b = []
        self.rnn_dropout_layers_input_f = []
        self.rnn_dropout_layers_recurrent_f = []
        self.rnn_dropout_layers_input_b = []
        self.rnn_dropout_layers_recurrent_b = []

        for i in range(len(self.cnn_channels_out) - 1):
            self.cnn_layers.append(torch.nn.Conv2d(
                in_channels=self.cnn_channels_out[i],
                out_channels=self.cnn_channels_out[i + 1],
                kernel_size=cnn_kernel_sizes[i],
                stride=self.cnn_strides[i],
                padding=self.cnn_paddings[i]
            ))
            self.bn_layers.append(torch.nn.BatchNorm2d(
                num_features=self.cnn_channels_out[i + 1]
            ))
            self.pooling_layers.append(torch.nn.MaxPool2d(
                kernel_size=self.max_pool_kernels[i],
                stride=self.max_pool_strides[i],
                padding=self.max_pool_paddings[i]
            ))
            self.cnn_dropout_layers.append(
                torch.nn.Dropout2d(self.dropout_cnn)
            )

            setattr(self, 'cnn_layer_{}'.format(i+1), self.cnn_layers[-1])
            setattr(self, 'bn_layer_{}'.format(i+1), self.bn_layers[-1])
            setattr(self, 'pool_layer_{}'.format(i+1), self.pooling_layers[-1])
            setattr(self, 'cnn_dropout_layer_{}'.format(i+1), self.cnn_dropout_layers[-1])
            setattr(self, 'cnn_activation_{}'.format(i + 1), self.cnn_activations[i])

        for i in range(len(self.rnn_out_dims) - 1):
            if i > 0:
                # output of last layer was bidirectional, input is bigger
                input_size = 2 * self.rnn_out_dims[i]
            else:
                input_size = self.rnn_out_dims[i]
            self.rnn_layers_f.append(torch.nn.GRUCell(
                input_size=input_size,
                hidden_size=self.rnn_out_dims[i + 1]
            ))
            self.rnn_layers_b.append(torch.nn.GRUCell(
                input_size=input_size,
                hidden_size=self.rnn_out_dims[i + 1]
            ))
            self.rnn_dropout_layers_input_f.append(
                torch.nn.Dropout(self.dropout_rnn_input_f))
            self.rnn_dropout_layers_recurrent_f.append(
                torch.nn.Dropout(self.dropout_rnn_recurrent_f))

            self.rnn_dropout_layers_input_b.append(
                torch.nn.Dropout(self.dropout_rnn_input_b))
            self.rnn_dropout_layers_recurrent_b.append(
                torch.nn.Dropout(self.dropout_rnn_recurrent_b))

            setattr(self, 'rnn_layer_f_{}'.format(i+1), self.rnn_layers_f[-1])
            setattr(self, 'rnn_layer_b_{}'.format(i+1), self.rnn_layers_b[-1])

            setattr(self, 'rnn_dropout_input_f_layer_{}'.format(i + 1),
                    self.rnn_dropout_layers_input_f[-1])
            setattr(self, 'rnn_dropout_recurrent_f_layer_{}'.format(i + 1),
                    self.rnn_dropout_layers_recurrent_f[-1])
            setattr(self, 'rnn_activation_f_{}'.format(i + 1),
                    self.rnn_activations_f[i])

            setattr(self, 'rnn_dropout_input_b_layer_{}'.format(i + 1),
                    self.rnn_dropout_layers_input_b[-1])
            setattr(self, 'rnn_dropout_recurrent_b_layer_{}'.format(i + 1),
                    self.rnn_dropout_layers_recurrent_b[-1])
            setattr(self, 'rnn_activation_b_{}'.format(i + 1),
                    self.rnn_activations_b[i])

        # Adding attention layer
        self.attention = GaussianAttention(
            dim=self.decoder_dim,
            monotonic=monotonic_attention, bias=attention_bias)

        self.decoder_cell = torch.nn.GRUCell(2*self.rnn_out_dims[-1], self.decoder_dim)
        self.output_linear = torch.nn.Linear(self.decoder_dim, self.output_classes)

        self.initialize()

    def init_gru_cell(self, module):
        xavier_normal(module.weight_ih.data)
        xavier_normal(module.weight_hh.data)
        constant(module.bias_ih.data, 0)
        constant(module.bias_hh.data, 0)

    def initialize(self):
        for module in self.cnn_layers:
            xavier_normal(module.weight.data)
            constant(module.bias.data, 0)
        for module in self.rnn_layers_f + self.rnn_layers_b:
            self.init_gru_cell(module)

        self.init_gru_cell(self.decoder_cell)
        xavier_normal(self.output_linear.weight.data)
        constant(self.output_linear.bias.data, 0)

    def get_initial_decoder_state(self, batch_size):
        # TODO: smarter initial state
        state = Variable(torch.zeros((batch_size, self.decoder_dim)))
        if torch.has_cudnn:
            state = state.cuda()
        return state

    def forward(self, x, output_len):

        output = self.pooling_layers[0](self.bn_layers[0](
            self.cnn_activations[0](self.cnn_layers[0](
                self.cnn_dropout_layers[0](x)
            ))
        ))

        for dropout, pooling, bn, activation, cnn in zip(
                self.cnn_dropout_layers[1:],
                self.pooling_layers[1:],
                self.bn_layers[1:],
                self.cnn_activations[1:],
                self.cnn_layers[1:]):
            output = pooling(bn(activation(cnn(dropout(output)))))

        output = output.permute(0, 2, 1, 3)
        o_size = output.size()
        output = output.resize(o_size[0], o_size[1], o_size[2] * o_size[3])
        o_size = output.size()

        for i in range(len(self.rnn_layers_f)):
            h_s_f = Variable(torch.zeros(o_size[0], o_size[1], self.rnn_out_dims[i+1]))
            h_s_b = Variable(torch.zeros(o_size[0], o_size[1], self.rnn_out_dims[i+1]))

            zeros_f = Variable(torch.zeros(o_size[0], self.rnn_out_dims[i+1]))
            zeros_b = Variable(torch.zeros(o_size[0], self.rnn_out_dims[i+1]))

            if torch.has_cudnn:
                h_s_f = h_s_f.cuda()
                h_s_b = h_s_b.cuda()
                zeros_f = zeros_f.cuda()
                zeros_b = zeros_b.cuda()

            h_s_f[:, 0, :] = self.rnn_activations_f[i](self.rnn_layers_f[i](
                self.rnn_dropout_layers_input_f[i](output[:, 0, :]),
                zeros_f))
            h_s_b[:, 0, :] = self.rnn_activations_b[i](self.rnn_layers_b[i](
                self.rnn_dropout_layers_input_b[i](output[:, -1, :]),
                zeros_b))

            for s_i in range(1, o_size[1]):
                h_s_f[:, s_i, :] = self.rnn_activations_f[i](self.rnn_layers_f[i](
                    self.rnn_dropout_layers_input_f[i](output[:, s_i, :]),
                    self.rnn_dropout_layers_recurrent_f[i](h_s_f[:, s_i - 1, :])
                ))
                h_s_b[:, s_i, :] = self.rnn_activations_b[i](self.rnn_layers_b[i](
                    self.rnn_dropout_layers_input_b[i](output[:, -(s_i + 1), :]),
                    self.rnn_dropout_layers_recurrent_b[i](h_s_b[:, s_i - 1, :])
                ))

            output = torch.cat([h_s_f, h_s_b], -1)
            o_size = output.size()
            u_l = o_size[1]
            u_l -= divmod(o_size[1], self.rnn_subsamplings[i])[-1]
            output = output[:, 0:u_l:self.rnn_subsamplings[i], :]
            o_size = output.size()

        hidden = self.get_initial_decoder_state(o_size[0])
        kappa = self.attention.get_initial_kappa(output)

        out_hidden = []
        out_weights = []

        for i in range(output_len):
            attended, weights, kappa = self.attention(hidden, output, kappa)
            hidden = self.decoder_cell(attended, hidden)

            out_hidden.append(self.output_linear(hidden).unsqueeze(1))
            out_weights.append(weights)

        return torch.cat(out_hidden, 1), out_weights


    def nb_trainable_parameters(self):
        # FIXME: add attention parameters
        s_1 = sum([
            reduce(mul, layer.weight.size(), 1) for layer in
            self.bn_layers + self.cnn_layers
        ])
        s_2 = sum([
            reduce(mul, layer.weight_hh.size(), 1) +
            reduce(mul, layer.weight_ih.size(), 1) for layer in
            self.rnn_layers_f + self.rnn_layers_b
        ])
        return s_1 + s_2


def main():
    from torch.autograd import Variable
    from torch.nn import functional
    x = Variable(
        torch.rand(2, 40, 862, 192).float()
    )

    nb_cnn_layers = 3

    b = CategoryBranch2(
        cnn_channels_in=40,
        cnn_channels_out=[40] * nb_cnn_layers,
        cnn_kernel_sizes=[(1, 3)] * nb_cnn_layers,
        cnn_strides=[(1, 2)] * nb_cnn_layers,
        cnn_paddings=[(0, 0)] * nb_cnn_layers,
        cnn_activations=functional.leaky_relu,
        max_pool_kernels=[(3, 2)] * nb_cnn_layers,
        max_pool_strides=[(3, 2)] * nb_cnn_layers,
        max_pool_paddings=[(0, 0)] * nb_cnn_layers,
        rnn_input_size=80,
        rnn_out_dims=[64] * 2,
        rnn_activations=functional.tanh,
        dropout_cnn=0.2,
        dropout_rnn_input=0.2,
        dropout_rnn_recurrent=0.2,
        rnn_subsamplings=3,
        decoder_dim=32,
        output_classes=10
    )

    y, weights = b(x, 3)

    print(y.size())
    print(b.nb_trainable_parameters())


if __name__ == '__main__':
    main()

# EOF
