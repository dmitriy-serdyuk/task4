# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.init import xavier_normal, constant, orthogonal
from torch.nn.functional import softmax, relu
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

from .attention import GaussianAttention

__docformat__ = 'reStructuredText'


def make_me_a_list(with_that, and_this_long=1):
    if type(with_that) is not list:
        with_that = [with_that]

    if len(with_that) == 1:
        with_that = with_that * and_this_long

    return with_that


def init_gru_cell(module, init):
    init(module.weight_ih.data)
    init(module.weight_hh.data)
    constant(module.bias_ih.data, 0)
    constant(module.bias_hh.data, 0)


class GRUDecoder(nn.Module):
    def __init__(self, decoder_dim, output_classes, context_dim,
                 monotonic_attention=False,
                 attention_bias=True, init=xavier_normal):
        super(GRUDecoder, self).__init__()
        self.attention = GaussianAttention(
            dim=self.decoder_dim,
            monotonic=monotonic_attention, bias=attention_bias)

        self.decoder_dim = decoder_dim
        self.output_classes = output_classes

        self.decoder_cell = nn.GRUCell(context_dim, self.decoder_dim)
        self.output_linear = nn.Linear(self.decoder_dim, self.output_classes)
        self.context_to_state = nn.Linear(context_dim, self.decoder_dim)

        self.initialize(init)

    def get_initial_decoder_state(self, context_forward, context_backward):
        state = self.input_to_output(
            torch.cat([context_forward[:, -1, :],
                       context_backward[:, 0, :]], -1))
        return state

    def initialize(self, init):
        init_gru_cell(self.decoder_cell, init)
        xavier_normal(self.output_linear.weight.data)
        constant(self.output_linear.bias.data, 0)

        xavier_normal(self.context_to_state.weight.data)
        constant(self.context_to_state.bias.data, 0)

    def forward(self, context_forward, context_backward, output_len):
        hidden = self.get_initial_decoder_state(
            context_forward, context_backward)

        context = torch.cat([context_forward, context_backward], -1)
        kappa = self.attention.get_initial_kappa(context)

        out_hidden = []
        out_weights = []

        for i in range(output_len):
            attended, weights, kappa = self.attention(hidden, context, kappa)
            hidden = self.decoder_cell(attended, hidden)

            out_hidden.append(softmax(self.output_linear(hidden).unsqueeze(1)))
            out_weights.append(weights)
        return torch.cat(out_hidden, 1), out_weights


class MLPDecoder(nn.Module):
    def __init__(self, decoder_dim, output_classes, context_dim,
                 monotonic_attention=False,
                 attention_bias=True, init=xavier_normal):
        super(MLPDecoder, self).__init__()
        self.decoder_dim = decoder_dim
        self.output_classes = output_classes

        self.attention = GaussianAttention(
            dim=self.decoder_dim,
            monotonic=monotonic_attention, bias=attention_bias)

        self.decoder_linear = nn.Linear(context_dim, self.decoder_dim)
        self.output_linear = nn.Linear(self.decoder_dim, self.output_classes)
        self.context_to_state = nn.Linear(context_dim, self.decoder_dim)

        self.initialize(init)

    def get_initial_decoder_state(self, context_forward, context_backward):
        state = self.context_to_state(
            torch.cat([context_forward[:, -1, :],
                       context_backward[:, 0, :]], -1))
        return state

    def initialize(self, init):
        xavier_normal(self.decoder_linear.weight.data)
        constant(self.decoder_linear.bias.data, 0)

        xavier_normal(self.output_linear.weight.data)
        constant(self.output_linear.bias.data, 0)

        xavier_normal(self.context_to_state.weight.data)
        constant(self.context_to_state.bias.data, 0)

    def forward(self, context_forward, context_backward, output_len):
        hidden = self.get_initial_decoder_state(
            context_forward, context_backward)

        context = torch.cat([context_forward, context_backward], -1)
        kappa = self.attention.get_initial_kappa(context)

        attended, weights, kappa = self.attention(hidden, context, kappa)
        hidden = relu(self.decoder_linear(attended))

        output = self.output_linear(hidden).unsqueeze(1)
        return output, weights


class CNNBottom(nn.Module):
    def __init__(self,
                 cnn_channels_in, cnn_channels_out,
                 cnn_kernel_sizes, cnn_strides, cnn_paddings, cnn_activations,
                 max_pool_kernels, max_pool_strides, max_pool_paddings,
                 dropout_cnn, init=xavier_normal):
        super(CNNBottom, self).__init__()

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

        self.dropout_cnn = dropout_cnn

        all_cnn_have_proper_len = all(
            [len(a) == cnn_len
             for a in [
                 self.cnn_kernel_sizes, self.cnn_strides,
                 self.cnn_paddings, self.cnn_activations
             ]])

        if not all_cnn_have_proper_len:
            raise AttributeError('Either provide arguments for all layers or '
                                 'just one (except of channels out)')
        self.cnn_layers = []
        self.cnn_dropout_layers = []
        self.bn_layers = []
        self.pooling_layers = []

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

            self.add_module('cnn_layer_{}'.format(i+1), self.cnn_layers[-1])
            self.add_module('bn_layer_{}'.format(i+1), self.bn_layers[-1])
            self.add_module(
                'pool_layer_{}'.format(i+1), self.pooling_layers[-1])
            self.add_module(
                'cnn_dropout_layer_{}'.format(i+1), self.cnn_dropout_layers[-1])
        self.initialize(init)

    def initialize(self, init):
        for module in self.cnn_layers:
            init(module.weight.data)
            constant(module.bias.data, 0)

    def forward(self, input_):
        output = self.pooling_layers[0](self.bn_layers[0](
            self.cnn_activations[0](self.cnn_layers[0](
                self.cnn_dropout_layers[0](input_)
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
        return output.contiguous()


class CNNRNNFastEncoder(nn.Module):
    def __init__(self,
                 cnn_channels_in, cnn_channels_out,
                 cnn_kernel_sizes, cnn_strides, cnn_paddings, cnn_activations,
                 max_pool_kernels, max_pool_strides, max_pool_paddings,
                 rnn_input_size, rnn_out_dims, rnn_activations,
                 cnn_dropout, rnn_dropout_input, rnn_dropout_recurrent,
                 rnn_subsamplings, init=xavier_normal):
        super(CNNRNNFastEncoder, self).__init__()

        self.cnn_bottom = CNNBottom(cnn_channels_in, cnn_channels_out,
                                    cnn_kernel_sizes, cnn_strides, cnn_paddings,
                                    cnn_activations, max_pool_kernels,
                                    max_pool_strides, max_pool_paddings,
                                    cnn_dropout, init)

        self.rnn_stack = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=rnn_out_dims[0],
            num_layers=len(rnn_out_dims),
            bias=True,
            batch_first=True,
            dropout=rnn_dropout_recurrent,
            bidirectional=True)

    def forward(self, input_, output_length):
        output = self.cnn_bottom(input_)
        batch_size, timesteps, _ = output.size()
        lengths = [timesteps] * batch_size
        output = pack_padded_sequence(output, lengths, batch_first=True)
        packed_output, _ = self.rnn_stack(output)
        return pad_packed_sequence(packed_output, batch_first=True)[0].contiguous()


class CNNRNNEncoder(nn.Module):
    def __init__(self,
                 cnn_channels_in, cnn_channels_out,
                 cnn_kernel_sizes, cnn_strides, cnn_paddings, cnn_activations,
                 max_pool_kernels, max_pool_strides, max_pool_paddings,
                 rnn_input_size, rnn_out_dims, rnn_activations,
                 dropout_cnn, dropout_rnn_input, dropout_rnn_recurrent,
                 rnn_subsamplings, init=xavier_normal):
        super(CNNRNNEncoder, self).__init__()

        self.cnn_bottom = CNNBottom(cnn_channels_in, cnn_channels_out,
                                    cnn_kernel_sizes, cnn_strides, cnn_paddings,
                                    cnn_activations, max_pool_kernels,
                                    max_pool_strides, max_pool_paddings,
                                    dropout_cnn, init)

        self.rnn_out_dims = [rnn_input_size] + rnn_out_dims
        rnn_len = len(self.rnn_out_dims) - 1
        self.rnn_activations_f = make_me_a_list(rnn_activations, rnn_len)
        self.rnn_activations_b = make_me_a_list(rnn_activations, rnn_len)

        self.rnn_subsamplings = make_me_a_list(rnn_subsamplings, rnn_len - 1) + [1]

        self.dropout_rnn_input_f = dropout_rnn_input
        self.dropout_rnn_input_b = dropout_rnn_input
        self.dropout_rnn_recurrent_f = dropout_rnn_recurrent
        self.dropout_rnn_recurrent_b = dropout_rnn_recurrent

        if not len(self.rnn_activations_f) == len(self.rnn_out_dims) - 1:
            raise AttributeError('Amount of rnn activations should be equal to '
                                 'the specified output dims of rnns')

        self.rnn_layers_f = []
        self.rnn_layers_b = []
        self.rnn_dropout_layers_input_f = []
        self.rnn_dropout_layers_recurrent_f = []
        self.rnn_dropout_layers_input_b = []
        self.rnn_dropout_layers_recurrent_b = []

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

        self.initialize(init)

    def initialize(self, init):
        for module in self.rnn_layers_f + self.rnn_layers_b:
            init_gru_cell(module, init)

    @property
    def context_dim(self):
        return self.rnn_out_dims[-1] * 2

    def forward(self, x, output_len):
        output = self.cnn_bottom(x)
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
            h_s_b[:, -1, :] = self.rnn_activations_b[i](self.rnn_layers_b[i](
                self.rnn_dropout_layers_input_b[i](output[:, -1, :]),
                zeros_b))

            for s_i in range(1, o_size[1]):
                h_s_f[:, s_i, :] = self.rnn_activations_f[i](self.rnn_layers_f[i](
                    self.rnn_dropout_layers_input_f[i](output[:, s_i, :]),
                    self.rnn_dropout_layers_recurrent_f[i](h_s_f[:, s_i - 1, :])
                ))
                h_s_b[:, -(s_i + 1), :] = self.rnn_activations_b[i](self.rnn_layers_b[i](
                    self.rnn_dropout_layers_input_b[i](output[:, -(s_i + 1), :]),
                    self.rnn_dropout_layers_recurrent_b[i](h_s_b[:, s_i - 1, :])
                ))

            output = torch.cat([h_s_f, h_s_b], -1)
            o_size = output.size()
            u_l = o_size[1]
            u_l -= divmod(o_size[1], self.rnn_subsamplings[i])[-1]
            output = output[:, 0:u_l:self.rnn_subsamplings[i], :]
            o_size = output.size()

        return h_s_f, h_s_b

    def nb_trainable_parameters(self):
        nb_params = 0
        for param in self.parameters():
            nb_params += np.product(param.size())

        return nb_params


def main():
    from torch.autograd import Variable
    from torch.nn import functional
    x = Variable(
        torch.rand(2, 40, 862, 192).float()
    )

    nb_cnn_layers = 3

    b = CNNRNNEncoder(
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
