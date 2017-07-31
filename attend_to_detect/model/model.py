from __future__ import print_function

import torch
from torch import nn
from torch.nn.functional import sigmoid
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss

from .category_specific_branch import CategoryBranch
from attend_to_detect.evaluation import binary_cross_entropy_with_logits


class AttendToDetect(nn.Module):

    def __init__(self, input_dim, decoder_dim, output_classes,
            common_filters=32, common_stride=(2,2), common_kernel_size=(3,3),
            enc_filters=64, enc_stride=(2, 2), enc_kernel_size=(3,3),
            monotonic_attention=False, bias=False):
        super(AttendToDetect, self).__init__()
        self.common = nn.Conv2D(1, common_filters,
                kernel_size=common_kernel_size,
                stride=common_stride,
                padding=((common_kernel_size[0]-1)//2, (common_kernel_size[1]-1)//2)
                )
        self.categoryA = CategoryBranch()
        self.categoryB = CategoryBranch()


    def forward(self, x, n_steps):
        common_feats = self.common(x)
        predA, weightsA = self.categoryA(common_feats, n_steps)
        predB, weightsB = self.categoryB(common_feats, n_steps)

        return (predA, weightsA), (predB, weightsB)


def train_fn(layers, optim, loss_criterion, batch):

    padded_batch = padder(batch)

    x, y = batch

    x = Variable(torch.from_numpy(x.astype('float32')).cuda())
    y = Variable(torch.from_numpy(y.astype('float32')).cuda(), requires_grad=False)

    mask = Variable(torch.ByteTensor(y.size()).fill_(1).cuda(),
                    requires_grad=False)
    for k, l in enumerate(lengths):
        mask[:l, k, :] = 0

    y_hat = layers.forward(x, hidden)

    # Apply mask
    y_hat.masked_fill_(mask, 0.0)
    y.masked_fill_(mask, 0.0)

    loss = loss_criterion(y_hat, y)

    optim.zero_grad()
    loss.backward()
    optim.step()

    return loss.data[0]


def valid_fn(model, criterion, batch):
    x, y, lengths = batch

    x = Variable(x.cuda(), volatile=True)
    y = Variable(y.cuda(), requires_grad=False)

    mask = Variable(torch.ByteTensor(y.size()).fill_(1).cuda(),
                    requires_grad=False)
    for k, l in enumerate(lengths):
        mask[:l, k, :] = 0

    hidden = model.init_hidden(x.size(0))
    y_hat = model.forward(x, hidden)

    # Apply mask
    y_hat.masked_fill_(mask, 0.0)
    y.masked_fill_(mask, 0.0)

    val_loss = criterion(y_hat, y).data[0]
    return val_loss


class Model(nn.Module):
    def __init__(self, encoder, decoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input, output_length):
        contexts = self.encoder(input)
        return self.decoder(*contexts, output_len=output_length)

    def cost(self, output, target):
        pass

    def accuracy(self, output, target):
        pass


class MultipleAttentionModel(nn.Module):
    def __init__(self, encoder, decoders):
        super(MultipleAttentionModel, self).__init__()

        self.encoder = encoder
        self.decoders = decoders
        for i, decoder in enumerate(decoders):
            self.add_module('decoder_{}'.format(i), decoder)

    def forward(self, input, output_length):
        contexts = self.encoder(input, output_length)
        return [decoder(*contexts, output_len=output_length)
                for decoder in self.decoders]

    def cost(self, outputs, target):
        outputs = [output for output, _ in outputs]
        outputs = torch.cat(outputs, 1)
        outputs = outputs.view(-1)

        target = target.view(-1)
        return binary_cross_entropy_with_logits(outputs, target)

    def probs(self, outputs):
        outputs = [output for output, weight in outputs]
        outputs = torch.cat(outputs, 1)
        outputs = sigmoid(outputs)
        return outputs

    def accuracy(self, output, target_categorical):
        probs = self.probs(output)
        predicted = probs > 0.5
        predicted_categorical = predicted.max(-1)[1]
        torch.ne(predicted_categorical, target_categorical).float().sum()


class CTCModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(CTCModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.loss = CTCLoss()

    def forward(self, input, output_length):
        contexts = self.encoder(input, output_length)
        context = torch.cat(contexts, -1)
        flat_context = self.flatten(context)
        flat_decoded = self.decoder(flat_context)
        return self.unflatten(flat_decoded, context.size())

    def flatten(self, tensor):
        size = tensor.size()
        return tensor.view(size[0] * size[1], size[2])

    def unflatten(self, tensor, size):
        return tensor.view(size[0], size[1], tensor.size(1))

    def prepare_output(self, output, concat=True):
        """

        :param output: (B, T, C) output or one-hot label
        :return:
        """
        # (B, C, T)
        output = output.transpose(1, 2).contiguous()
        # (BxC, T, 1)
        output = output.view(-1, output.size(2), 1)
        # (BxC, T, 2): [-act, act]
        if concat:
            output = torch.cat([-output, output], -1)
        else:
            output = output[:, :, 0]

        # (T, BxC, 2): for CTC loss
        output = output.transpose(0, 1).contiguous()
        return output

    def cost(self, output, target):
        batch_size = output.size(0)
        #output = torch.log(self.probs(output))
        output = self.prepare_output(output)

        target = self.prepare_output(
                target.unsqueeze(1), concat=False).int().cpu()
        target_dummy = Variable(torch.ones(target.size()).int())
        #target = target_dummy  #torch.cat(
        #    [target, target_dummy, target_dummy, target_dummy], 0)

        output_sizes = Variable(
            torch.IntTensor([output.size(0)] * output.size(1)))
        label_sizes = 0 * (1 - target[0]) + target[0]
        return self.loss(output, target, output_sizes, label_sizes) / batch_size
        #return ((output - target.cuda().float().unsqueeze(2).expand_as(output))**2).sum()

    def probs(self, output):
        return sigmoid(output)

    def accuracy(self, output, target_categorical):
        probs = self.probs(output)
        predicted = probs > 0.5
        predicted_categorical = predicted.max(-1)[1]
        torch.ne(predicted_categorical, target_categorical).float().sum()


#######################


def main():
    x_ = Variable(torch.randn(5, 4, 161).cuda())
    y = Variable(torch.randn(5, 4, 161).cuda(), requires_grad=False)
    model = RNNModelWithSkipConnections(161, 20, 20, 161).cuda()
    h0_ = model.init_hidden(4)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for k in range(1000):
        y_hat = model.forward(x_, h0_)
        loss = criterion(y_hat, y)
        print('It. {}: {}'.format(k, loss.cpu().data[0]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    main()

