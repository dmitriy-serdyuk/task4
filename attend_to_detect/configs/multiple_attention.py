from torch.optim import Adam, SGD
from torch.nn import functional
from torch.nn.init import orthogonal, xavier_uniform, xavier_normal

# General variables
batch_size = 64
epochs = 300
dataset_full_path = '/Tmp/serdyuk/data/dcase_2017_task_4_test.hdf5'

grad_clip_norm = 0.
network_loss_weight = True

# Optimizer parameters
optimizer = SGD
optimizer_lr = 1e-4
l1_factor = 0.
l2_factor = 0.

encoder_config = dict(
    cnn_channels_in=1,
    cnn_channels_out=[64, 128, 256],
    cnn_kernel_sizes=[(3, 3), (3, 3), (3, 3)],
    cnn_strides=[(2, 2), (2, 2), (2, 2)],
    cnn_paddings=[(1, 1), (1, 1), (1, 1)],
    cnn_activations=[functional.leaky_relu,
                     functional.leaky_relu,
                     functional.leaky_relu],
    max_pool_kernels=[(3, 3), (3, 3), (3, 3)],
    max_pool_strides=[(2, 2), (2, 2), (2, 2)],
    max_pool_paddings=[(1, 1), (1, 1), (1, 1)],
    rnn_input_size=768,
    rnn_out_dims=[256, 256],
    rnn_activations=[functional.tanh, functional.tanh],
    dropout_cnn=0.,
    dropout_rnn_input=0.,
    dropout_rnn_recurrent=0.0,
    rnn_subsamplings=[1])

network_attention_bias = True
network_init = xavier_normal

network_decoder_dim = 512

# EOF
