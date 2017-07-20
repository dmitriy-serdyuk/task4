from torch.optim import Adam
from torch.nn import functional
from torch.nn.init import orthogonal, xavier_uniform, xavier_normal

# General variables
batch_size = 64
epochs = 300
dataset_full_path = '/data/lisatmp4/santosjf/task4/attend_to_detect/create_dataset/dcase_2017_task_4_test.hdf5'

grad_clip_norm = 0.
network_loss_weight = True

# Optimizer parameters
optimizer = Adam
optimizer_lr = 1e-4
l1_factor = 0.
l2_factor = 1e-4

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
    rnn_out_dims=[128, 128],
    rnn_activations=[functional.tanh, functional.tanh],
    dropout_cnn=0.5,
    dropout_rnn_input=0.5,
    dropout_rnn_recurrent=0.0,
    rnn_subsamplings=[1])

network_attention_bias = True
network_init = xavier_normal

network_decoder_dim = 256

# EOF
