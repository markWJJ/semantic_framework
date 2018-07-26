# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from functools import reduce
from operator import mul

from tensorflow.contrib.rnn import DropoutWrapper

VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER


def get_last_state(rnn_out_put, mask): # correct
    '''
    get_last_state of rnn output
    :param rnn_out_put: [d1,d2,dn-1,max_len,d]
    :param mask: [d1,d2,dn-1,max_len]
    :return: [d1,d2,dn-1,d]
    '''
    rnn_out_put_flatten = flatten(rnn_out_put, 2)# [X, ml, d]
    mask_flatten = flatten(mask,1) # [X,ml]
    idxs = tf.reduce_sum(tf.cast(mask_flatten,tf.int32),-1) - 1 # [X]
    indices = tf.stack([tf.range(tf.shape(idxs)[0]), idxs], axis=-1) #[X] => [X,2]
    flatten_res = tf.expand_dims(tf.gather_nd(rnn_out_put_flatten, indices),-2 )# #[x,d]->[x,1,d]
    return tf.squeeze(reconstruct(flatten_res,rnn_out_put,2),-2) #[d1,d2,dn-1,1,d] ->[d1,d2,dn-1,d]


def expand_tile(tensor,pattern,tile_num = None, scope=None): # todo: add more func
    with tf.name_scope(scope or 'expand_tile'):
        assert isinstance(pattern,(tuple,list))
        assert isinstance(tile_num,(tuple,list)) or tile_num is None
        assert len(pattern) == len(tile_num) or tile_num is None
        idx_pattern = list([(dim, p) for dim, p in enumerate(pattern)])
        for dim,p in idx_pattern:
            if p == 'x':
                tensor = tf.expand_dims(tensor,dim)
    return tf.tile(tensor,tile_num) if tile_num is not None else tensor


def get_initializer(matrix):
    def _initializer(shape, dtype=None, partition_info=None, **kwargs): return matrix
    return _initializer


def mask(val, mask, name=None):
    if name is None:
        name = 'mask'
    return tf.multiply(val, tf.cast(mask, 'float'), name=name)


def mask_for_high_rank(val, val_mask, name=None):
    val_mask = tf.expand_dims(val_mask, -1)
    return tf.multiply(val, tf.cast(val_mask, tf.float32), name=name or 'mask_for_high_rank')


def exp_mask(val, mask, name=None):
    """Give very negative number to unmasked elements in val.
    For example, [-3, -2, 10], [True, True, False] -> [-3, -2, -1e9].
    Typically, this effectively masks in exponential space (e.g. softmax)
    Args:
        val: values to be masked
        mask: masking boolean tensor, same shape as tensor
        name: name for output tensor

    Returns:
        Same shape as val, where some elements are very small (exponentially zero)
    """
    if name is None:
        name = "exp_mask"
    return tf.add(val, (1 - tf.cast(mask, 'float')) * VERY_NEGATIVE_NUMBER, name=name)


def exp_mask_for_high_rank(val, val_mask, name=None):
    val_mask = tf.expand_dims(val_mask, -1)
    return tf.add(val, (1 - tf.cast(val_mask, tf.float32)) * VERY_NEGATIVE_NUMBER,
                  name=name or 'exp_mask_for_high_rank')


def flatten(tensor, keep):
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
    flat = tf.reshape(tensor, out_shape)
    return flat


def reconstruct(tensor, ref, keep, dim_reduced_keep=None):
    dim_reduced_keep = dim_reduced_keep or keep

    ref_shape = ref.get_shape().as_list() # original shape
    tensor_shape = tensor.get_shape().as_list() # current shape
    ref_stop = len(ref_shape) - keep # flatten dims list
    tensor_start = len(tensor_shape) - dim_reduced_keep  # start
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)] #
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))] #
    # pre_shape = [tf.shape(ref)[i] for i in range(len(ref.get_shape().as_list()[:-keep]))]
    # keep_shape = tensor.get_shape().as_list()[-keep:]
    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    return out


def add_wd(wd, scope=None):
    scope = scope or tf.get_variable_scope().name
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    counter = 0
    with tf.name_scope("weight_decay"):
        for var in variables:
            counter+=1
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd,
                                       name="{}-wd".format('-'.join(str(var.op.name).split('/'))))
            tf.add_to_collection('losses', weight_decay)
        return counter


def add_wd_without_bias(wd, scope=None):
    scope = scope or tf.get_variable_scope().name
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    counter = 0
    with tf.name_scope("weight_decay"):
        for var in variables:
            if len(var.get_shape().as_list()) <= 1: continue
            counter += 1
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd,
                                       name="{}-wd".format('-'.join(str(var.op.name).split('/'))))
            tf.add_to_collection('losses', weight_decay)
        return counter


def add_reg_without_bias(scope=None):
    scope = scope or tf.get_variable_scope().name
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    counter = 0
    for var in variables:
        if len(var.get_shape().as_list()) <= 1: continue
        tf.add_to_collection('reg_vars', var)
        counter += 1
    return counter


def add_var_reg(var):
    tf.add_to_collection('reg_vars', var)


def add_wd_for_var(var, wd):
    with tf.name_scope("weight_decay"):
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd,
                                   name="{}-wd".format('-'.join(str(var.op.name).split('/'))))
        tf.add_to_collection('losses', weight_decay)



# ------------- selu ----------------
import numbers
from tensorflow.contrib import layers
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import utils


# (1) scale inputs to zero mean and unit variance


# (2) use SELUs
def selu(x):
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))


# (3) initialize weights with stddev sqrt(1/n)
# e.g. use:
initializer = layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN')


# (4) use this dropout
def dropout_selu(x, rate, alpha= -1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0,
                 noise_shape=None, seed=None, name=None, training=False):
    """Dropout to a value with rescaling."""

    def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
        keep_prob = 1.0 - rate
        x = ops.convert_to_tensor(x, name="x")
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                                             "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
        alpha.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        if tensor_util.constant_value(keep_prob) == 1:
            return x

        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor + alpha * (1-binary_tensor)

        a = math_ops.sqrt(fixedPointVar / (keep_prob *((1-keep_prob) * math_ops.pow(alpha-fixedPointMean,2) + fixedPointVar)))

        b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
        ret = a * ret + b
        ret.set_shape(x.get_shape())
        return ret

    with ops.name_scope(name, "dropout", [x]) as name:
        return utils.smart_cond(training,
            lambda: dropout_selu_impl(x, rate, alpha, noise_shape, seed, name),
            lambda: array_ops.identity(x))




# ----------------------fundamental-----------------------------
def dropout(x, keep_prob, is_train, noise_shape=None, seed=None, name=None):
    with tf.name_scope(name or "dropout"):
        assert is_train is not None
        if keep_prob < 1.0:
            d = tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed)
            out = tf.cond(is_train, lambda: d, lambda: x)
            return out
        return x


def softmax(logits, mask=None, scope=None):
    with tf.name_scope(scope or "Softmax"):
        if mask is not None:
            logits = exp_mask(logits, mask)
        out = tf.nn.softmax(logits,-1)
        return out


def softsel(target, logits, mask=None, scope=None):
    """
    :param target: [ ..., J, d] dtype=float #(b,sn,sl,ql,d)
    :param logits: [ ..., J], dtype=float
    :param mask: [ ..., J], dtype=bool
    :param scope:
    :return: [..., d], dtype=float
    """
    with tf.name_scope(scope or "Softsel"):
        a = softmax(logits, mask=mask)
        target_rank = len(target.get_shape().as_list())
        out = tf.reduce_sum(tf.expand_dims(a, -1) * target, target_rank - 2)
        return out


def softsel_with_dropout(target, logits, mask=None,
                         keep_prob=1., is_train=None, scope=None):
    """
    :param target: [ ..., J, d] dtype=float #(b,sn,sl,ql,d)
    :param logits: [ ..., J], dtype=float
    :param mask: [ ..., J], dtype=bool
    :param scope:
    :return: [..., d], dtype=float
    """
    with tf.name_scope(scope or "softsel_with_dropout"):
        a = softmax(logits, mask=mask)
        if keep_prob < 1.0:
            assert is_train is not None
            a = tf.cond(is_train, lambda: tf.nn.dropout(a, keep_prob), lambda: a)
        target_rank = len(target.get_shape().as_list())
        out = tf.reduce_sum(tf.expand_dims(a, -1) * target, target_rank - 2)
        return out

# ------------------------------------------------------
# ------------------ special ---------------------------


def _linear(xs,output_size,bias,bias_start=0., scope=None):
    with tf.variable_scope(scope or 'linear_layer'):
        x = tf.concat(xs,-1)
        input_size = x.get_shape()[-1]
        W = tf.get_variable('W', shape=[input_size,output_size],dtype=tf.float32,
                            )
        if bias:
            bias = tf.get_variable('bias', shape=[output_size],dtype=tf.float32,
                                   initializer=tf.constant_initializer(bias_start))
            out = tf.matmul(x, W) + bias
        else:
            out = tf.matmul(x, W)
        return out


def linear(args, output_size, bias, bias_start=0.0, scope=None, squeeze=False, wd=0.0, input_keep_prob=1.0,
           is_train=None):
    if args is None or (isinstance(args, (tuple, list)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (tuple, list)):
        args = [args]

    flat_args = [flatten(arg, 1) for arg in args] # for dense layer [(-1, d)]
    if input_keep_prob < 1.0:
        assert is_train is not None
        flat_args = [tf.cond(is_train, lambda: tf.nn.dropout(arg, input_keep_prob), lambda: arg)# for dense layer [(-1, d)]
                     for arg in flat_args]
    flat_out = _linear(flat_args, output_size, bias, bias_start=bias_start, scope=scope) # dense
    out = reconstruct(flat_out, args[0], 1) # ()
    if squeeze:
        out = tf.squeeze(out, [len(args[0].get_shape().as_list())-1])

    if wd:
        add_reg_without_bias()

    return out


def linear_3d(tensor, hn, bias, bias_start=0.0, scope=None, squeeze=False, wd=0.0, input_keep_prob=1.0,
              is_train=None):

    with tf.variable_scope(scope or 'linear_3d'):
        assert len(tensor.get_shape().as_list()) == 3
        num_int = tensor.get_shape()[0]
        vec_int = tensor.get_shape()[-1]
        weight_3d = tf.get_variable('weight_3d', [num_int, vec_int, hn], tf.float32)

        if input_keep_prob < 1.0:
            assert is_train is not None
            tensor = tf.cond(is_train, lambda: tf.nn.dropout(tensor, input_keep_prob), lambda: tensor)
        if bias:
            bias_3d = tf.get_variable('bias_3d', [num_int, 1, hn], tf.float32,
                                      tf.constant_initializer(bias_start))
            linear_output = tf.matmul(tensor, weight_3d) + bias_3d
        else:
            linear_output = tf.matmul(tensor, weight_3d)

        if squeeze:
            assert hn == 1
            linear_output = tf.squeeze(linear_output, -1)
        if wd:
            add_var_reg(weight_3d)
        return linear_output


def conv1d(in_, filter_size, height, padding, is_train=None, keep_prob=1.0, scope=None):
    with tf.variable_scope(scope or "conv1d"):
        num_channels = in_.get_shape()[-1] #dc
        filter_ = tf.get_variable("filter", shape=[1, height, num_channels, filter_size], dtype='float')
        bias = tf.get_variable("bias", shape=[filter_size], dtype='float')
        strides = [1, 1, 1, 1]
        if is_train is not None and keep_prob < 1.0:
            in_ = dropout(in_, keep_prob, is_train)
        xxc = tf.nn.conv2d(in_, filter_, strides, padding) + bias  # [N*M, JX, W/filter_stride, d] # (b,l,wl,d')
        out = tf.reduce_max(tf.nn.relu(xxc), 2)  # [-1, JX, d] # (b,l,d')
        return out


def multi_conv1d(in_, filter_sizes, heights, padding, is_train=None, keep_prob=1.0, scope=None):
    with tf.variable_scope(scope or "multi_conv1d"):
        assert len(filter_sizes) == len(heights)
        outs = []
        for filter_size, height in zip(filter_sizes, heights):
            if filter_size == 0:
                continue
            # (b*sn,sl,wl,dc)
            out = conv1d(in_, filter_size, height, padding, is_train=is_train, keep_prob=keep_prob, scope="conv1d_{}".format(height)) #(b,l,d')
            outs.append(out)
        concat_out = tf.concat(outs,2) #(b,l,d)
        return concat_out


def highway_layer(arg, bias, bias_start=0.0, scope=None, wd=0.0, input_keep_prob=1.0, is_train=None):
    with tf.variable_scope(scope or "highway_layer"):
        d = arg.get_shape()[-1]  # embedding dim
        trans = linear([arg], d, bias, bias_start=bias_start, scope='trans', wd=wd, input_keep_prob=input_keep_prob,
                       is_train=is_train)
        trans = tf.nn.relu(trans)
        gate = linear([arg], d, bias, bias_start=bias_start, scope='gate', wd=wd, input_keep_prob=input_keep_prob,
                      is_train=is_train)
        gate = tf.nn.sigmoid(gate)
        out = gate * trans + (1 - gate) * arg
        return out

        # read


def highway_network(arg, num_layers, bias, bias_start=0.0, scope=None, wd=0.0, input_keep_prob=1.0, is_train=None):
    with tf.variable_scope(scope or "highway_network"):
        prev = arg
        cur = None
        for layer_idx in range(num_layers):
            cur = highway_layer(prev, bias, bias_start=bias_start, scope="layer_{}".format(layer_idx), wd=wd,
                                input_keep_prob=input_keep_prob, is_train=is_train)
            prev = cur
        return cur


def highway_net(
        input_tensor, hn, bias, bias_start=0.0, scope=None, activation='relu', enable_bn=False,
        wd=0., keep_prob=1.0, is_train=None):
    ivec = input_tensor.get_shape().as_list()[-1]
    with tf.variable_scope(scope or "highway_layer"):
        trans = bn_dense_layer(
            input_tensor, ivec, bias, bias_start, 'map', activation, enable_bn, wd, keep_prob, is_train)
        gate = bn_dense_layer(
            input_tensor, ivec, bias, bias_start, 'gate', 'linear', enable_bn, wd, keep_prob, is_train)
        gate = tf.nn.sigmoid(gate)
        out = gate * trans + (1 - gate) * input_tensor

        return out
# ------------------------------------------------------------
# --------------------get logits------------------------------

def get_logits(args, size, bias, bias_start=0.0, scope=None, mask=None, wd=0.0,
               input_keep_prob=1.0, is_train=None, func=None):
    if func is None:
        func = "linear"
    if func == 'sum':
        return sum_logits(args, mask=mask, name=scope)
    elif func == 'linear':
        return linear_logits(args, bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd, input_keep_prob=input_keep_prob,
                             is_train=is_train)
    elif func == 'double':
        return double_linear_logits(args, size, bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd, input_keep_prob=input_keep_prob,
                                    is_train=is_train)
    elif func == 'dot':
        assert len(args) == 2
        arg = args[0] * args[1]
        return sum_logits([arg], mask=mask, name=scope)
    elif func == 'mul_linear':
        assert len(args) == 2
        arg = args[0] * args[1]
        return linear_logits([arg], bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd, input_keep_prob=input_keep_prob,
                             is_train=is_train)
    elif func == 'proj':
        assert len(args) == 2
        d = args[1].get_shape()[-1]
        proj = linear([args[0]], d, False, bias_start=bias_start, scope=scope, wd=wd, input_keep_prob=input_keep_prob,
                      is_train=is_train)
        return sum_logits([proj * args[1]], mask=mask)
    elif func == 'tri_linear':
        assert len(args) == 2
        new_arg = args[0] * args[1]
        return linear_logits([args[0], args[1], new_arg], bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd, input_keep_prob=input_keep_prob,
                             is_train=is_train)
    else:
        raise Exception()

def double_linear_logits(args, size, bias, bias_start=0.0, scope=None, mask=None, wd=0.0, input_keep_prob=1.0,
                         is_train=None):
    with tf.variable_scope(scope or "Double_Linear_Logits"):
        first = tf.tanh(linear(args, size, bias, bias_start=bias_start, scope='first',
                               wd=wd, input_keep_prob=input_keep_prob, is_train=is_train))
        second = linear(first, 1, bias, bias_start=bias_start, squeeze=True, scope='second',
                        wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
        if mask is not None:
            second = exp_mask(second, mask)
        return second


def linear_logits(args, bias, bias_start=0.0, scope=None, mask=None, wd=0.0, input_keep_prob=1.0, is_train=None):
    with tf.variable_scope(scope or "Linear_Logits"):
        logits = linear(args, 1, bias, bias_start=bias_start, squeeze=True, scope='first',
                        wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
        if mask is not None:
            logits = exp_mask(logits, mask)
        return logits


def sum_logits(args, mask=None, name=None):
    with tf.name_scope(name or "sum_logits"):
        if args is None or (isinstance(args, (tuple, list)) and not args):
            raise ValueError("`args` must be specified")
        if not isinstance(args, (tuple, list)):
            args = [args]
        rank = len(args[0].get_shape())
        logits = sum(tf.reduce_sum(arg, rank - 1) for arg in args)
        if mask is not None:
            logits = exp_mask(logits, mask)
        return logits
# ------------------------------------------------------------
# ------------------------------------------------------------

# ### --------------- other key func


def feature_combination(org_tensor, new_features, wd=0., keep_prob=1., is_train=None, scope=None):
    """
    Features Combination 1: ruminating layer implementation
    z = tanh(Wz0*in + Wz1*x1+..Wzn*xn+b);
    f = tanh(Wf0*in + Wf1*x1+..Wfn*xn+b)
    out = fquan\elem∗⁆t in+(1-f)\elem∗z
    :param org_tensor: rank 3 with shape [bs,sl,vec]
    :param new_features: list of tensor with rank 2 [bs,vec_x1] or [bs,sl,vec_x2]
    :param wd: 
    :param keep_prob: 
    :param is_train: 
    :param scope: 
    :return: 
    """

    with tf.variable_scope(scope or 'fea_comb'):
        bs, sl, vec = tf.shape(org_tensor)[0],tf.shape(org_tensor)[1],tf.shape(org_tensor)[2]
        vec_int = org_tensor.get_shape()[2]
        features = [new_fea if len(new_fea.get_shape().as_list())==3 else tf.expand_dims(new_fea, 1)
                    for new_fea in new_features]

        # --- z ---
        z_0 = linear([org_tensor], vec_int, True, scope='linear_W_z_0',
                     wd=wd, input_keep_prob=keep_prob, is_train=is_train)
        z_other = [linear([fea], vec_int, False, scope='linear_W_z_%d' % (idx_f + 1),
                          wd=wd, input_keep_prob=keep_prob, is_train=is_train)
                   for idx_f, fea in enumerate(features)]
        z = tf.nn.tanh(sum([z_0] + z_other))

        # --- f ---
        f_0 = linear([org_tensor], vec_int, True, scope='linear_W_f_0',
                     wd=wd, input_keep_prob=keep_prob, is_train=is_train)
        f_other = [linear([fea], vec_int, False, scope='linear_W_f_%d' % (idx_f + 1),
                          wd=wd, input_keep_prob=keep_prob, is_train=is_train)
                   for idx_f, fea in enumerate(features)]
        f = tf.nn.sigmoid(sum([f_0] + f_other))

        return f*org_tensor + (1-f)*z


def pooling_with_mask(rep_tensor, rep_mask, method='max', scope=None):
    # rep_tensor have one more rank than rep_mask
    with tf.name_scope(scope or '%s_pooling' % method):

        if method == 'max':
            rep_tensor_masked = exp_mask_for_high_rank(rep_tensor, rep_mask)
            output = tf.reduce_max(rep_tensor_masked, -2)
        elif method == 'mean':
            rep_tensor_masked = mask_for_high_rank(rep_tensor, rep_mask)  # [...,sl,hn]
            rep_sum = tf.reduce_sum(rep_tensor_masked, -2)  #[..., hn]
            denominator = tf.reduce_sum(tf.cast(rep_mask, tf.int32), -1, True)  # [..., 1]
            denominator = tf.where(tf.equal(denominator, tf.zeros_like(denominator, tf.int32)),
                                   tf.ones_like(denominator, tf.int32),
                                   denominator)
            output = rep_sum / tf.cast(denominator, tf.float32)
        else:
            raise AttributeError('No Pooling method name as %s' % method)
        return output


def fusion_two_mat(input1, input2, hn=None, scope=None, wd=0., keep_prob=1., is_train=None):
    ivec1 = input1.get_shape()[-1]
    ivec2 = input2.get_shape()[-1]
    if hn is None:
        hn = ivec1
    with tf.variable_scope(scope or 'fusion_two_mat'):
        part1 = linear(input1, hn, False, 0., 'linear_1', False, wd, keep_prob, is_train)
        part2 = linear(input2, hn, True, 0., 'linear_2', False, wd, keep_prob, is_train)
        return part1 + part2


# # ----------- with normalization ------------
def bn_dense_layer(input_tensor, hn, bias, bias_start=0.0, scope=None,
                   activation='relu', enable_bn=True,
                   wd=0., keep_prob=1.0, is_train=None):
    if is_train is None:
        is_train = False

    # activation
    if activation == 'linear':
        activation_func = tf.identity
    elif activation == 'relu':
        activation_func = tf.nn.relu
    elif activation == 'elu':
        activation_func = tf.nn.elu
    elif activation == 'selu':
        activation_func = selu
    elif activation == 'sigmoid':
        activation_func = tf.nn.sigmoid
    elif activation == 'tanh':
        activation_func = tf.nn.tanh
    else:
        raise AttributeError('no activation function named as %s' % activation)

    with tf.variable_scope(scope or 'bn_dense_layer'):
        linear_map = linear(input_tensor, hn, bias, bias_start, 'linear_map',
                            False, wd, keep_prob, is_train)
        if enable_bn:
            # with tf.variable_scope('bn_module'):
            #     linear_map = tf.cond(
            #         is_train,
            #         lambda: tf.contrib.layers.batch_norm(
            #             linear_map, center=True, scale=True, is_training=True,
            #             scope='bn'),
            #         lambda: tf.contrib.layers.batch_norm(
            #             linear_map, center=True, scale=True, is_training=False,
            #             scope='bn', reuse=True),
            #     )
            linear_map = tf.contrib.layers.batch_norm(
                linear_map, center=True, scale=True, is_training=is_train,
                updates_collections=None,  decay=0.9,
                scope='bn')

        return activation_func(linear_map)


def bn_layer(input_tensor, is_train, enable,scope=None):
    with tf.variable_scope(scope or 'bn_layer'):
        if enable:
            return tf.contrib.layers.batch_norm(
                input_tensor, center=True, scale=True, is_training=is_train, scope='bn')
        else:
            return tf.identity(input_tensor)


# -------------- emb mat--------------
def generate_embedding_mat(dict_size, emb_len, init_mat=None, extra_mat=None,
                           extra_trainable=False, scope=None):
    """
    generate embedding matrix for looking up
    :param dict_size: indices 0 and 1 corresponding to empty and unknown token
    :param emb_len:
    :param init_mat: init mat matching for [dict_size, emb_len]
    :param extra_mat: extra tensor [extra_dict_size, emb_len]
    :param extra_trainable:
    :param scope:
    :return: if extra_mat is None, return[dict_size+extra_dict_size,emb_len], else [dict_size,emb_len]
    """
    with tf.variable_scope(scope or 'gene_emb_mat'):
        emb_mat_ept_and_unk = tf.constant(value=0, dtype=tf.float32, shape=[2, emb_len])
        if init_mat is None:
            emb_mat_other = tf.get_variable('emb_mat',[dict_size - 2, emb_len], tf.float32)
        else:
            emb_mat_other = tf.get_variable("emb_mat",[dict_size - 2, emb_len], tf.float32,
                                            initializer=tf.constant_initializer(init_mat[2:], dtype=tf.float32,
                                                                                verify_shape=True))
        emb_mat = tf.concat([emb_mat_ept_and_unk, emb_mat_other], 0)

        if extra_mat is not None:
            if extra_trainable:
                extra_mat_var = tf.get_variable("extra_emb_mat",extra_mat.shape, tf.float32,
                                                initializer=tf.constant_initializer(extra_mat,
                                                                                    dtype=tf.float32,
                                                                                    verify_shape=True))
                return tf.concat([emb_mat, extra_mat_var], 0)
            else:
                extra_mat_con = tf.constant(extra_mat, dtype=tf.float32)
                return tf.concat([emb_mat, extra_mat_con], 0)
        else:
            return emb_mat


def token_and_char_emb(if_token_emb=True, context_token=None, tds=None, tel=None,
                       token_emb_mat=None, glove_emb_mat=None,
                       if_char_emb=True, context_char=None, cds=None, cel=None,
                       cos=None, ocd=None, fh=None, use_highway=True,highway_layer_num=None,
                       wd=0., keep_prob=1., is_train=None):
    with tf.variable_scope('token_and_char_emb'):
        if if_token_emb:
            with tf.variable_scope('token_emb'):
                token_emb_mat = generate_embedding_mat(tds, tel, init_mat=token_emb_mat,
                                                       extra_mat=glove_emb_mat,
                                                       scope='gene_token_emb_mat')

                c_token_emb = tf.nn.embedding_lookup(token_emb_mat, context_token)  # bs,sl,tel

        if if_char_emb:
            with tf.variable_scope('char_emb'):
                char_emb_mat = generate_embedding_mat(cds, cel, scope='gene_char_emb_mat')
                c_char_lu_emb = tf.nn.embedding_lookup(char_emb_mat, context_char)  # bs,sl,tl,cel

                assert sum(ocd) == cos and len(ocd) == len(fh)

                with tf.variable_scope('conv'):
                    c_char_emb = multi_conv1d(c_char_lu_emb, ocd, fh, "VALID",
                                              is_train, keep_prob, scope="xx")  # bs,sl,cocn
        if if_token_emb and if_char_emb:
            c_emb = tf.concat([c_token_emb, c_char_emb], -1)  # bs,sl,cocn+tel
        elif if_token_emb:
            c_emb = c_token_emb
        elif if_char_emb:
            c_emb = c_char_emb
        else:
            raise AttributeError('No embedding!')

    if use_highway:
        with tf.variable_scope('highway'):
            c_emb = highway_network(c_emb, highway_layer_num, True, wd=wd,
                                    input_keep_prob=keep_prob,is_train=is_train)
    return c_emb


def generate_feature_emb_for_c_and_q(feature_dict_size, feature_emb_len,
                                     feature_name , c_feature, q_feature=None, scope=None):
    with tf.variable_scope(scope or '%s_feature_emb' % feature_name):
        emb_mat = generate_embedding_mat(feature_dict_size, feature_emb_len, scope='emb_mat')
        c_feature_emb = tf.nn.embedding_lookup(emb_mat, c_feature)
        if q_feature is not None:
            q_feature_emb = tf.nn.embedding_lookup(emb_mat, q_feature)
        else:
            q_feature_emb = None
        return c_feature_emb, q_feature_emb

# -------------------END---------------------

"""
@Author: Tao Shen
Tensorflow implementation for CNN in sentence encoding
"""

import tensorflow as tf

# ----------- multi-window CNN -------------
def cnn_for_context_fusion(
        rep_tensor, rep_mask, filter_sizes=(3,4,5), num_filters=200, scope=None,
        is_train=None, keep_prob=1., wd=0.):
    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape().as_list()[2]

    with tf.variable_scope(scope or 'cnn_for_sentence_encoding'):
        rep_tensor = mask_for_high_rank(rep_tensor, rep_mask)
        rep_tensor_expand = tf.expand_dims(rep_tensor, 3)  # bs, sl,
        rep_tensor_expand_dp = dropout(rep_tensor_expand, keep_prob, is_train)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, ivec, 1, num_filters]
                W = tf.get_variable('W', filter_shape, tf.float32)
                b = tf.get_variable('b', [num_filters], tf.float32)

                # # pading in the sequence
                if filter_size % 2 == 1:
                    padding_front = padding_back = int((filter_size - 1) / 2)
                else:
                    padding_front = (filter_size - 1) // 2
                    padding_back = padding_front + 1
                padding = [[0, 0], [padding_front, padding_back], [0, 0], [0, 0]]
                rep_tensor_expand_dp_pad = tf.pad(rep_tensor_expand_dp, padding)

                conv = tf.nn.conv2d(
                    rep_tensor_expand_dp_pad,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")  # bs, sl, 1, fn
                h_squeeze = tf.squeeze(h, [2])  # bs, sl, fn
                pooled_outputs.append(h_squeeze)

        # Combine all the pooled features
        result = tf.concat(pooled_outputs, 2)  # bs, sl, 3 * fn

        if wd > 0.:
            add_reg_without_bias()

        return result


def cnn_for_sentence_encoding( # kim
        rep_tensor, rep_mask, filter_sizes=(3,4,5), num_filters=200, scope=None,
        is_train=None, keep_prob=1., wd=0.):
    """

    :param rep_tensor:
    :param rep_mask:
    :param filter_sizes:
    :param num_filters:
    :param scope:
    :param is_train:
    :param keep_prob:
    :param wd:
    :return:
    """
    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape().as_list()[2]

    with tf.variable_scope(scope or 'cnn_for_sentence_encoding'):
        rep_tensor = mask_for_high_rank(rep_tensor, rep_mask)
        rep_tensor_expand = tf.expand_dims(rep_tensor, 3)
        rep_tensor_expand_dp = dropout(rep_tensor_expand, keep_prob, is_train)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, ivec, 1, num_filters]
                W = tf.get_variable('W', filter_shape, tf.float32)
                b = tf.get_variable('b', [num_filters], tf.float32)

                conv = tf.nn.conv2d(
                    rep_tensor_expand_dp,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")  # bs, sl-fs+1, 1, fn
                # Maxpooling over the outputs
                # pooled = tf.nn.max_pool(
                #     h,
                #     ksize=[1, sl - filter_size + 1, 1, 1],
                #     strides=[1, 1, 1, 1],
                #     padding='VALID',
                #     name="pool")
                pooled = tf.reduce_max(h, 1, True)  # bs, 1, 1, fn
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        if wd > 0.:
            add_reg_without_bias()

        return h_pool_flat


# ----------- hierarchical CNN -------------
def hierarchical_cnn_res_gate(
        rep_tensor, rep_mask, n_gram=5, layer_num=5, hn=None, scope=None,
        is_train=None, keep_prob=1., wd=0.):
    # padding
    if n_gram % 2 == 1:
        padding_front = padding_back = int((n_gram - 1) / 2)
    else:
        padding_front = (n_gram - 1) // 2
        padding_back = padding_front + 1
    padding = [[0, 0], [padding_front, padding_back], [0, 0], [0, 0]]

    # lengths
    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    org_ivec = rep_tensor.get_shape().as_list()[2]
    ivec = hn or org_ivec

    with tf.variable_scope(scope or 'cnn_for_sentence_encoding'):
        rep_tensor = mask_for_high_rank(rep_tensor, rep_mask)  # bs, sl, hn

        iter_rep = rep_tensor
        layer_res_list = []

        for layer_idx in range(layer_num):
            with tf.variable_scope("conv_maxpool_%s" % layer_idx):

                iter_rep_etd = tf.expand_dims(iter_rep, 3)  # bs,sl,hn,1
                iter_rep_etd_dp = dropout(iter_rep_etd, keep_prob, is_train)
                # Convolution Layer
                feature_size = org_ivec if layer_idx == 0 else ivec
                filter_shape = [n_gram, feature_size, 1, 2 * ivec]
                W = tf.get_variable('W', filter_shape, tf.float32)
                b = tf.get_variable('b', [2 * ivec], tf.float32)
                iter_rep_etd_pad = tf.pad(iter_rep_etd_dp, padding)
                conv = tf.nn.conv2d(
                    iter_rep_etd_pad,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                map_res = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")  # bs,sl,1,2hn
                map_res = tf.squeeze(map_res, [2])  # bs,sl,2*hn
                # gate
                map_res_a, map_res_b = tf.split(map_res, num_or_size_splits=2, axis=2)
                iter_rep = map_res_a * tf.nn.sigmoid(map_res_b)

                # res
                if len(layer_res_list) > 0:
                    iter_rep = iter_rep + layer_res_list[-1]
                layer_res_list.append(iter_rep)

        if wd > 0.:
            add_reg_without_bias()
        return iter_rep


def dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None,
                dtype=None, parallel_iterations=None, swap_memory=False,
                time_major=False, scope=None):
    assert not time_major  # TODO : to be implemented later!
    flat_inputs = flatten(inputs, 2)  # [-1, J, d]
    flat_len = None if sequence_length is None else tf.cast(flatten(sequence_length, 0), 'int64')

    flat_outputs, final_state = tf.nn.dynamic_rnn(cell, flat_inputs, sequence_length=flat_len,
                                             initial_state=initial_state, dtype=dtype,
                                             parallel_iterations=parallel_iterations, swap_memory=swap_memory,
                                             time_major=time_major, scope=scope)

    outputs = reconstruct(flat_outputs, inputs, 2)
    return outputs, final_state


def bw_dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None,
                   dtype=None, parallel_iterations=None, swap_memory=False,
                   time_major=False, scope=None):
    assert not time_major  # TODO : to be implemented later!

    flat_inputs = flatten(inputs, 2)  # [-1, J, d]
    flat_len = None if sequence_length is None else tf.cast(flatten(sequence_length, 0), 'int64')

    flat_inputs = tf.reverse(flat_inputs, [1]) if sequence_length is None \
        else tf.reverse_sequence(flat_inputs, sequence_length, 1)
    flat_outputs, final_state = tf.nn.dynamic_rnn(cell, flat_inputs, sequence_length=flat_len,
                                             initial_state=initial_state, dtype=dtype,
                                             parallel_iterations=parallel_iterations, swap_memory=swap_memory,
                                             time_major=time_major, scope=scope)
    flat_outputs = tf.reverse(flat_outputs, [1]) if sequence_length is None \
        else tf.reverse_sequence(flat_outputs, sequence_length, 1)

    outputs = reconstruct(flat_outputs, inputs, 2)
    return outputs, final_state


def bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=None,
                              initial_state_fw=None, initial_state_bw=None,
                              dtype=None, parallel_iterations=None,
                              swap_memory=False, time_major=False, scope=None):
    assert not time_major

    flat_inputs = flatten(inputs, 2)  # [-1, J, d]
    flat_len = None if sequence_length is None else tf.cast(flatten(sequence_length, 0), 'int64')

    (flat_fw_outputs, flat_bw_outputs), final_state = \
        tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, flat_inputs, sequence_length=flat_len,
                                   initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw,
                                   dtype=dtype, parallel_iterations=parallel_iterations, swap_memory=swap_memory,
                                   time_major=time_major, scope=scope)

    fw_outputs = reconstruct(flat_fw_outputs, inputs, 2)
    bw_outputs = reconstruct(flat_bw_outputs, inputs, 2)
    # FIXME : final state is not reshaped!
    return (fw_outputs, bw_outputs), final_state


# ---------------- RNN Cell ----------------
class SwitchableDropoutWrapper(DropoutWrapper):
    def __init__(self, cell, is_train, input_keep_prob=1.0, output_keep_prob=1.0,
                 seed=None):
        super(SwitchableDropoutWrapper, self).__init__(cell,
                                                       input_keep_prob=input_keep_prob,
                                                       output_keep_prob=output_keep_prob,
                                                       seed=seed)
        self.is_train = is_train

    def __call__(self, inputs, state, scope=None):
        outputs_do, new_state_do = super(SwitchableDropoutWrapper, self).__call__(inputs, state, scope=scope)
        tf.get_variable_scope().reuse_variables()
        outputs, new_state = self._cell(inputs, state, scope)
        outputs = tf.cond(self.is_train, lambda: outputs_do, lambda: outputs)
        if isinstance(state, tf.contrib.rnn.LSTMStateTuple):
            new_state = state.__class__(*[tf.cond(self.is_train, lambda: new_state_do_i, lambda: new_state_i)
                                          for new_state_do_i, new_state_i in zip(new_state_do, new_state)])
        elif isinstance(state, tuple):
            new_state = state.__class__([tf.cond(self.is_train, lambda: new_state_do_i, lambda: new_state_i)
                                         for new_state_do_i, new_state_i in zip(new_state_do, new_state)])
        else:
            new_state = tf.cond(self.is_train, lambda: new_state_do, lambda: new_state)
        return outputs, new_state


class NormalSRUCell(tf.contrib.rnn.RNNCell):
    def __init__(self, num_units, activation=None, reuse=None):
        super(NormalSRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or tf.nn.tanh

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """

        :param inputs: [bs, vec]
        :param state:
        :param scope:
        :return:
        """
        with tf.variable_scope(scope or "SRU_cell"):
            b_f = tf.get_variable('b_f', [self._num_units], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0))
            b_r = tf.get_variable('b_r', [self._num_units], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0))
            U_d = bn_dense_layer(inputs, 3 * self._num_units, False, 0., 'get_frc', 'linear')  # bs, 3vec
            x_t = tf.identity(inputs, 'x_t')
            x_dt, f_t, r_t = tf.split(U_d, 3, 1)
            f_t = tf.nn.sigmoid(f_t + b_f)
            r_t = tf.nn.sigmoid(r_t + b_r)
            c_t = f_t * state + (1 - f_t) * x_dt
            h_t = r_t * self._activation(c_t) + (1 - r_t) * x_t
            return h_t, c_t


# ---------- accelerated SRU --------------
def bi_sru_recurrent_network(
        rep_tensor, rep_mask, is_train=None, keep_prob=1., wd=0.,
        scope=None, hn=None, reuse=None):
    """

    :param rep_tensor: [Tensor/tf.float32] rank is 3 with shape [batch_size/bs, max_sent_len/sl, vec]
    :param rep_mask: [Tensor/tf.bool]rank is 2 with shape [bs,sl]
    :param is_train: [Scalar Tensor/tf.bool]scalar tensor to indicate whether the mode is training or not
    :param keep_prob: [float] dropout keep probability in the range of (0,1)
    :param wd: [float]for L2 regularization, if !=0, add tensors to tf collection "reg_vars"
    :param scope: [str]variable scope name
    :param hn:
    :param
    :return: [Tensor/tf.float32] with shape [bs, sl, 2vec] for forward and backward
    """
    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape().as_list()[2]
    ivec = hn or ivec

    with tf.variable_scope(scope or 'bi_sru_recurrent_network'):
        # U_d = bn_dense_layer([rep_tensor], 6 * ivec, False, 0., 'get_frc', 'linear',
        #                    False, wd, keep_prob, is_train)  # bs, sl, 6vec
        # U_d_fw, U_d_bw = tf.split(U_d, 2, 2)
        with tf.variable_scope('forward'):
            U_d_fw = bn_dense_layer([rep_tensor], 3 * ivec, False, 0., 'get_frc_fw', 'linear',
                                    False, wd, keep_prob, is_train)  # bs, sl, 6vec
            U_fw = tf.concat([rep_tensor, U_d_fw], -1)
            fw_SRUCell = SwitchableDropoutWrapper(SRUCell(ivec, tf.nn.tanh, reuse), is_train, keep_prob)
            fw_output, _ = dynamic_rnn(
                fw_SRUCell, U_fw, tf.reduce_sum(tf.cast(rep_mask, tf.int32), -1),
                dtype=tf.float32, scope='forward_sru')  # bs, sl, vec

        with tf.variable_scope('backward'):
            U_d_bw = bn_dense_layer([rep_tensor], 3 * ivec, False, 0., 'get_frc_bw', 'linear',
                                    False, wd, keep_prob, is_train)  # bs, sl, 6vec
            U_bw = tf.concat([rep_tensor, U_d_bw], -1)
            bw_SRUCell = SwitchableDropoutWrapper(SRUCell(ivec, tf.nn.tanh, reuse), is_train, keep_prob)
            bw_output, _ = bw_dynamic_rnn(
                bw_SRUCell, U_bw, tf.reduce_sum(tf.cast(rep_mask, tf.int32), -1),
                dtype=tf.float32, scope='backward_sru')  # bs, sl, vec

        all_output = tf.concat([fw_output, bw_output], -1)  # bs, sl, 2vec
        return all_output


class SRUCell(tf.contrib.rnn.RNNCell):
    def __init__(self, num_units, activation=None, reuse=None):
        super(SRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or tf.nn.tanh

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """

        :param inputs: [bs,4*vec]
        :param state: [bs, vec]
        :return:
        """
        b_f = tf.get_variable('b_f', [self._num_units], dtype=tf.float32,
                              initializer=tf.constant_initializer(0))
        b_r = tf.get_variable('b_r', [self._num_units], dtype=tf.float32,
                              initializer=tf.constant_initializer(0))

        x_t, x_dt, f_t, r_t = tf.split(inputs, 4, 1)
        f_t = tf.nn.sigmoid(f_t + b_f)
        r_t = tf.nn.sigmoid(r_t + b_r)
        c_t = f_t * state + (1 - f_t) * x_dt
        h_t = r_t * self._activation(c_t) + (1 - r_t) * x_t
        return h_t, c_t

# ----------------- END --------------------


# ----------------- RNN integration -------
def contextual_bi_rnn(tensor_rep, mask_rep, hn, cell_type, only_final=False,
                      wd=0., keep_prob=1.,is_train=None, scope=None):
    """
    fusing contextual information using bi-direction rnn
    :param tensor_rep: [..., sl, vec]
    :param mask_rep: [..., sl]
    :param hn:
    :param cell_type: 'gru', 'lstm', basic_lstm' and 'basic_rnn'
    :param only_final: True or False
    :param wd:
    :param keep_prob:
    :param is_train:
    :param scope:
    :return:
    """
    with tf.variable_scope(scope or 'contextual_bi_rnn'): # correct
        reuse = None if not tf.get_variable_scope().reuse else True

        if cell_type == 'sru':
            rnn_outputs = bi_sru_recurrent_network(
                tensor_rep, mask_rep, is_train, keep_prob, wd, 'bi_sru_recurrent_network', hn, reuse)
        else:
            if cell_type == 'gru':
                cell_fw = tf.contrib.rnn.GRUCell(hn, reuse=reuse)
                cell_bw = tf.contrib.rnn.GRUCell(hn, reuse=reuse)
            elif cell_type == 'lstm':
                cell_fw = tf.contrib.rnn.LSTMCell(hn, reuse=reuse)
                cell_bw = tf.contrib.rnn.LSTMCell(hn, reuse=reuse)
            elif cell_type == 'basic_lstm':
                cell_fw = tf.contrib.rnn.BasicLSTMCell(hn, reuse=reuse)
                cell_bw = tf.contrib.rnn.BasicLSTMCell(hn, reuse=reuse)
            elif cell_type == 'basic_rnn':
                cell_fw = tf.contrib.rnn.BasicRNNCell(hn, reuse=reuse)
                cell_bw = tf.contrib.rnn.BasicRNNCell(hn, reuse=reuse)
            elif cell_type == 'sru_normal':
                cell_fw = NormalSRUCell(hn, reuse=reuse)
                cell_bw = NormalSRUCell(hn, reuse=reuse)
            else:
                raise AttributeError('no cell type \'%s\'' % cell_type)
            cell_dp_fw = SwitchableDropoutWrapper(cell_fw,is_train,keep_prob)
            cell_dp_bw = SwitchableDropoutWrapper(cell_bw,is_train,keep_prob)

            tensor_len = tf.reduce_sum(tf.cast(mask_rep, tf.int32), -1)  # [bs]

            (outputs_fw, output_bw), _=bidirectional_dynamic_rnn(
                cell_dp_fw, cell_dp_bw, tensor_rep, tensor_len,
                dtype=tf.float32)
            rnn_outputs = tf.concat([outputs_fw,output_bw],-1)  # [...,sl,2hn]

        if wd > 0:
            add_reg_without_bias()
        if not only_final:
            return rnn_outputs  # [....,sl, 2hn]
        else:
            return get_last_state(rnn_outputs, mask_rep)  # [...., 2hn]


def traditional_attention(rep_tensor, rep_mask, scope=None,
                          keep_prob=1., is_train=None, wd=0., activation='elu',
                          tensor_dict=None, name=None):
    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape()[2]
    with tf.variable_scope(scope or 'traditional_attention'):
        rep_tensor_map = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map', activation,
                                        False, wd, keep_prob, is_train)

        rep_tensor_logits = get_logits([rep_tensor_map], None, False, scope='self_attn_logits',
                                       mask=rep_mask, input_keep_prob=keep_prob, is_train=is_train)  # bs,sl
        attn_res = softsel(rep_tensor, rep_tensor_logits, rep_mask)  # bs,vec

        # save attn
        if tensor_dict is not None and name is not None:
            tensor_dict[name] = tf.nn.softmax(rep_tensor_logits)

        return attn_res


def multi_dimensional_attention(rep_tensor, rep_mask, scope=None,
                                keep_prob=1., is_train=None, wd=0., activation='elu',
                                tensor_dict=None, name=None):
    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape()[2]
    with tf.variable_scope(scope or 'multi_dimensional_attention'):
        map1 = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map1', activation,
                              False, wd, keep_prob, is_train)
        map2 = bn_dense_layer(map1, ivec, True, 0., 'bn_dense_map2', 'linear',
                              False, wd, keep_prob, is_train)
        map2_masked = exp_mask_for_high_rank(map2, rep_mask)

        soft = tf.nn.softmax(map2_masked, 1)  # bs,sl,vec
        attn_output = tf.reduce_sum(soft * rep_tensor, 1)  # bs, vec

        # save attn
        if tensor_dict is not None and name is not None:
            tensor_dict[name] = soft

        return attn_output


def directional_attention_with_dense(
        rep_tensor, rep_mask, direction=None, scope=None,
        keep_prob=1., is_train=None, wd=0., activation='elu',
        tensor_dict=None, name=None, hn=None):

    def scaled_tanh(x, scale=5.):
        return scale * tf.nn.tanh(1./scale * x)

    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape().as_list()[2]
    ivec = hn or ivec
    with tf.variable_scope(scope or 'directional_attention_%s' % direction or 'diag'):
        # mask generation
        sl_indices = tf.range(sl, dtype=tf.int32)
        sl_col, sl_row = tf.meshgrid(sl_indices, sl_indices)
        if direction is None:
            direct_mask = tf.cast(tf.diag(- tf.ones([sl], tf.int32)) + 1, tf.bool)
        else:
            if direction == 'forward':
                direct_mask = tf.greater(sl_row, sl_col)
            else:
                direct_mask = tf.greater(sl_col, sl_row)
        direct_mask_tile = tf.tile(tf.expand_dims(direct_mask, 0), [bs, 1, 1])  # bs,sl,sl
        rep_mask_tile = tf.tile(tf.expand_dims(rep_mask, 1), [1, sl, 1])  # bs,sl,sl
        attn_mask = tf.logical_and(direct_mask_tile, rep_mask_tile)  # bs,sl,sl

        # non-linear
        rep_map = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map', activation,
                                 False, wd, keep_prob, is_train)
        rep_map_tile = tf.tile(tf.expand_dims(rep_map, 1), [1, sl, 1, 1])  # bs,sl,sl,vec
        rep_map_dp = dropout(rep_map, keep_prob, is_train)

        # attention
        with tf.variable_scope('attention'):  # bs,sl,sl,vec
            f_bias = tf.get_variable('f_bias',[ivec], tf.float32, tf.constant_initializer(0.))
            dependent = linear(rep_map_dp, ivec, False, scope='linear_dependent')  # bs,sl,vec
            dependent_etd = tf.expand_dims(dependent, 1)  # bs,1,sl,vec
            head = linear(rep_map_dp, ivec, False, scope='linear_head') # bs,sl,vec
            head_etd = tf.expand_dims(head, 2)  # bs,sl,1,vec

            logits = scaled_tanh(dependent_etd + head_etd + f_bias, 5.0)  # bs,sl,sl,vec

            logits_masked = exp_mask_for_high_rank(logits, attn_mask)
            attn_score = tf.nn.softmax(logits_masked, 2)  # bs,sl,sl,vec
            attn_score = mask_for_high_rank(attn_score, attn_mask)

            attn_result = tf.reduce_sum(attn_score * rep_map_tile, 2)  # bs,sl,vec

        with tf.variable_scope('output'):
            o_bias = tf.get_variable('o_bias',[ivec], tf.float32, tf.constant_initializer(0.))
            # input gate
            fusion_gate = tf.nn.sigmoid(
                linear(rep_map, ivec, True, 0., 'linear_fusion_i', False, wd, keep_prob, is_train) +
                linear(attn_result, ivec, True, 0., 'linear_fusion_a', False, wd, keep_prob, is_train) +
                o_bias)
            output = fusion_gate * rep_map + (1-fusion_gate) * attn_result
            output = mask_for_high_rank(output, rep_mask)

        # save attn
        if tensor_dict is not None and name is not None:
            tensor_dict[name + '_dependent'] = dependent
            tensor_dict[name + '_head'] = head
            tensor_dict[name] = attn_score
            tensor_dict[name + '_gate'] = fusion_gate
        return output


# ----------------- bi-blosan -----------
def bi_directional_simple_block_attention(
        rep_tensor, rep_mask, block_len=5, scope=None,
        keep_prob=1., is_train=None, wd=0., activation='elu', hn=None):
    with tf.variable_scope(scope or 'bi_directional_simple_block_attn'):

        fw_attn_res = simple_block_attention(
            rep_tensor, rep_mask, block_len, "forward_attn", "forward",
            keep_prob, is_train, wd, activation, hn)
        bw_attn_res = simple_block_attention(
            rep_tensor, rep_mask, block_len, "backward_attn", "backward",
            keep_prob, is_train, wd, activation, hn)
        attn_res = tf.concat([fw_attn_res, bw_attn_res], -1)
        return attn_res


def simple_block_attention(
        rep_tensor, rep_mask, block_len=5, scope=None, direction=None,
        keep_prob=1., is_train=None, wd=0., activation='elu', hn=None):
    assert direction is not None

    def scaled_tanh(x, scale=5.):
        return scale * tf.nn.tanh(1. / scale * x)

    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    org_ivec = rep_tensor.get_shape().as_list()[2]
    ivec = hn or org_ivec
    with tf.variable_scope(scope or 'block_simple'):
        # @1. split sequence
        with tf.variable_scope('split_seq'):
            block_num = tf.cast(tf.ceil(tf.divide(tf.cast(sl, tf.float32), tf.cast(block_len, tf.float32))), tf.int32)
            comp_len = block_num * block_len - sl

            rep_tensor_comp = tf.concat([rep_tensor, tf.zeros([bs, comp_len, org_ivec], tf.float32)], 1)
            rep_mask_comp = tf.concat([rep_mask, tf.cast(tf.zeros([bs, comp_len], tf.int32), tf.bool)], 1)

            rep_tensor_split = tf.reshape(rep_tensor_comp, [bs, block_num, block_len, org_ivec])  # bs,bn,bl,d
            rep_mask_split = tf.reshape(rep_mask_comp, [bs, block_num, block_len])  # bs,bn,bl

            # non-linear
            rep_map = bn_dense_layer(rep_tensor_split, ivec, True, 0., 'bn_dense_map', activation,
                                     False, wd, keep_prob, is_train)  # bs,bn,bl,vec
            rep_map_tile = tf.tile(tf.expand_dims(rep_map, 2), [1, 1, block_len, 1, 1])  # bs,bn,bl,bl,vec
            # rep_map_dp = dropout(rep_map, keep_prob, is_train)
            bn = block_num
            bl = block_len

        with tf.variable_scope('self_attention'):
            # @2.self-attention in block
            # mask generation
            sl_indices = tf.range(block_len, dtype=tf.int32)
            sl_col, sl_row = tf.meshgrid(sl_indices, sl_indices)
            if direction == 'forward':
                direct_mask = tf.greater(sl_row, sl_col)  # bl,bl
            else:
                direct_mask = tf.greater(sl_col, sl_row)  # bl,bl
            direct_mask_tile = tf.tile(
                tf.expand_dims(tf.expand_dims(direct_mask, 0), 0), [bs, bn, 1, 1])  # bs,bn,bl,bl
            rep_mask_tile_1 = tf.tile(tf.expand_dims(rep_mask_split, 2), [1, 1, bl, 1])  # bs,bn,bl,bl
            rep_mask_tile_2 = tf.tile(tf.expand_dims(rep_mask_split, 3), [1, 1, 1, bl])  # bs,bn,bl,bl
            rep_mask_tile = tf.logical_and(rep_mask_tile_1, rep_mask_tile_2)
            attn_mask = tf.logical_and(direct_mask_tile, rep_mask_tile, name='attn_mask')  # bs,bn,bl,bl

            # attention
            f_bias = tf.get_variable('f_bias', [ivec], tf.float32, tf.constant_initializer(0.))
            dependent_head = linear(
                rep_map, 2 * ivec, False, 0., 'linear_dependent_head', False, wd, keep_prob, is_train)  # bs,bn,bl,2vec
            dependent, head = tf.split(dependent_head, 2, 3)
            dependent_etd = tf.expand_dims(dependent, 2)  # bs,bn,1,bl,vec
            head_etd = tf.expand_dims(head, 3)  # bs,bn,bl,1,vec
            logits = scaled_tanh(dependent_etd + head_etd + f_bias, 5.0)  # bs,bn,bl,bl,vec
            logits_masked = exp_mask_for_high_rank(logits, attn_mask)
            attn_score = tf.nn.softmax(logits_masked, 3)  # bs,bn,bl,bl,vec
            attn_score = mask_for_high_rank(attn_score, attn_mask)  # bs,bn,bl,bl,vec
            self_attn_result = tf.reduce_sum(attn_score * rep_map_tile, 3)  # bs,bn,bl,vec

        with tf.variable_scope('source2token_self_attn'):
            inter_block_logits = bn_dense_layer(self_attn_result, ivec, True, 0., 'bn_dense_map', 'linear',
                                                False, wd, keep_prob, is_train)  # bs,bn,bl,vec
            inter_block_logits_masked = exp_mask_for_high_rank(inter_block_logits, rep_mask_split)  # bs,bn,bl,vec
            inter_block_soft = tf.nn.softmax(inter_block_logits_masked, 2)  # bs,bn,bl,vec
            inter_block_attn_output = tf.reduce_sum(self_attn_result * inter_block_soft, 2)  # bs,bn,vec

        with tf.variable_scope('self_attn_inter_block'):
            inter_block_attn_output_mask = tf.cast(tf.ones([bs, bn], tf.int32), tf.bool)
            block_ct_res = directional_attention_with_dense(
                inter_block_attn_output, inter_block_attn_output_mask, direction, 'disa',
                keep_prob, is_train, wd, activation
            )  # [bs,bn,vec]

            block_ct_res_tile = tf.tile(tf.expand_dims(block_ct_res, 2), [1, 1, bl, 1])#[bs,bn,vec]->[bs,bn,bl,vec]

        with tf.variable_scope('combination'):
            # input:1.rep_map[bs,bn,bl,vec]; 2.self_attn_result[bs,bn,bl,vec]; 3.rnn_res_tile[bs,bn,bl,vec]
            rep_tensor_with_ct = tf.concat([rep_map, self_attn_result, block_ct_res_tile], -1)  # [bs,bn,bl,3vec]
            new_context_and_gate = linear(rep_tensor_with_ct, 2 * ivec, True, 0., 'linear_new_context_and_gate',
                                          False, wd, keep_prob, is_train)  # [bs,bn,bl,2vec]
            new_context, gate = tf.split(new_context_and_gate, 2, 3)  # bs,bn,bl,vec
            if activation == "relu":
                new_context_act = tf.nn.relu(new_context)
            elif activation == "elu":
                new_context_act = tf.nn.elu(new_context)
            elif activation == "linear":
                new_context_act = tf.identity(new_context)
            else:
                raise RuntimeError
            gate_sig = tf.nn.sigmoid(gate)
            combination_res = gate_sig * new_context_act + (1 - gate_sig) * rep_map  # bs,bn,bl,vec

        with tf.variable_scope('restore_original_length'):
            combination_res_reshape = tf.reshape(combination_res, [bs, bn * bl, ivec])  # bs,bn*bl,vec
            output = combination_res_reshape[:, :sl, :]
            return output


# multi-head attention
# https://github.com/Kyubyong/transformer/blob/master/modules.py#L167
def multi_head_attention_git(rep_tensor, rep_mask, num_heads=8, num_units=64,scope=None,
        is_train=None, keep_prob=1., wd=0.):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    causality = False
    with tf.variable_scope(scope or "multihead_attention"):
        # because of self-attention, queries and keys is equal to rep_tensor
        queries = rep_tensor
        keys = rep_tensor

        # Set the fall back option for num_units
        if num_units is None:  # hn
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = rep_mask  # tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)  # exp mask
        outputs = tf.where(key_masks, outputs, paddings)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = rep_mask # tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= tf.cast(query_masks, tf.float32)  # broadcasting. (N, T_q, C)

        # Dropouts
        # outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        outputs = dropout(outputs, keep_prob, is_train)

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        # outputs += queries

        # Normalize
        # outputs = normalize(outputs)  # (N, T_q, C)

    return outputs


def multi_head_attention(
        rep_tensor, rep_mask, head_num=8, hidden_units_num=64,scope=None,
        is_train=None, keep_prob=1., wd=0.):
    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape().as_list()[2]

    with tf.variable_scope(scope or 'multi_head_attention'):

        with tf.variable_scope('positional_encoding'):
            seq_idxs = tf.tile(tf.expand_dims(tf.range(sl), 1), [1, ivec])  # sl, ivec
            feature_idxs = tf.tile(tf.expand_dims(tf.range(ivec), 0), [sl, 1])  # sl, ivec
            pos_enc = tf.where(
                tf.equal(tf.mod(feature_idxs, 2), 0),
                tf.sin(tf.cast(seq_idxs, tf.float32) /
                       tf.pow(10000., 2.0 * tf.cast(feature_idxs, tf.float32) / (1.0 * ivec))),
                tf.cos(tf.cast(seq_idxs, tf.float32) /
                       tf.pow(10000., 2.0 * tf.cast(feature_idxs - 1, tf.float32) / (1.0 * ivec))),
            )
            rep_tensor_pos = mask_for_high_rank(rep_tensor + pos_enc, rep_mask)  # bs, sl, ivec


        with tf.variable_scope('multi_head_attention'):
            W = tf.get_variable('W', [3, head_num, ivec, hidden_units_num], tf.float32)
            rep_tile = tf.tile(
                tf.expand_dims(tf.expand_dims(rep_tensor_pos, 0), 0),
                [3, head_num, 1, 1, 1])  # 3,head_num,bs,sl,ivec
            rep_tile_reshape = tf.reshape(rep_tile, [3, head_num, bs * sl, ivec])  # head_num,bs*sl,ivec

            maps = tf.reshape( # 3,head_num,bs*sl,hn ->  3,head_num,bs,sl,hn
                tf.matmul(dropout(rep_tile_reshape, keep_prob, is_train), W),
                [3, head_num, bs, sl, hidden_units_num])
            Q_map, K_map, V_map = tf.split(maps, 3, 0)
            Q_map = tf.squeeze(Q_map, [0])  # head_num,bs,sl,hn
            K_map = tf.squeeze(K_map, [0])  # head_num,bs,sl,hn
            V_map = tf.squeeze(V_map, [0])  # head_num,bs,sl,hn

            # head_num,bs,sl,sl
            # similarity_mat = tf.reduce_sum(Q_map_tile * K_map_tile, -1) / math.sqrt(1. * hidden_units_num)
            similarity_mat = tf.matmul(
                Q_map, tf.transpose(K_map, [0,1,3,2])
            ) / math.sqrt(1. * hidden_units_num)

            # mask: bs,sl -> head_num,bs,sl
            multi_mask = tf.tile(tf.expand_dims(rep_mask, 0), [head_num, 1, 1])  # head_num,bs,sl
            multi_mask_tile_1 = tf.expand_dims(multi_mask, 2)  # head_num,bs,1,sl
            multi_mask_tile_2 = tf.expand_dims(multi_mask, 3)  # head_num,bs,sl,1
            multi_mask_tile = tf.logical_and(multi_mask_tile_1, multi_mask_tile_2)  # head_num,bs,sl,sl
            similarity_mat_masked = exp_mask(similarity_mat, multi_mask_tile)  # head_num,bs,sl,sl
            prob_dist = tf.nn.softmax(similarity_mat_masked)  # head_num,bs,sl,sl
            prob_dist_dp = dropout(prob_dist, keep_prob, is_train)

            attn_res = tf.matmul(prob_dist_dp, V_map)  # head_num,bs,sl,hn

            attn_res_tran = tf.transpose(attn_res, [1,2,0,3])
            output = tf.reshape(attn_res_tran, [bs, sl, head_num * hidden_units_num])

            if wd > 0.:
                add_reg_without_bias()

            return output


"""
This is the baseline layers of context fusion layers and sentence-encoding models
"""

def context_fusion_layers(
        rep_tensor, rep_mask, method, activation_function,
        scope=None, wd=0., is_train=None, keep_prob=1., **kwargs):
    method_name_list = [
        'lstm', 'gru', 'sru', 'sru_normal',  # rnn
        'multi_cnn', 'hrchy_cnn',
        'multi_head', 'multi_head_git', 'disa',
        'block'
    ]
    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]

    if 'hn' in kwargs.keys():
        hn = kwargs['hn']
    else:
        hn = None

    ivec = hn or rep_tensor.get_shape().as_list()[2]

    context_fusion_output = None
    with tf.variable_scope(scope or 'context_fusion_layers'):
        if method in ['lstm', 'gru', 'sru_normal', 'sru']:
            context_fusion_output = contextual_bi_rnn(
                rep_tensor, rep_mask, ivec, method,
                False, wd, keep_prob, is_train, 'ct_bi_%s' % method)
        elif method == 'multi_cnn':
            assert 2 * ivec % 3 == 0
            sub_hn = 2 * ivec // 3
            context_fusion_output = cnn_for_context_fusion(
                rep_tensor, rep_mask, (3,4,5), sub_hn, 'ct_cnn', is_train, keep_prob, wd)
        elif method == 'hrchy_cnn':
            context_fusion_output = hierarchical_cnn_res_gate(
                rep_tensor, rep_mask, 5, 3, ivec, 'hierarchical_cnn', is_train, keep_prob, wd)
        elif method == 'multi_head':
            assert 2 * ivec % 8 == 0
            sub_hn = 2 * ivec // 8
            context_fusion_output = multi_head_attention(
                rep_tensor, rep_mask, 8, 75, 'ct_multi_head', is_train, keep_prob, wd)
        elif method == 'multi_head_git':
            assert 2 * ivec % 8 == 0
            context_fusion_output = multi_head_attention_git(
                rep_tensor, rep_mask, 8, 2 * ivec, 'ct_multi_head', is_train, keep_prob, wd)
        elif method == 'disa':
            with tf.variable_scope('ct_disa'):
                disa_fw = directional_attention_with_dense(
                    rep_tensor, rep_mask,'forward', 'fw_disa',
                    keep_prob, is_train, wd, activation_function, hn=ivec)
                disa_bw = directional_attention_with_dense(
                    rep_tensor, rep_mask, 'backward', 'bw_disa',
                    keep_prob, is_train, wd, activation_function, hn=ivec)
                context_fusion_output = tf.concat([disa_fw, disa_bw], -1)
        elif method == 'block':
            if 'block_len' in kwargs.keys():
                block_len = kwargs['block_len']
            else:
                block_len = None
            if block_len is None:
                block_len = tf.cast(tf.ceil(tf.pow(tf.cast(2 * sl, tf.float32), 1.0 / 3)), tf.int32)
            context_fusion_output = bi_directional_simple_block_attention(
                rep_tensor, rep_mask, block_len, 'ct_block_attn',
                keep_prob, is_train, wd, activation_function, hn)
        else:
            raise RuntimeError

        return context_fusion_output


def sentence_encoding_models(
        rep_tensor, rep_mask, method, activation_function,
        scope=None, wd=0., is_train=None, keep_prob=1., **kwargs):
    method_name_list = [
        'cnn_kim',
        'no_ct',
        'lstm', 'gru', 'sru', 'sru_normal',  # rnn
        'multi_cnn', 'hrchy_cnn',
        'multi_head', 'multi_head_git', 'disa',
        'block'
    ]

    if 'hn' in kwargs.keys():
        hn = kwargs['hn']
    else:
        hn = None
    ivec = hn or rep_tensor.get_shape().as_list()[2]

    with tf.variable_scope(scope or 'sentence_encoding_models'):
        if method == 'cnn_kim':
            assert 2 * ivec % 3 == 0
            sub_hn = 2 * ivec // 3
            sent_encoding = cnn_for_sentence_encoding(
                rep_tensor, rep_mask, (3,4,5), sub_hn, 'sent_encoding_cnn_kim', is_train, keep_prob, wd)
        else:
            ct_rep = None
            if method == 'no_ct':
                ct_rep = bn_dense_layer(
                    rep_tensor, 2*ivec, True, 0., 'no_ct', activation_function, False, wd, keep_prob, is_train)
            else:
                ct_rep = context_fusion_layers(
                    rep_tensor, rep_mask, method, activation_function,
                    None, wd, is_train, keep_prob, **kwargs)

            sent_encoding = multi_dimensional_attention(
                ct_rep, rep_mask, 'multi_dim_attn_for_%s' % method,
                keep_prob, is_train, wd, activation_function)

        return sent_encoding



