def quantize_tensor(x, num_bits=16):
    """Quantize a tensor using num_bits encoded integers"""
    qmin = 0.0
    qmax = 2.0 ** num_bits - 1.0
    min_val, max_val = x.min(), x.max()

    scale = (max_val - min_val) / (qmax - qmin)

    initial_zero_point = qmin - min_val / scale

    zero_point = 0
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)
    q_x = zero_point + x / scale
    # clip
    q_x.clamp_(qmin, qmax).round_()
    # quantize
    q_x = q_x.round()
    return q_x, scale, zero_point


def dequantize_tensor(q_x, scale, zero_point):
    """Dequantize a tensor using the quantization results"""
    return scale * (q_x.float() - zero_point)
