import tensorflow as tf
import tensorflow.keras.layers as tkl


def ResBlock(shape, inp, batchnorm=True, filters=16, kernel_size=3, data_format='channels_last'):
    if batchnorm:
        bn_axis = -1
        if data_format != 'channels_last':
            bn_axis = 1

        res = tkl.BatchNormalization(axis=bn_axis)(inp)
        res = tkl.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu', data_format=data_format)(res)
        res = tkl.BatchNormalization(axis=bn_axis)(res)
    else:
        res = tkl.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu', data_format=data_format)(inp)

    res = tkl.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu', data_format=data_format)(res)
    res = tkl.Add()([inp, res])
    return res
    

def SubsBlock(shape, inp, batchnorm=True, filters=16, kernel_size=3, data_format='channels_last'):
    if batchnorm:
        bn_axis = -1
        if data_format != 'channels_last':
            bn_axis = 1

        subsample = tkl.BatchNormalization(axis=bn_axis)(inp)
        subsample = tkl.Conv2D(filters=filters, kernel_size=kernel_size, strides=2, padding='same', activation='relu', data_format=data_format)(subsample)
        subsample = tkl.BatchNormalization(axis=bn_axis)(subsample)
    else:
        subsample = tkl.Conv2D(filters=filters, kernel_size=kernel_size, strides=2, padding='same', activation='relu', data_format=data_format)(inp)

    subsample = tkl.Conv2D(filters=2*filters, kernel_size=kernel_size, padding='same', activation='relu', data_format=data_format)(subsample)
    return subsample

    
def UpsBlock(shape, inp, batchnorm=True, filters=16, kernel_size=3, data_format='channels_last'):
    if batchnorm:
        bn_axis = -1
        if data_format != 'channels_last':
            bn_axis = 1

        upsample = tkl.BatchNormalization(axis=bn_axis)(inp)
        upsample = tkl.Conv2DTranspose(filters=2*filters, kernel_size=kernel_size, strides=2, padding='same', activation='relu', data_format=data_format)(upsample)
        upsample = tkl.BatchNormalization(axis=bn_axis)(upsample)
    else:
        upsample = tkl.Conv2DTranspose(filters=2*filters, kernel_size=kernel_size, strides=2, padding='same', activation='relu', data_format=data_format)(inp)

    upsample = tkl.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu', data_format=data_format)(upsample)
    return upsample


def GridNet(shape,
            rows=2,
            cols=2,
            batchnorm=True,
            filters=16,
            kernel_size=3,
            filters_out=3,
            final_activation='relu',
            data_format='channels_last'):
    """Implementation of GridNet (https://arxiv.org/pdf/1707.07958.pdf) with rows x cols structure dimension"""
    assert cols % 2 == 0
    
    x = [[[] for i in range(cols)] for j in range(rows)]
    
    # left part
    inp = tkl.Input(shape=shape)
    x[0][0] = tkl.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu', data_format=data_format)(inp)

    for i in range(1, rows):
        x[i][0] = SubsBlock(shape, x[i-1][0], batchnorm=batchnorm, filters=filters*2**(i-1), kernel_size=kernel_size, data_format=data_format)

    for j in range(1, cols//2):
        x[0][j] = ResBlock(shape, x[0][j-1], batchnorm=batchnorm, filters=filters, kernel_size=kernel_size, data_format=data_format)
        for i in range(1, rows):
            x[i][j].append(ResBlock(shape, x[i][j-1], batchnorm=batchnorm, filters=filters*2**i, kernel_size=kernel_size, data_format=data_format))

        for i in range(1, rows):
            x[i][j].append(SubsBlock(shape, x[i-1][j], batchnorm=batchnorm, filters=filters*2**(i-1), kernel_size=kernel_size, data_format=data_format))
            x[i][j] = tkl.Add()(x[i][j])

    # center part
    k = cols // 2
    for i in range(rows-1):
        x[i][k].append(ResBlock(shape, x[i][k-1], batchnorm=batchnorm, filters=filters*2**i, kernel_size=kernel_size, data_format=data_format))
    x[rows-1][k] = ResBlock(shape, x[rows-1][k-1], batchnorm=batchnorm, filters=filters*2**(rows-1), kernel_size=kernel_size, data_format=data_format)

    # right part
    for j in range(cols//2, cols-1):
        for i in range(rows-1, 0, -1):
            x[i-1][j].append(UpsBlock(shape, x[i][j], batchnorm=batchnorm, filters=filters*2**(i-1), kernel_size=kernel_size, data_format=data_format))
            x[i-1][j] = tkl.Add()(x[i-1][j])

        x[rows-1][j+1] = ResBlock(shape, x[rows-1][j], batchnorm=batchnorm, filters=filters*2**(rows-1), kernel_size=kernel_size, data_format=data_format)
        for i in range(rows-1, 0, -1):
            x[i-1][j+1].append(ResBlock(shape, x[i-1][j], batchnorm=batchnorm, filters=filters*2**(i-1), kernel_size=kernel_size, data_format=data_format))

    for i in range(rows-1, 0, -1):
        x[i-1][cols-1].append(UpsBlock(shape, x[i][cols-1], batchnorm=batchnorm, filters=filters*2**(i-1), kernel_size=kernel_size, data_format=data_format))
        x[i-1][cols-1] = tkl.Add()(x[i-1][cols-1])
    
    outp = tkl.Conv2D(filters=filters_out, kernel_size=kernel_size, padding='same', activation=final_activation, data_format=data_format)(x[0][cols-1])
    model = tf.keras.Model(inputs=inp, outputs=outp, name=f"GridNet_{rows}x{cols}")

    return model
