"""AUTO-GENERATED frozen legacy->new weight mapping (ckpt-319992, 39 classes).

Each entry maps an exact legacy checkpoint key -> a stable canonical id of the
new variable (architecture position, NOT the Keras auto-name, so it is valid on
any TF/Keras build). Verified pair-by-pair: shapes match and same-shape siblings
map to distinct architecturally-correct sub-blocks. See
tools/legacy_checkpoint_structure.md for the derivation. Hand-edit a value here
to override a single pair.

Totals: backbone 135, decoder 90, head 111 = 336.
"""

LEGACY_TO_NEW = {
    # ================= BACKBONE (135) =================
    'backbone/layer_with_weights-0/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk0/-/kernel',  # (3, 3, 3, 32)  backbone/stem_conv1/conv2d/kernel
    'backbone/layer_with_weights-0/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk0/-/gamma',  # (32,)  backbone/stem_conv1/batch_normalization/gamma
    'backbone/layer_with_weights-0/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk0/-/beta',  # (32,)  backbone/stem_conv1/batch_normalization/beta
    'backbone/layer_with_weights-0/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk0/-/moving_mean',  # (32,)  backbone/stem_conv1/batch_normalization/moving_mean
    'backbone/layer_with_weights-0/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk0/-/moving_variance',  # (32,)  backbone/stem_conv1/batch_normalization/moving_variance
    'backbone/layer_with_weights-1/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk1/-/kernel',  # (3, 3, 32, 64)  backbone/stem_conv2/conv2d_1/kernel
    'backbone/layer_with_weights-1/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk1/-/gamma',  # (64,)  backbone/stem_conv2/batch_normalization_1/gamma
    'backbone/layer_with_weights-1/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk1/-/beta',  # (64,)  backbone/stem_conv2/batch_normalization_1/beta
    'backbone/layer_with_weights-1/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk1/-/moving_mean',  # (64,)  backbone/stem_conv2/batch_normalization_1/moving_mean
    'backbone/layer_with_weights-1/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk1/-/moving_variance',  # (64,)  backbone/stem_conv2/batch_normalization_1/moving_variance
    'backbone/layer_with_weights-2/_route/_conv2/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk2/cv1/kernel',  # (1, 1, 64, 64)  backbone/stem_c2f/cv1/conv2d_2/kernel
    'backbone/layer_with_weights-2/_route/_conv2/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk2/cv1/gamma',  # (64,)  backbone/stem_c2f/cv1/batch_normalization_2/gamma
    'backbone/layer_with_weights-2/_route/_conv2/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk2/cv1/beta',  # (64,)  backbone/stem_c2f/cv1/batch_normalization_2/beta
    'backbone/layer_with_weights-2/_route/_conv2/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk2/cv1/moving_mean',  # (64,)  backbone/stem_c2f/cv1/batch_normalization_2/moving_mean
    'backbone/layer_with_weights-2/_route/_conv2/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk2/cv1/moving_variance',  # (64,)  backbone/stem_c2f/cv1/batch_normalization_2/moving_variance
    'backbone/layer_with_weights-2/_connect/_conv1/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk2/cv2/kernel',  # (1, 1, 96, 64)  backbone/stem_c2f/cv2/conv2d_3/kernel
    'backbone/layer_with_weights-2/_connect/_conv1/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk2/cv2/gamma',  # (64,)  backbone/stem_c2f/cv2/batch_normalization_3/gamma
    'backbone/layer_with_weights-2/_connect/_conv1/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk2/cv2/beta',  # (64,)  backbone/stem_c2f/cv2/batch_normalization_3/beta
    'backbone/layer_with_weights-2/_connect/_conv1/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk2/cv2/moving_mean',  # (64,)  backbone/stem_c2f/cv2/batch_normalization_3/moving_mean
    'backbone/layer_with_weights-2/_connect/_conv1/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk2/cv2/moving_variance',  # (64,)  backbone/stem_c2f/cv2/batch_normalization_3/moving_variance
    'backbone/layer_with_weights-2/_model_to_wrap/0/_conv1/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk2/bn0/cv1/kernel',  # (3, 3, 32, 32)  backbone/stem_c2f/bn0/cv1/conv2d_4/kernel
    'backbone/layer_with_weights-2/_model_to_wrap/0/_conv1/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk2/bn0/cv1/gamma',  # (32,)  backbone/stem_c2f/bn0/cv1/batch_normalization_4/gamma
    'backbone/layer_with_weights-2/_model_to_wrap/0/_conv1/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk2/bn0/cv1/beta',  # (32,)  backbone/stem_c2f/bn0/cv1/batch_normalization_4/beta
    'backbone/layer_with_weights-2/_model_to_wrap/0/_conv1/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk2/bn0/cv1/moving_mean',  # (32,)  backbone/stem_c2f/bn0/cv1/batch_normalization_4/moving_mean
    'backbone/layer_with_weights-2/_model_to_wrap/0/_conv1/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk2/bn0/cv1/moving_variance',  # (32,)  backbone/stem_c2f/bn0/cv1/batch_normalization_4/moving_variance
    'backbone/layer_with_weights-2/_model_to_wrap/0/_conv2/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk2/bn0/cv2/kernel',  # (3, 3, 32, 32)  backbone/stem_c2f/bn0/cv2/conv2d_5/kernel
    'backbone/layer_with_weights-2/_model_to_wrap/0/_conv2/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk2/bn0/cv2/gamma',  # (32,)  backbone/stem_c2f/bn0/cv2/batch_normalization_5/gamma
    'backbone/layer_with_weights-2/_model_to_wrap/0/_conv2/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk2/bn0/cv2/beta',  # (32,)  backbone/stem_c2f/bn0/cv2/batch_normalization_5/beta
    'backbone/layer_with_weights-2/_model_to_wrap/0/_conv2/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk2/bn0/cv2/moving_mean',  # (32,)  backbone/stem_c2f/bn0/cv2/batch_normalization_5/moving_mean
    'backbone/layer_with_weights-2/_model_to_wrap/0/_conv2/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk2/bn0/cv2/moving_variance',  # (32,)  backbone/stem_c2f/bn0/cv2/batch_normalization_5/moving_variance
    'backbone/layer_with_weights-3/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk3/-/kernel',  # (3, 3, 64, 128)  backbone/down1/conv2d_6/kernel
    'backbone/layer_with_weights-3/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk3/-/gamma',  # (128,)  backbone/down1/batch_normalization_6/gamma
    'backbone/layer_with_weights-3/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk3/-/beta',  # (128,)  backbone/down1/batch_normalization_6/beta
    'backbone/layer_with_weights-3/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk3/-/moving_mean',  # (128,)  backbone/down1/batch_normalization_6/moving_mean
    'backbone/layer_with_weights-3/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk3/-/moving_variance',  # (128,)  backbone/down1/batch_normalization_6/moving_variance
    'backbone/layer_with_weights-4/_route/_conv2/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk4/cv1/kernel',  # (1, 1, 128, 128)  backbone/c2f_p3/cv1/conv2d_7/kernel
    'backbone/layer_with_weights-4/_route/_conv2/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk4/cv1/gamma',  # (128,)  backbone/c2f_p3/cv1/batch_normalization_7/gamma
    'backbone/layer_with_weights-4/_route/_conv2/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk4/cv1/beta',  # (128,)  backbone/c2f_p3/cv1/batch_normalization_7/beta
    'backbone/layer_with_weights-4/_route/_conv2/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk4/cv1/moving_mean',  # (128,)  backbone/c2f_p3/cv1/batch_normalization_7/moving_mean
    'backbone/layer_with_weights-4/_route/_conv2/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk4/cv1/moving_variance',  # (128,)  backbone/c2f_p3/cv1/batch_normalization_7/moving_variance
    'backbone/layer_with_weights-4/_connect/_conv1/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk4/cv2/kernel',  # (1, 1, 256, 128)  backbone/c2f_p3/cv2/conv2d_8/kernel
    'backbone/layer_with_weights-4/_connect/_conv1/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk4/cv2/gamma',  # (128,)  backbone/c2f_p3/cv2/batch_normalization_8/gamma
    'backbone/layer_with_weights-4/_connect/_conv1/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk4/cv2/beta',  # (128,)  backbone/c2f_p3/cv2/batch_normalization_8/beta
    'backbone/layer_with_weights-4/_connect/_conv1/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk4/cv2/moving_mean',  # (128,)  backbone/c2f_p3/cv2/batch_normalization_8/moving_mean
    'backbone/layer_with_weights-4/_connect/_conv1/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk4/cv2/moving_variance',  # (128,)  backbone/c2f_p3/cv2/batch_normalization_8/moving_variance
    'backbone/layer_with_weights-4/_model_to_wrap/0/_conv1/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk4/bn0/cv1/kernel',  # (3, 3, 64, 64)  backbone/c2f_p3/bn0/cv1/conv2d_9/kernel
    'backbone/layer_with_weights-4/_model_to_wrap/0/_conv1/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk4/bn0/cv1/gamma',  # (64,)  backbone/c2f_p3/bn0/cv1/batch_normalization_9/gamma
    'backbone/layer_with_weights-4/_model_to_wrap/0/_conv1/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk4/bn0/cv1/beta',  # (64,)  backbone/c2f_p3/bn0/cv1/batch_normalization_9/beta
    'backbone/layer_with_weights-4/_model_to_wrap/0/_conv1/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk4/bn0/cv1/moving_mean',  # (64,)  backbone/c2f_p3/bn0/cv1/batch_normalization_9/moving_mean
    'backbone/layer_with_weights-4/_model_to_wrap/0/_conv1/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk4/bn0/cv1/moving_variance',  # (64,)  backbone/c2f_p3/bn0/cv1/batch_normalization_9/moving_variance
    'backbone/layer_with_weights-4/_model_to_wrap/0/_conv2/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk4/bn0/cv2/kernel',  # (3, 3, 64, 64)  backbone/c2f_p3/bn0/cv2/conv2d_10/kernel
    'backbone/layer_with_weights-4/_model_to_wrap/0/_conv2/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk4/bn0/cv2/gamma',  # (64,)  backbone/c2f_p3/bn0/cv2/batch_normalization_10/gamma
    'backbone/layer_with_weights-4/_model_to_wrap/0/_conv2/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk4/bn0/cv2/beta',  # (64,)  backbone/c2f_p3/bn0/cv2/batch_normalization_10/beta
    'backbone/layer_with_weights-4/_model_to_wrap/0/_conv2/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk4/bn0/cv2/moving_mean',  # (64,)  backbone/c2f_p3/bn0/cv2/batch_normalization_10/moving_mean
    'backbone/layer_with_weights-4/_model_to_wrap/0/_conv2/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk4/bn0/cv2/moving_variance',  # (64,)  backbone/c2f_p3/bn0/cv2/batch_normalization_10/moving_variance
    'backbone/layer_with_weights-4/_model_to_wrap/1/_conv1/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk4/bn1/cv1/kernel',  # (3, 3, 64, 64)  backbone/c2f_p3/bn1/cv1/conv2d_11/kernel
    'backbone/layer_with_weights-4/_model_to_wrap/1/_conv1/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk4/bn1/cv1/gamma',  # (64,)  backbone/c2f_p3/bn1/cv1/batch_normalization_11/gamma
    'backbone/layer_with_weights-4/_model_to_wrap/1/_conv1/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk4/bn1/cv1/beta',  # (64,)  backbone/c2f_p3/bn1/cv1/batch_normalization_11/beta
    'backbone/layer_with_weights-4/_model_to_wrap/1/_conv1/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk4/bn1/cv1/moving_mean',  # (64,)  backbone/c2f_p3/bn1/cv1/batch_normalization_11/moving_mean
    'backbone/layer_with_weights-4/_model_to_wrap/1/_conv1/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk4/bn1/cv1/moving_variance',  # (64,)  backbone/c2f_p3/bn1/cv1/batch_normalization_11/moving_variance
    'backbone/layer_with_weights-4/_model_to_wrap/1/_conv2/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk4/bn1/cv2/kernel',  # (3, 3, 64, 64)  backbone/c2f_p3/bn1/cv2/conv2d_12/kernel
    'backbone/layer_with_weights-4/_model_to_wrap/1/_conv2/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk4/bn1/cv2/gamma',  # (64,)  backbone/c2f_p3/bn1/cv2/batch_normalization_12/gamma
    'backbone/layer_with_weights-4/_model_to_wrap/1/_conv2/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk4/bn1/cv2/beta',  # (64,)  backbone/c2f_p3/bn1/cv2/batch_normalization_12/beta
    'backbone/layer_with_weights-4/_model_to_wrap/1/_conv2/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk4/bn1/cv2/moving_mean',  # (64,)  backbone/c2f_p3/bn1/cv2/batch_normalization_12/moving_mean
    'backbone/layer_with_weights-4/_model_to_wrap/1/_conv2/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk4/bn1/cv2/moving_variance',  # (64,)  backbone/c2f_p3/bn1/cv2/batch_normalization_12/moving_variance
    'backbone/layer_with_weights-5/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk5/-/kernel',  # (3, 3, 128, 256)  backbone/down2/conv2d_13/kernel
    'backbone/layer_with_weights-5/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk5/-/gamma',  # (256,)  backbone/down2/batch_normalization_13/gamma
    'backbone/layer_with_weights-5/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk5/-/beta',  # (256,)  backbone/down2/batch_normalization_13/beta
    'backbone/layer_with_weights-5/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk5/-/moving_mean',  # (256,)  backbone/down2/batch_normalization_13/moving_mean
    'backbone/layer_with_weights-5/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk5/-/moving_variance',  # (256,)  backbone/down2/batch_normalization_13/moving_variance
    'backbone/layer_with_weights-6/_route/_conv2/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk6/cv1/kernel',  # (1, 1, 256, 256)  backbone/c2f_p4/cv1/conv2d_14/kernel
    'backbone/layer_with_weights-6/_route/_conv2/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk6/cv1/gamma',  # (256,)  backbone/c2f_p4/cv1/batch_normalization_14/gamma
    'backbone/layer_with_weights-6/_route/_conv2/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk6/cv1/beta',  # (256,)  backbone/c2f_p4/cv1/batch_normalization_14/beta
    'backbone/layer_with_weights-6/_route/_conv2/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk6/cv1/moving_mean',  # (256,)  backbone/c2f_p4/cv1/batch_normalization_14/moving_mean
    'backbone/layer_with_weights-6/_route/_conv2/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk6/cv1/moving_variance',  # (256,)  backbone/c2f_p4/cv1/batch_normalization_14/moving_variance
    'backbone/layer_with_weights-6/_connect/_conv1/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk6/cv2/kernel',  # (1, 1, 512, 256)  backbone/c2f_p4/cv2/conv2d_15/kernel
    'backbone/layer_with_weights-6/_connect/_conv1/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk6/cv2/gamma',  # (256,)  backbone/c2f_p4/cv2/batch_normalization_15/gamma
    'backbone/layer_with_weights-6/_connect/_conv1/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk6/cv2/beta',  # (256,)  backbone/c2f_p4/cv2/batch_normalization_15/beta
    'backbone/layer_with_weights-6/_connect/_conv1/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk6/cv2/moving_mean',  # (256,)  backbone/c2f_p4/cv2/batch_normalization_15/moving_mean
    'backbone/layer_with_weights-6/_connect/_conv1/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk6/cv2/moving_variance',  # (256,)  backbone/c2f_p4/cv2/batch_normalization_15/moving_variance
    'backbone/layer_with_weights-6/_model_to_wrap/0/_conv1/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk6/bn0/cv1/kernel',  # (3, 3, 128, 128)  backbone/c2f_p4/bn0/cv1/conv2d_16/kernel
    'backbone/layer_with_weights-6/_model_to_wrap/0/_conv1/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk6/bn0/cv1/gamma',  # (128,)  backbone/c2f_p4/bn0/cv1/batch_normalization_16/gamma
    'backbone/layer_with_weights-6/_model_to_wrap/0/_conv1/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk6/bn0/cv1/beta',  # (128,)  backbone/c2f_p4/bn0/cv1/batch_normalization_16/beta
    'backbone/layer_with_weights-6/_model_to_wrap/0/_conv1/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk6/bn0/cv1/moving_mean',  # (128,)  backbone/c2f_p4/bn0/cv1/batch_normalization_16/moving_mean
    'backbone/layer_with_weights-6/_model_to_wrap/0/_conv1/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk6/bn0/cv1/moving_variance',  # (128,)  backbone/c2f_p4/bn0/cv1/batch_normalization_16/moving_variance
    'backbone/layer_with_weights-6/_model_to_wrap/0/_conv2/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk6/bn0/cv2/kernel',  # (3, 3, 128, 128)  backbone/c2f_p4/bn0/cv2/conv2d_17/kernel
    'backbone/layer_with_weights-6/_model_to_wrap/0/_conv2/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk6/bn0/cv2/gamma',  # (128,)  backbone/c2f_p4/bn0/cv2/batch_normalization_17/gamma
    'backbone/layer_with_weights-6/_model_to_wrap/0/_conv2/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk6/bn0/cv2/beta',  # (128,)  backbone/c2f_p4/bn0/cv2/batch_normalization_17/beta
    'backbone/layer_with_weights-6/_model_to_wrap/0/_conv2/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk6/bn0/cv2/moving_mean',  # (128,)  backbone/c2f_p4/bn0/cv2/batch_normalization_17/moving_mean
    'backbone/layer_with_weights-6/_model_to_wrap/0/_conv2/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk6/bn0/cv2/moving_variance',  # (128,)  backbone/c2f_p4/bn0/cv2/batch_normalization_17/moving_variance
    'backbone/layer_with_weights-6/_model_to_wrap/1/_conv1/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk6/bn1/cv1/kernel',  # (3, 3, 128, 128)  backbone/c2f_p4/bn1/cv1/conv2d_18/kernel
    'backbone/layer_with_weights-6/_model_to_wrap/1/_conv1/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk6/bn1/cv1/gamma',  # (128,)  backbone/c2f_p4/bn1/cv1/batch_normalization_18/gamma
    'backbone/layer_with_weights-6/_model_to_wrap/1/_conv1/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk6/bn1/cv1/beta',  # (128,)  backbone/c2f_p4/bn1/cv1/batch_normalization_18/beta
    'backbone/layer_with_weights-6/_model_to_wrap/1/_conv1/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk6/bn1/cv1/moving_mean',  # (128,)  backbone/c2f_p4/bn1/cv1/batch_normalization_18/moving_mean
    'backbone/layer_with_weights-6/_model_to_wrap/1/_conv1/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk6/bn1/cv1/moving_variance',  # (128,)  backbone/c2f_p4/bn1/cv1/batch_normalization_18/moving_variance
    'backbone/layer_with_weights-6/_model_to_wrap/1/_conv2/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk6/bn1/cv2/kernel',  # (3, 3, 128, 128)  backbone/c2f_p4/bn1/cv2/conv2d_19/kernel
    'backbone/layer_with_weights-6/_model_to_wrap/1/_conv2/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk6/bn1/cv2/gamma',  # (128,)  backbone/c2f_p4/bn1/cv2/batch_normalization_19/gamma
    'backbone/layer_with_weights-6/_model_to_wrap/1/_conv2/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk6/bn1/cv2/beta',  # (128,)  backbone/c2f_p4/bn1/cv2/batch_normalization_19/beta
    'backbone/layer_with_weights-6/_model_to_wrap/1/_conv2/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk6/bn1/cv2/moving_mean',  # (128,)  backbone/c2f_p4/bn1/cv2/batch_normalization_19/moving_mean
    'backbone/layer_with_weights-6/_model_to_wrap/1/_conv2/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk6/bn1/cv2/moving_variance',  # (128,)  backbone/c2f_p4/bn1/cv2/batch_normalization_19/moving_variance
    'backbone/layer_with_weights-7/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk7/-/kernel',  # (3, 3, 256, 512)  backbone/down3/conv2d_20/kernel
    'backbone/layer_with_weights-7/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk7/-/gamma',  # (512,)  backbone/down3/batch_normalization_20/gamma
    'backbone/layer_with_weights-7/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk7/-/beta',  # (512,)  backbone/down3/batch_normalization_20/beta
    'backbone/layer_with_weights-7/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk7/-/moving_mean',  # (512,)  backbone/down3/batch_normalization_20/moving_mean
    'backbone/layer_with_weights-7/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk7/-/moving_variance',  # (512,)  backbone/down3/batch_normalization_20/moving_variance
    'backbone/layer_with_weights-8/_route/_conv2/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk8/cv1/kernel',  # (1, 1, 512, 512)  backbone/c2f_p5_pre/cv1/conv2d_21/kernel
    'backbone/layer_with_weights-8/_route/_conv2/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk8/cv1/gamma',  # (512,)  backbone/c2f_p5_pre/cv1/batch_normalization_21/gamma
    'backbone/layer_with_weights-8/_route/_conv2/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk8/cv1/beta',  # (512,)  backbone/c2f_p5_pre/cv1/batch_normalization_21/beta
    'backbone/layer_with_weights-8/_route/_conv2/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk8/cv1/moving_mean',  # (512,)  backbone/c2f_p5_pre/cv1/batch_normalization_21/moving_mean
    'backbone/layer_with_weights-8/_route/_conv2/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk8/cv1/moving_variance',  # (512,)  backbone/c2f_p5_pre/cv1/batch_normalization_21/moving_variance
    'backbone/layer_with_weights-8/_connect/_conv1/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk8/cv2/kernel',  # (1, 1, 768, 512)  backbone/c2f_p5_pre/cv2/conv2d_22/kernel
    'backbone/layer_with_weights-8/_connect/_conv1/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk8/cv2/gamma',  # (512,)  backbone/c2f_p5_pre/cv2/batch_normalization_22/gamma
    'backbone/layer_with_weights-8/_connect/_conv1/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk8/cv2/beta',  # (512,)  backbone/c2f_p5_pre/cv2/batch_normalization_22/beta
    'backbone/layer_with_weights-8/_connect/_conv1/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk8/cv2/moving_mean',  # (512,)  backbone/c2f_p5_pre/cv2/batch_normalization_22/moving_mean
    'backbone/layer_with_weights-8/_connect/_conv1/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk8/cv2/moving_variance',  # (512,)  backbone/c2f_p5_pre/cv2/batch_normalization_22/moving_variance
    'backbone/layer_with_weights-8/_model_to_wrap/0/_conv1/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk8/bn0/cv1/kernel',  # (3, 3, 256, 256)  backbone/c2f_p5_pre/bn0/cv1/conv2d_23/kernel
    'backbone/layer_with_weights-8/_model_to_wrap/0/_conv1/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk8/bn0/cv1/gamma',  # (256,)  backbone/c2f_p5_pre/bn0/cv1/batch_normalization_23/gamma
    'backbone/layer_with_weights-8/_model_to_wrap/0/_conv1/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk8/bn0/cv1/beta',  # (256,)  backbone/c2f_p5_pre/bn0/cv1/batch_normalization_23/beta
    'backbone/layer_with_weights-8/_model_to_wrap/0/_conv1/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk8/bn0/cv1/moving_mean',  # (256,)  backbone/c2f_p5_pre/bn0/cv1/batch_normalization_23/moving_mean
    'backbone/layer_with_weights-8/_model_to_wrap/0/_conv1/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk8/bn0/cv1/moving_variance',  # (256,)  backbone/c2f_p5_pre/bn0/cv1/batch_normalization_23/moving_variance
    'backbone/layer_with_weights-8/_model_to_wrap/0/_conv2/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk8/bn0/cv2/kernel',  # (3, 3, 256, 256)  backbone/c2f_p5_pre/bn0/cv2/conv2d_24/kernel
    'backbone/layer_with_weights-8/_model_to_wrap/0/_conv2/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk8/bn0/cv2/gamma',  # (256,)  backbone/c2f_p5_pre/bn0/cv2/batch_normalization_24/gamma
    'backbone/layer_with_weights-8/_model_to_wrap/0/_conv2/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk8/bn0/cv2/beta',  # (256,)  backbone/c2f_p5_pre/bn0/cv2/batch_normalization_24/beta
    'backbone/layer_with_weights-8/_model_to_wrap/0/_conv2/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk8/bn0/cv2/moving_mean',  # (256,)  backbone/c2f_p5_pre/bn0/cv2/batch_normalization_24/moving_mean
    'backbone/layer_with_weights-8/_model_to_wrap/0/_conv2/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk8/bn0/cv2/moving_variance',  # (256,)  backbone/c2f_p5_pre/bn0/cv2/batch_normalization_24/moving_variance
    'backbone/layer_with_weights-9/_conv1/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk9/cv1/kernel',  # (1, 1, 512, 256)  backbone/sppf/cv1/conv2d_25/kernel
    'backbone/layer_with_weights-9/_conv1/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk9/cv1/gamma',  # (256,)  backbone/sppf/cv1/batch_normalization_25/gamma
    'backbone/layer_with_weights-9/_conv1/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk9/cv1/beta',  # (256,)  backbone/sppf/cv1/batch_normalization_25/beta
    'backbone/layer_with_weights-9/_conv1/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk9/cv1/moving_mean',  # (256,)  backbone/sppf/cv1/batch_normalization_25/moving_mean
    'backbone/layer_with_weights-9/_conv1/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk9/cv1/moving_variance',  # (256,)  backbone/sppf/cv1/batch_normalization_25/moving_variance
    'backbone/layer_with_weights-9/_conv2/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk9/cv2/kernel',  # (1, 1, 1024, 512)  backbone/sppf/cv2/conv2d_26/kernel
    'backbone/layer_with_weights-9/_conv2/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk9/cv2/gamma',  # (512,)  backbone/sppf/cv2/batch_normalization_26/gamma
    'backbone/layer_with_weights-9/_conv2/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk9/cv2/beta',  # (512,)  backbone/sppf/cv2/batch_normalization_26/beta
    'backbone/layer_with_weights-9/_conv2/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk9/cv2/moving_mean',  # (512,)  backbone/sppf/cv2/batch_normalization_26/moving_mean
    'backbone/layer_with_weights-9/_conv2/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'backbone/blk9/cv2/moving_variance',  # (512,)  backbone/sppf/cv2/batch_normalization_26/moving_variance
    # ================= DECODER (90) =================
    'decoder/layer_with_weights-0/_route/_conv2/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk0/cv1/kernel',  # (1, 1, 768, 256)  decoder/fpn_c2f_p4/cv1/conv2d_27/kernel
    'decoder/layer_with_weights-0/_route/_conv2/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk0/cv1/gamma',  # (256,)  decoder/fpn_c2f_p4/cv1/batch_normalization_27/gamma
    'decoder/layer_with_weights-0/_route/_conv2/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk0/cv1/beta',  # (256,)  decoder/fpn_c2f_p4/cv1/batch_normalization_27/beta
    'decoder/layer_with_weights-0/_route/_conv2/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk0/cv1/moving_mean',  # (256,)  decoder/fpn_c2f_p4/cv1/batch_normalization_27/moving_mean
    'decoder/layer_with_weights-0/_route/_conv2/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk0/cv1/moving_variance',  # (256,)  decoder/fpn_c2f_p4/cv1/batch_normalization_27/moving_variance
    'decoder/layer_with_weights-0/_connect/_conv1/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk0/cv2/kernel',  # (1, 1, 384, 256)  decoder/fpn_c2f_p4/cv2/conv2d_28/kernel
    'decoder/layer_with_weights-0/_connect/_conv1/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk0/cv2/gamma',  # (256,)  decoder/fpn_c2f_p4/cv2/batch_normalization_28/gamma
    'decoder/layer_with_weights-0/_connect/_conv1/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk0/cv2/beta',  # (256,)  decoder/fpn_c2f_p4/cv2/batch_normalization_28/beta
    'decoder/layer_with_weights-0/_connect/_conv1/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk0/cv2/moving_mean',  # (256,)  decoder/fpn_c2f_p4/cv2/batch_normalization_28/moving_mean
    'decoder/layer_with_weights-0/_connect/_conv1/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk0/cv2/moving_variance',  # (256,)  decoder/fpn_c2f_p4/cv2/batch_normalization_28/moving_variance
    'decoder/layer_with_weights-0/_model_to_wrap/0/_conv1/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk0/bn0/cv1/kernel',  # (3, 3, 128, 128)  decoder/fpn_c2f_p4/bn0/cv1/conv2d_29/kernel
    'decoder/layer_with_weights-0/_model_to_wrap/0/_conv1/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk0/bn0/cv1/gamma',  # (128,)  decoder/fpn_c2f_p4/bn0/cv1/batch_normalization_29/gamma
    'decoder/layer_with_weights-0/_model_to_wrap/0/_conv1/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk0/bn0/cv1/beta',  # (128,)  decoder/fpn_c2f_p4/bn0/cv1/batch_normalization_29/beta
    'decoder/layer_with_weights-0/_model_to_wrap/0/_conv1/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk0/bn0/cv1/moving_mean',  # (128,)  decoder/fpn_c2f_p4/bn0/cv1/batch_normalization_29/moving_mean
    'decoder/layer_with_weights-0/_model_to_wrap/0/_conv1/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk0/bn0/cv1/moving_variance',  # (128,)  decoder/fpn_c2f_p4/bn0/cv1/batch_normalization_29/moving_variance
    'decoder/layer_with_weights-0/_model_to_wrap/0/_conv2/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk0/bn0/cv2/kernel',  # (3, 3, 128, 128)  decoder/fpn_c2f_p4/bn0/cv2/conv2d_30/kernel
    'decoder/layer_with_weights-0/_model_to_wrap/0/_conv2/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk0/bn0/cv2/gamma',  # (128,)  decoder/fpn_c2f_p4/bn0/cv2/batch_normalization_30/gamma
    'decoder/layer_with_weights-0/_model_to_wrap/0/_conv2/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk0/bn0/cv2/beta',  # (128,)  decoder/fpn_c2f_p4/bn0/cv2/batch_normalization_30/beta
    'decoder/layer_with_weights-0/_model_to_wrap/0/_conv2/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk0/bn0/cv2/moving_mean',  # (128,)  decoder/fpn_c2f_p4/bn0/cv2/batch_normalization_30/moving_mean
    'decoder/layer_with_weights-0/_model_to_wrap/0/_conv2/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk0/bn0/cv2/moving_variance',  # (128,)  decoder/fpn_c2f_p4/bn0/cv2/batch_normalization_30/moving_variance
    'decoder/layer_with_weights-1/_route/_conv2/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk1/cv1/kernel',  # (1, 1, 384, 128)  decoder/fpn_c2f_p3/cv1/conv2d_31/kernel
    'decoder/layer_with_weights-1/_route/_conv2/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk1/cv1/gamma',  # (128,)  decoder/fpn_c2f_p3/cv1/batch_normalization_31/gamma
    'decoder/layer_with_weights-1/_route/_conv2/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk1/cv1/beta',  # (128,)  decoder/fpn_c2f_p3/cv1/batch_normalization_31/beta
    'decoder/layer_with_weights-1/_route/_conv2/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk1/cv1/moving_mean',  # (128,)  decoder/fpn_c2f_p3/cv1/batch_normalization_31/moving_mean
    'decoder/layer_with_weights-1/_route/_conv2/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk1/cv1/moving_variance',  # (128,)  decoder/fpn_c2f_p3/cv1/batch_normalization_31/moving_variance
    'decoder/layer_with_weights-1/_connect/_conv1/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk1/cv2/kernel',  # (1, 1, 192, 128)  decoder/fpn_c2f_p3/cv2/conv2d_32/kernel
    'decoder/layer_with_weights-1/_connect/_conv1/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk1/cv2/gamma',  # (128,)  decoder/fpn_c2f_p3/cv2/batch_normalization_32/gamma
    'decoder/layer_with_weights-1/_connect/_conv1/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk1/cv2/beta',  # (128,)  decoder/fpn_c2f_p3/cv2/batch_normalization_32/beta
    'decoder/layer_with_weights-1/_connect/_conv1/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk1/cv2/moving_mean',  # (128,)  decoder/fpn_c2f_p3/cv2/batch_normalization_32/moving_mean
    'decoder/layer_with_weights-1/_connect/_conv1/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk1/cv2/moving_variance',  # (128,)  decoder/fpn_c2f_p3/cv2/batch_normalization_32/moving_variance
    'decoder/layer_with_weights-1/_model_to_wrap/0/_conv1/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk1/bn0/cv1/kernel',  # (3, 3, 64, 64)  decoder/fpn_c2f_p3/bn0/cv1/conv2d_33/kernel
    'decoder/layer_with_weights-1/_model_to_wrap/0/_conv1/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk1/bn0/cv1/gamma',  # (64,)  decoder/fpn_c2f_p3/bn0/cv1/batch_normalization_33/gamma
    'decoder/layer_with_weights-1/_model_to_wrap/0/_conv1/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk1/bn0/cv1/beta',  # (64,)  decoder/fpn_c2f_p3/bn0/cv1/batch_normalization_33/beta
    'decoder/layer_with_weights-1/_model_to_wrap/0/_conv1/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk1/bn0/cv1/moving_mean',  # (64,)  decoder/fpn_c2f_p3/bn0/cv1/batch_normalization_33/moving_mean
    'decoder/layer_with_weights-1/_model_to_wrap/0/_conv1/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk1/bn0/cv1/moving_variance',  # (64,)  decoder/fpn_c2f_p3/bn0/cv1/batch_normalization_33/moving_variance
    'decoder/layer_with_weights-1/_model_to_wrap/0/_conv2/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk1/bn0/cv2/kernel',  # (3, 3, 64, 64)  decoder/fpn_c2f_p3/bn0/cv2/conv2d_34/kernel
    'decoder/layer_with_weights-1/_model_to_wrap/0/_conv2/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk1/bn0/cv2/gamma',  # (64,)  decoder/fpn_c2f_p3/bn0/cv2/batch_normalization_34/gamma
    'decoder/layer_with_weights-1/_model_to_wrap/0/_conv2/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk1/bn0/cv2/beta',  # (64,)  decoder/fpn_c2f_p3/bn0/cv2/batch_normalization_34/beta
    'decoder/layer_with_weights-1/_model_to_wrap/0/_conv2/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk1/bn0/cv2/moving_mean',  # (64,)  decoder/fpn_c2f_p3/bn0/cv2/batch_normalization_34/moving_mean
    'decoder/layer_with_weights-1/_model_to_wrap/0/_conv2/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk1/bn0/cv2/moving_variance',  # (64,)  decoder/fpn_c2f_p3/bn0/cv2/batch_normalization_34/moving_variance
    'decoder/layer_with_weights-2/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk2/-/kernel',  # (3, 3, 128, 128)  decoder/pan_down_p3/conv2d_35/kernel
    'decoder/layer_with_weights-2/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk2/-/gamma',  # (128,)  decoder/pan_down_p3/batch_normalization_35/gamma
    'decoder/layer_with_weights-2/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk2/-/beta',  # (128,)  decoder/pan_down_p3/batch_normalization_35/beta
    'decoder/layer_with_weights-2/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk2/-/moving_mean',  # (128,)  decoder/pan_down_p3/batch_normalization_35/moving_mean
    'decoder/layer_with_weights-2/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk2/-/moving_variance',  # (128,)  decoder/pan_down_p3/batch_normalization_35/moving_variance
    'decoder/layer_with_weights-3/_route/_conv2/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk3/cv1/kernel',  # (1, 1, 384, 256)  decoder/pan_c2f_p4/cv1/conv2d_36/kernel
    'decoder/layer_with_weights-3/_route/_conv2/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk3/cv1/gamma',  # (256,)  decoder/pan_c2f_p4/cv1/batch_normalization_36/gamma
    'decoder/layer_with_weights-3/_route/_conv2/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk3/cv1/beta',  # (256,)  decoder/pan_c2f_p4/cv1/batch_normalization_36/beta
    'decoder/layer_with_weights-3/_route/_conv2/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk3/cv1/moving_mean',  # (256,)  decoder/pan_c2f_p4/cv1/batch_normalization_36/moving_mean
    'decoder/layer_with_weights-3/_route/_conv2/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk3/cv1/moving_variance',  # (256,)  decoder/pan_c2f_p4/cv1/batch_normalization_36/moving_variance
    'decoder/layer_with_weights-3/_connect/_conv1/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk3/cv2/kernel',  # (1, 1, 384, 256)  decoder/pan_c2f_p4/cv2/conv2d_37/kernel
    'decoder/layer_with_weights-3/_connect/_conv1/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk3/cv2/gamma',  # (256,)  decoder/pan_c2f_p4/cv2/batch_normalization_37/gamma
    'decoder/layer_with_weights-3/_connect/_conv1/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk3/cv2/beta',  # (256,)  decoder/pan_c2f_p4/cv2/batch_normalization_37/beta
    'decoder/layer_with_weights-3/_connect/_conv1/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk3/cv2/moving_mean',  # (256,)  decoder/pan_c2f_p4/cv2/batch_normalization_37/moving_mean
    'decoder/layer_with_weights-3/_connect/_conv1/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk3/cv2/moving_variance',  # (256,)  decoder/pan_c2f_p4/cv2/batch_normalization_37/moving_variance
    'decoder/layer_with_weights-3/_model_to_wrap/0/_conv1/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk3/bn0/cv1/kernel',  # (3, 3, 128, 128)  decoder/pan_c2f_p4/bn0/cv1/conv2d_38/kernel
    'decoder/layer_with_weights-3/_model_to_wrap/0/_conv1/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk3/bn0/cv1/gamma',  # (128,)  decoder/pan_c2f_p4/bn0/cv1/batch_normalization_38/gamma
    'decoder/layer_with_weights-3/_model_to_wrap/0/_conv1/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk3/bn0/cv1/beta',  # (128,)  decoder/pan_c2f_p4/bn0/cv1/batch_normalization_38/beta
    'decoder/layer_with_weights-3/_model_to_wrap/0/_conv1/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk3/bn0/cv1/moving_mean',  # (128,)  decoder/pan_c2f_p4/bn0/cv1/batch_normalization_38/moving_mean
    'decoder/layer_with_weights-3/_model_to_wrap/0/_conv1/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk3/bn0/cv1/moving_variance',  # (128,)  decoder/pan_c2f_p4/bn0/cv1/batch_normalization_38/moving_variance
    'decoder/layer_with_weights-3/_model_to_wrap/0/_conv2/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk3/bn0/cv2/kernel',  # (3, 3, 128, 128)  decoder/pan_c2f_p4/bn0/cv2/conv2d_39/kernel
    'decoder/layer_with_weights-3/_model_to_wrap/0/_conv2/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk3/bn0/cv2/gamma',  # (128,)  decoder/pan_c2f_p4/bn0/cv2/batch_normalization_39/gamma
    'decoder/layer_with_weights-3/_model_to_wrap/0/_conv2/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk3/bn0/cv2/beta',  # (128,)  decoder/pan_c2f_p4/bn0/cv2/batch_normalization_39/beta
    'decoder/layer_with_weights-3/_model_to_wrap/0/_conv2/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk3/bn0/cv2/moving_mean',  # (128,)  decoder/pan_c2f_p4/bn0/cv2/batch_normalization_39/moving_mean
    'decoder/layer_with_weights-3/_model_to_wrap/0/_conv2/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk3/bn0/cv2/moving_variance',  # (128,)  decoder/pan_c2f_p4/bn0/cv2/batch_normalization_39/moving_variance
    'decoder/layer_with_weights-4/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk4/-/kernel',  # (3, 3, 256, 256)  decoder/pan_down_p4/conv2d_40/kernel
    'decoder/layer_with_weights-4/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk4/-/gamma',  # (256,)  decoder/pan_down_p4/batch_normalization_40/gamma
    'decoder/layer_with_weights-4/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk4/-/beta',  # (256,)  decoder/pan_down_p4/batch_normalization_40/beta
    'decoder/layer_with_weights-4/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk4/-/moving_mean',  # (256,)  decoder/pan_down_p4/batch_normalization_40/moving_mean
    'decoder/layer_with_weights-4/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk4/-/moving_variance',  # (256,)  decoder/pan_down_p4/batch_normalization_40/moving_variance
    'decoder/layer_with_weights-5/_route/_conv2/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk5/cv1/kernel',  # (1, 1, 768, 512)  decoder/pan_c2f_p5/cv1/conv2d_41/kernel
    'decoder/layer_with_weights-5/_route/_conv2/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk5/cv1/gamma',  # (512,)  decoder/pan_c2f_p5/cv1/batch_normalization_41/gamma
    'decoder/layer_with_weights-5/_route/_conv2/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk5/cv1/beta',  # (512,)  decoder/pan_c2f_p5/cv1/batch_normalization_41/beta
    'decoder/layer_with_weights-5/_route/_conv2/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk5/cv1/moving_mean',  # (512,)  decoder/pan_c2f_p5/cv1/batch_normalization_41/moving_mean
    'decoder/layer_with_weights-5/_route/_conv2/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk5/cv1/moving_variance',  # (512,)  decoder/pan_c2f_p5/cv1/batch_normalization_41/moving_variance
    'decoder/layer_with_weights-5/_connect/_conv1/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk5/cv2/kernel',  # (1, 1, 768, 512)  decoder/pan_c2f_p5/cv2/conv2d_42/kernel
    'decoder/layer_with_weights-5/_connect/_conv1/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk5/cv2/gamma',  # (512,)  decoder/pan_c2f_p5/cv2/batch_normalization_42/gamma
    'decoder/layer_with_weights-5/_connect/_conv1/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk5/cv2/beta',  # (512,)  decoder/pan_c2f_p5/cv2/batch_normalization_42/beta
    'decoder/layer_with_weights-5/_connect/_conv1/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk5/cv2/moving_mean',  # (512,)  decoder/pan_c2f_p5/cv2/batch_normalization_42/moving_mean
    'decoder/layer_with_weights-5/_connect/_conv1/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk5/cv2/moving_variance',  # (512,)  decoder/pan_c2f_p5/cv2/batch_normalization_42/moving_variance
    'decoder/layer_with_weights-5/_model_to_wrap/0/_conv1/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk5/bn0/cv1/kernel',  # (3, 3, 256, 256)  decoder/pan_c2f_p5/bn0/cv1/conv2d_43/kernel
    'decoder/layer_with_weights-5/_model_to_wrap/0/_conv1/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk5/bn0/cv1/gamma',  # (256,)  decoder/pan_c2f_p5/bn0/cv1/batch_normalization_43/gamma
    'decoder/layer_with_weights-5/_model_to_wrap/0/_conv1/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk5/bn0/cv1/beta',  # (256,)  decoder/pan_c2f_p5/bn0/cv1/batch_normalization_43/beta
    'decoder/layer_with_weights-5/_model_to_wrap/0/_conv1/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk5/bn0/cv1/moving_mean',  # (256,)  decoder/pan_c2f_p5/bn0/cv1/batch_normalization_43/moving_mean
    'decoder/layer_with_weights-5/_model_to_wrap/0/_conv1/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk5/bn0/cv1/moving_variance',  # (256,)  decoder/pan_c2f_p5/bn0/cv1/batch_normalization_43/moving_variance
    'decoder/layer_with_weights-5/_model_to_wrap/0/_conv2/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk5/bn0/cv2/kernel',  # (3, 3, 256, 256)  decoder/pan_c2f_p5/bn0/cv2/conv2d_44/kernel
    'decoder/layer_with_weights-5/_model_to_wrap/0/_conv2/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk5/bn0/cv2/gamma',  # (256,)  decoder/pan_c2f_p5/bn0/cv2/batch_normalization_44/gamma
    'decoder/layer_with_weights-5/_model_to_wrap/0/_conv2/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk5/bn0/cv2/beta',  # (256,)  decoder/pan_c2f_p5/bn0/cv2/batch_normalization_44/beta
    'decoder/layer_with_weights-5/_model_to_wrap/0/_conv2/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk5/bn0/cv2/moving_mean',  # (256,)  decoder/pan_c2f_p5/bn0/cv2/batch_normalization_44/moving_mean
    'decoder/layer_with_weights-5/_model_to_wrap/0/_conv2/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'decoder/blk5/bn0/cv2/moving_variance',  # (256,)  decoder/pan_c2f_p5/bn0/cv2/batch_normalization_44/moving_variance
    # ================= HEAD (111) =================
    'head/_head/3/cv2feat/layer_with_weights-0/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'head/L3/cv2feat_s1/kernel',  # (3, 3, 128, 136)  head/__conv_bn_act/conv2d_45/kernel
    'head/_head/3/cv2feat/layer_with_weights-0/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'head/L3/cv2feat_s1/gamma',  # (136,)  head/__conv_bn_act/batch_normalization_45/gamma
    'head/_head/3/cv2feat/layer_with_weights-0/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'head/L3/cv2feat_s1/beta',  # (136,)  head/__conv_bn_act/batch_normalization_45/beta
    'head/_head/3/cv2feat/layer_with_weights-0/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'head/L3/cv2feat_s1/moving_mean',  # (136,)  head/__conv_bn_act/batch_normalization_45/moving_mean
    'head/_head/3/cv2feat/layer_with_weights-0/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'head/L3/cv2feat_s1/moving_variance',  # (136,)  head/__conv_bn_act/batch_normalization_45/moving_variance
    'head/_head/3/cv2feat/layer_with_weights-1/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'head/L3/cv2feat_s2/kernel',  # (3, 3, 136, 136)  head/__conv_bn_act_1/conv2d_46/kernel
    'head/_head/3/cv2feat/layer_with_weights-1/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'head/L3/cv2feat_s2/gamma',  # (136,)  head/__conv_bn_act_1/batch_normalization_46/gamma
    'head/_head/3/cv2feat/layer_with_weights-1/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'head/L3/cv2feat_s2/beta',  # (136,)  head/__conv_bn_act_1/batch_normalization_46/beta
    'head/_head/3/cv2feat/layer_with_weights-1/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'head/L3/cv2feat_s2/moving_mean',  # (136,)  head/__conv_bn_act_1/batch_normalization_46/moving_mean
    'head/_head/3/cv2feat/layer_with_weights-1/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'head/L3/cv2feat_s2/moving_variance',  # (136,)  head/__conv_bn_act_1/batch_normalization_46/moving_variance
    'head/_head/3/box/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'head/L3/box_pred/kernel',  # (1, 1, 136, 64)  head/box_pred_3/kernel
    'head/_head/3/box/conv/bias/.ATTRIBUTES/VARIABLE_VALUE': 'head/L3/box_pred/bias',  # (64,)  head/box_pred_3/bias
    'head/_head/3/cv3/layer_with_weights-0/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'head/L3/cls_s1/kernel',  # (3, 3, 128, 128)  head/__conv_bn_act_2/conv2d_47/kernel
    'head/_head/3/cv3/layer_with_weights-0/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'head/L3/cls_s1/gamma',  # (128,)  head/__conv_bn_act_2/batch_normalization_47/gamma
    'head/_head/3/cv3/layer_with_weights-0/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'head/L3/cls_s1/beta',  # (128,)  head/__conv_bn_act_2/batch_normalization_47/beta
    'head/_head/3/cv3/layer_with_weights-0/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'head/L3/cls_s1/moving_mean',  # (128,)  head/__conv_bn_act_2/batch_normalization_47/moving_mean
    'head/_head/3/cv3/layer_with_weights-0/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'head/L3/cls_s1/moving_variance',  # (128,)  head/__conv_bn_act_2/batch_normalization_47/moving_variance
    'head/_head/3/cv3/layer_with_weights-1/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'head/L3/cls_s2/kernel',  # (3, 3, 128, 128)  head/__conv_bn_act_3/conv2d_48/kernel
    'head/_head/3/cv3/layer_with_weights-1/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'head/L3/cls_s2/gamma',  # (128,)  head/__conv_bn_act_3/batch_normalization_48/gamma
    'head/_head/3/cv3/layer_with_weights-1/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'head/L3/cls_s2/beta',  # (128,)  head/__conv_bn_act_3/batch_normalization_48/beta
    'head/_head/3/cv3/layer_with_weights-1/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'head/L3/cls_s2/moving_mean',  # (128,)  head/__conv_bn_act_3/batch_normalization_48/moving_mean
    'head/_head/3/cv3/layer_with_weights-1/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'head/L3/cls_s2/moving_variance',  # (128,)  head/__conv_bn_act_3/batch_normalization_48/moving_variance
    'head/_head/3/cv3/layer_with_weights-2/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'head/L3/cls_pred/kernel',  # (1, 1, 128, 39)  head/cls_pred_3/kernel
    'head/_head/3/cv3/layer_with_weights-2/conv/bias/.ATTRIBUTES/VARIABLE_VALUE': 'head/L3/cls_pred/bias',  # (39,)  head/cls_pred_3/bias
    'head/_head/3/poly_angle/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'head/L3/pa_pred/kernel',  # (1, 1, 136, 24)  head/pa_pred_3/kernel
    'head/_head/3/poly_angle/conv/bias/.ATTRIBUTES/VARIABLE_VALUE': 'head/L3/pa_pred/bias',  # (24,)  head/pa_pred_3/bias
    'head/_head/3/poly_dist/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'head/L3/pd_pred/kernel',  # (1, 1, 136, 24)  head/pd_pred_3/kernel
    'head/_head/3/poly_dist/conv/bias/.ATTRIBUTES/VARIABLE_VALUE': 'head/L3/pd_pred/bias',  # (24,)  head/pd_pred_3/bias
    'head/_head/3/poly_conf/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'head/L3/pc_pred/kernel',  # (1, 1, 136, 24)  head/pc_pred_3/kernel
    'head/_head/3/poly_conf/conv/bias/.ATTRIBUTES/VARIABLE_VALUE': 'head/L3/pc_pred/bias',  # (24,)  head/pc_pred_3/bias
    'head/_head/3/cv4/layer_with_weights-0/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'head/L3/dist_s0/kernel',  # (3, 3, 128, 128)  head/__conv_bn_act_4/conv2d_49/kernel
    'head/_head/3/cv4/layer_with_weights-0/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'head/L3/dist_s0/gamma',  # (128,)  head/__conv_bn_act_4/batch_normalization_49/gamma
    'head/_head/3/cv4/layer_with_weights-0/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'head/L3/dist_s0/beta',  # (128,)  head/__conv_bn_act_4/batch_normalization_49/beta
    'head/_head/3/cv4/layer_with_weights-0/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'head/L3/dist_s0/moving_mean',  # (128,)  head/__conv_bn_act_4/batch_normalization_49/moving_mean
    'head/_head/3/cv4/layer_with_weights-0/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'head/L3/dist_s0/moving_variance',  # (128,)  head/__conv_bn_act_4/batch_normalization_49/moving_variance
    'head/_head/3/cv4/layer_with_weights-1/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'head/L3/dist_pred/kernel',  # (1, 1, 128, 1)  head/dist_pred_3/kernel
    'head/_head/3/cv4/layer_with_weights-1/conv/bias/.ATTRIBUTES/VARIABLE_VALUE': 'head/L3/dist_pred/bias',  # (1,)  head/dist_pred_3/bias
    'head/_head/4/cv2feat/layer_with_weights-0/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'head/L4/cv2feat_s1/kernel',  # (3, 3, 256, 136)  head/__conv_bn_act_5/conv2d_50/kernel
    'head/_head/4/cv2feat/layer_with_weights-0/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'head/L4/cv2feat_s1/gamma',  # (136,)  head/__conv_bn_act_5/batch_normalization_50/gamma
    'head/_head/4/cv2feat/layer_with_weights-0/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'head/L4/cv2feat_s1/beta',  # (136,)  head/__conv_bn_act_5/batch_normalization_50/beta
    'head/_head/4/cv2feat/layer_with_weights-0/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'head/L4/cv2feat_s1/moving_mean',  # (136,)  head/__conv_bn_act_5/batch_normalization_50/moving_mean
    'head/_head/4/cv2feat/layer_with_weights-0/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'head/L4/cv2feat_s1/moving_variance',  # (136,)  head/__conv_bn_act_5/batch_normalization_50/moving_variance
    'head/_head/4/cv2feat/layer_with_weights-1/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'head/L4/cv2feat_s2/kernel',  # (3, 3, 136, 136)  head/__conv_bn_act_6/conv2d_51/kernel
    'head/_head/4/cv2feat/layer_with_weights-1/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'head/L4/cv2feat_s2/gamma',  # (136,)  head/__conv_bn_act_6/batch_normalization_51/gamma
    'head/_head/4/cv2feat/layer_with_weights-1/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'head/L4/cv2feat_s2/beta',  # (136,)  head/__conv_bn_act_6/batch_normalization_51/beta
    'head/_head/4/cv2feat/layer_with_weights-1/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'head/L4/cv2feat_s2/moving_mean',  # (136,)  head/__conv_bn_act_6/batch_normalization_51/moving_mean
    'head/_head/4/cv2feat/layer_with_weights-1/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'head/L4/cv2feat_s2/moving_variance',  # (136,)  head/__conv_bn_act_6/batch_normalization_51/moving_variance
    'head/_head/4/box/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'head/L4/box_pred/kernel',  # (1, 1, 136, 64)  head/box_pred_4/kernel
    'head/_head/4/box/conv/bias/.ATTRIBUTES/VARIABLE_VALUE': 'head/L4/box_pred/bias',  # (64,)  head/box_pred_4/bias
    'head/_head/4/cv3/layer_with_weights-0/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'head/L4/cls_s1/kernel',  # (3, 3, 256, 128)  head/__conv_bn_act_7/conv2d_52/kernel
    'head/_head/4/cv3/layer_with_weights-0/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'head/L4/cls_s1/gamma',  # (128,)  head/__conv_bn_act_7/batch_normalization_52/gamma
    'head/_head/4/cv3/layer_with_weights-0/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'head/L4/cls_s1/beta',  # (128,)  head/__conv_bn_act_7/batch_normalization_52/beta
    'head/_head/4/cv3/layer_with_weights-0/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'head/L4/cls_s1/moving_mean',  # (128,)  head/__conv_bn_act_7/batch_normalization_52/moving_mean
    'head/_head/4/cv3/layer_with_weights-0/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'head/L4/cls_s1/moving_variance',  # (128,)  head/__conv_bn_act_7/batch_normalization_52/moving_variance
    'head/_head/4/cv3/layer_with_weights-1/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'head/L4/cls_s2/kernel',  # (3, 3, 128, 128)  head/__conv_bn_act_8/conv2d_53/kernel
    'head/_head/4/cv3/layer_with_weights-1/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'head/L4/cls_s2/gamma',  # (128,)  head/__conv_bn_act_8/batch_normalization_53/gamma
    'head/_head/4/cv3/layer_with_weights-1/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'head/L4/cls_s2/beta',  # (128,)  head/__conv_bn_act_8/batch_normalization_53/beta
    'head/_head/4/cv3/layer_with_weights-1/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'head/L4/cls_s2/moving_mean',  # (128,)  head/__conv_bn_act_8/batch_normalization_53/moving_mean
    'head/_head/4/cv3/layer_with_weights-1/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'head/L4/cls_s2/moving_variance',  # (128,)  head/__conv_bn_act_8/batch_normalization_53/moving_variance
    'head/_head/4/cv3/layer_with_weights-2/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'head/L4/cls_pred/kernel',  # (1, 1, 128, 39)  head/cls_pred_4/kernel
    'head/_head/4/cv3/layer_with_weights-2/conv/bias/.ATTRIBUTES/VARIABLE_VALUE': 'head/L4/cls_pred/bias',  # (39,)  head/cls_pred_4/bias
    'head/_head/4/poly_angle/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'head/L4/pa_pred/kernel',  # (1, 1, 136, 24)  head/pa_pred_4/kernel
    'head/_head/4/poly_angle/conv/bias/.ATTRIBUTES/VARIABLE_VALUE': 'head/L4/pa_pred/bias',  # (24,)  head/pa_pred_4/bias
    'head/_head/4/poly_dist/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'head/L4/pd_pred/kernel',  # (1, 1, 136, 24)  head/pd_pred_4/kernel
    'head/_head/4/poly_dist/conv/bias/.ATTRIBUTES/VARIABLE_VALUE': 'head/L4/pd_pred/bias',  # (24,)  head/pd_pred_4/bias
    'head/_head/4/poly_conf/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'head/L4/pc_pred/kernel',  # (1, 1, 136, 24)  head/pc_pred_4/kernel
    'head/_head/4/poly_conf/conv/bias/.ATTRIBUTES/VARIABLE_VALUE': 'head/L4/pc_pred/bias',  # (24,)  head/pc_pred_4/bias
    'head/_head/4/cv4/layer_with_weights-0/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'head/L4/dist_s0/kernel',  # (3, 3, 256, 128)  head/__conv_bn_act_9/conv2d_54/kernel
    'head/_head/4/cv4/layer_with_weights-0/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'head/L4/dist_s0/gamma',  # (128,)  head/__conv_bn_act_9/batch_normalization_54/gamma
    'head/_head/4/cv4/layer_with_weights-0/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'head/L4/dist_s0/beta',  # (128,)  head/__conv_bn_act_9/batch_normalization_54/beta
    'head/_head/4/cv4/layer_with_weights-0/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'head/L4/dist_s0/moving_mean',  # (128,)  head/__conv_bn_act_9/batch_normalization_54/moving_mean
    'head/_head/4/cv4/layer_with_weights-0/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'head/L4/dist_s0/moving_variance',  # (128,)  head/__conv_bn_act_9/batch_normalization_54/moving_variance
    'head/_head/4/cv4/layer_with_weights-1/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'head/L4/dist_pred/kernel',  # (1, 1, 128, 1)  head/dist_pred_4/kernel
    'head/_head/4/cv4/layer_with_weights-1/conv/bias/.ATTRIBUTES/VARIABLE_VALUE': 'head/L4/dist_pred/bias',  # (1,)  head/dist_pred_4/bias
    'head/_head/5/cv2feat/layer_with_weights-0/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'head/L5/cv2feat_s1/kernel',  # (3, 3, 512, 136)  head/__conv_bn_act_10/conv2d_55/kernel
    'head/_head/5/cv2feat/layer_with_weights-0/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'head/L5/cv2feat_s1/gamma',  # (136,)  head/__conv_bn_act_10/batch_normalization_55/gamma
    'head/_head/5/cv2feat/layer_with_weights-0/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'head/L5/cv2feat_s1/beta',  # (136,)  head/__conv_bn_act_10/batch_normalization_55/beta
    'head/_head/5/cv2feat/layer_with_weights-0/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'head/L5/cv2feat_s1/moving_mean',  # (136,)  head/__conv_bn_act_10/batch_normalization_55/moving_mean
    'head/_head/5/cv2feat/layer_with_weights-0/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'head/L5/cv2feat_s1/moving_variance',  # (136,)  head/__conv_bn_act_10/batch_normalization_55/moving_variance
    'head/_head/5/cv2feat/layer_with_weights-1/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'head/L5/cv2feat_s2/kernel',  # (3, 3, 136, 136)  head/__conv_bn_act_11/conv2d_56/kernel
    'head/_head/5/cv2feat/layer_with_weights-1/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'head/L5/cv2feat_s2/gamma',  # (136,)  head/__conv_bn_act_11/batch_normalization_56/gamma
    'head/_head/5/cv2feat/layer_with_weights-1/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'head/L5/cv2feat_s2/beta',  # (136,)  head/__conv_bn_act_11/batch_normalization_56/beta
    'head/_head/5/cv2feat/layer_with_weights-1/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'head/L5/cv2feat_s2/moving_mean',  # (136,)  head/__conv_bn_act_11/batch_normalization_56/moving_mean
    'head/_head/5/cv2feat/layer_with_weights-1/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'head/L5/cv2feat_s2/moving_variance',  # (136,)  head/__conv_bn_act_11/batch_normalization_56/moving_variance
    'head/_head/5/box/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'head/L5/box_pred/kernel',  # (1, 1, 136, 64)  head/box_pred_5/kernel
    'head/_head/5/box/conv/bias/.ATTRIBUTES/VARIABLE_VALUE': 'head/L5/box_pred/bias',  # (64,)  head/box_pred_5/bias
    'head/_head/5/cv3/layer_with_weights-0/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'head/L5/cls_s1/kernel',  # (3, 3, 512, 128)  head/__conv_bn_act_12/conv2d_57/kernel
    'head/_head/5/cv3/layer_with_weights-0/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'head/L5/cls_s1/gamma',  # (128,)  head/__conv_bn_act_12/batch_normalization_57/gamma
    'head/_head/5/cv3/layer_with_weights-0/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'head/L5/cls_s1/beta',  # (128,)  head/__conv_bn_act_12/batch_normalization_57/beta
    'head/_head/5/cv3/layer_with_weights-0/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'head/L5/cls_s1/moving_mean',  # (128,)  head/__conv_bn_act_12/batch_normalization_57/moving_mean
    'head/_head/5/cv3/layer_with_weights-0/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'head/L5/cls_s1/moving_variance',  # (128,)  head/__conv_bn_act_12/batch_normalization_57/moving_variance
    'head/_head/5/cv3/layer_with_weights-1/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'head/L5/cls_s2/kernel',  # (3, 3, 128, 128)  head/__conv_bn_act_13/conv2d_58/kernel
    'head/_head/5/cv3/layer_with_weights-1/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'head/L5/cls_s2/gamma',  # (128,)  head/__conv_bn_act_13/batch_normalization_58/gamma
    'head/_head/5/cv3/layer_with_weights-1/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'head/L5/cls_s2/beta',  # (128,)  head/__conv_bn_act_13/batch_normalization_58/beta
    'head/_head/5/cv3/layer_with_weights-1/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'head/L5/cls_s2/moving_mean',  # (128,)  head/__conv_bn_act_13/batch_normalization_58/moving_mean
    'head/_head/5/cv3/layer_with_weights-1/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'head/L5/cls_s2/moving_variance',  # (128,)  head/__conv_bn_act_13/batch_normalization_58/moving_variance
    'head/_head/5/cv3/layer_with_weights-2/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'head/L5/cls_pred/kernel',  # (1, 1, 128, 39)  head/cls_pred_5/kernel
    'head/_head/5/cv3/layer_with_weights-2/conv/bias/.ATTRIBUTES/VARIABLE_VALUE': 'head/L5/cls_pred/bias',  # (39,)  head/cls_pred_5/bias
    'head/_head/5/poly_angle/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'head/L5/pa_pred/kernel',  # (1, 1, 136, 24)  head/pa_pred_5/kernel
    'head/_head/5/poly_angle/conv/bias/.ATTRIBUTES/VARIABLE_VALUE': 'head/L5/pa_pred/bias',  # (24,)  head/pa_pred_5/bias
    'head/_head/5/poly_dist/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'head/L5/pd_pred/kernel',  # (1, 1, 136, 24)  head/pd_pred_5/kernel
    'head/_head/5/poly_dist/conv/bias/.ATTRIBUTES/VARIABLE_VALUE': 'head/L5/pd_pred/bias',  # (24,)  head/pd_pred_5/bias
    'head/_head/5/poly_conf/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'head/L5/pc_pred/kernel',  # (1, 1, 136, 24)  head/pc_pred_5/kernel
    'head/_head/5/poly_conf/conv/bias/.ATTRIBUTES/VARIABLE_VALUE': 'head/L5/pc_pred/bias',  # (24,)  head/pc_pred_5/bias
    'head/_head/5/cv4/layer_with_weights-0/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'head/L5/dist_s0/kernel',  # (3, 3, 512, 128)  head/__conv_bn_act_14/conv2d_59/kernel
    'head/_head/5/cv4/layer_with_weights-0/bn/gamma/.ATTRIBUTES/VARIABLE_VALUE': 'head/L5/dist_s0/gamma',  # (128,)  head/__conv_bn_act_14/batch_normalization_59/gamma
    'head/_head/5/cv4/layer_with_weights-0/bn/beta/.ATTRIBUTES/VARIABLE_VALUE': 'head/L5/dist_s0/beta',  # (128,)  head/__conv_bn_act_14/batch_normalization_59/beta
    'head/_head/5/cv4/layer_with_weights-0/bn/moving_mean/.ATTRIBUTES/VARIABLE_VALUE': 'head/L5/dist_s0/moving_mean',  # (128,)  head/__conv_bn_act_14/batch_normalization_59/moving_mean
    'head/_head/5/cv4/layer_with_weights-0/bn/moving_variance/.ATTRIBUTES/VARIABLE_VALUE': 'head/L5/dist_s0/moving_variance',  # (128,)  head/__conv_bn_act_14/batch_normalization_59/moving_variance
    'head/_head/5/cv4/layer_with_weights-1/conv/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'head/L5/dist_pred/kernel',  # (1, 1, 128, 1)  head/dist_pred_5/kernel
    'head/_head/5/cv4/layer_with_weights-1/conv/bias/.ATTRIBUTES/VARIABLE_VALUE': 'head/L5/dist_pred/bias',  # (1,)  head/dist_pred_5/bias
}
