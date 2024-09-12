# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


@tf.function
def JointsMSE_Loss1(y_pred, target, target_weight):
    """

    :param y_pred:
    :param target:
    :param target_weight: <class 'tuple'>: (16, 10, 1)
    :return:
    """
    batch_size = y_pred.shape[0]
    num_of_joints = y_pred.shape[-1]
    pred = tf.reshape(tensor=y_pred, shape=(batch_size, -1, num_of_joints))
    heatmap_pred_list = tf.split(value=pred, num_or_size_splits=num_of_joints, axis=-1)
    gt = tf.reshape(tensor=target, shape=(batch_size, -1, num_of_joints))
    heatmap_gt_list = tf.split(value=gt, num_or_size_splits=num_of_joints, axis=-1)
    loss = 0.0
    for i in range(num_of_joints):
        heatmap_pred = tf.squeeze(heatmap_pred_list[i])
        heatmap_gt = tf.squeeze(heatmap_gt_list[i])
        loss += 0.5 * tf.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)(
            y_true=heatmap_pred * target_weight[:, i],
            y_pred=heatmap_gt * target_weight[:, i]
        )
    loss = tf.reduce_sum(loss) * (1. / batch_size)
    return loss / num_of_joints


@tf.function
def JointsMSE_Loss2(output, target, target_weight, use_target_weight=True):
    """
    torch.Size([16, 10, 1])
    :param output:
    :param target:
    :param target_weight:torch.Size([16, 10, 1])
    :return:
    """
    batch_size = output.shape[0]
    num_joints = output.shape[-1]
    heatmaps_pred = tf.reshape(tensor=output, shape=(batch_size, -1, num_joints))
    heatmaps_gt = tf.reshape(tensor=target, shape=(batch_size, -1, num_joints))
    heatmaps_pred = tf.split(value=heatmaps_pred, num_or_size_splits=num_joints, axis=-1)
    heatmaps_gt = tf.split(value=heatmaps_gt, num_or_size_splits=num_joints, axis=-1)

    loss = 0
    for idx in range(num_joints):
        heatmap_pred = heatmaps_pred[idx]  # (16, 3072, 1)
        heatmap_gt = heatmaps_gt[idx]
        heatmap_pred = tf.squeeze(heatmap_pred)  # (16, 3072)
        heatmap_gt = tf.squeeze(heatmap_gt)
        if use_target_weight:
            l = 0.5 * tf.keras.losses.MSE(
                heatmap_pred * target_weight[:, idx],
                heatmap_gt * target_weight[:, idx]
            )
            loss += tf.reduce_mean(l)
        else:
            l = 0.5 * tf.keras.losses.MSE(heatmap_pred, heatmap_gt)
            loss += tf.reduce_mean(l)
    return loss / num_joints


class JointsMSELoss():
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = tf.keras.losses.MSE
        self.use_target_weight = use_target_weight

    def __call__(self, output, target, target_weight):
        """
        torch.Size([16, 10, 1])
        :param output:
        :param target:
        :param target_weight:torch.Size([16, 10, 1])
        :return:
        """
        batch_size = output.shape[0]
        num_joints = output.shape[-1]
        heatmaps_pred = tf.reshape(tensor=output, shape=(batch_size, -1, num_joints))
        heatmaps_gt = tf.reshape(tensor=target, shape=(batch_size, -1, num_joints))
        heatmaps_pred = tf.split(value=heatmaps_pred, num_or_size_splits=num_joints, axis=-1)
        heatmaps_gt = tf.split(value=heatmaps_gt, num_or_size_splits=num_joints, axis=-1)

        loss = 0
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx]  # (16, 3072, 1)
            heatmap_gt = heatmaps_gt[idx]
            heatmap_pred = tf.squeeze(heatmap_pred)  # (16, 3072)
            heatmap_gt = tf.squeeze(heatmap_gt)
            if self.use_target_weight:
                l = 0.5 * self.criterion(
                    heatmap_pred * target_weight[:, idx],
                    heatmap_gt * target_weight[:, idx]
                )
                loss += tf.reduce_mean(l)

            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
        return loss / num_joints
