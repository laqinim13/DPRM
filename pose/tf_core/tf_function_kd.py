# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import os

import numpy as np
import torch

from pose.config.config import get_model_name
from pose.core.evaluate import accuracy
from pose.core.inference import get_final_preds
import tensorflow as tf

logger = logging.getLogger(__name__)


def get_model_feature_bk(model, input):
    with torch.no_grad():
        output = model(input)  # torch.Size([16, 10, 64, 48])
        layers_features = model.get_layers_features()
    for k, v in layers_features.items():
        layers_features[k] = v.detach().cpu().numpy().transpose(0, 2, 3, 1)
    return layers_features


def get_model_feature(model, input, kd_keys):
    layers_features = {}
    with torch.no_grad():
        for name, m in model.items():
            output = m(input)  # torch.Size([16, 10, 64, 48])
            layers_feature = m.get_layers_features()
            for k, v in layers_feature.items():
                if k in kd_keys:
                    layers_feature[k] = v.detach().cpu().numpy().transpose(0, 2, 3, 1)
                    layers_features[name] = layers_feature
    return layers_features


# @tf.function
def train_step(model, teacher_features, criterion, kd_criterion, optimizer, images, target, target_weight, kd_keys):
    with tf.GradientTape() as tape:
        # forward_tic = time.time()
        output = model(images, training=True)  # (16, 64, 48, 10)
        # forward_toc = time.time()
        # hard loss
        loss_h = criterion(output, target, target_weight, use_target_weight=True)
        # loss_h = criterion(output, target, target_weight)
        student_features = model.get_layers_features()
        # soft loss
        loss_s={}
        for name, model_feature in teacher_features.items():
            # soft loss
            kd_loss = {}
            if "backbone" in kd_keys:
                backbone_loss = kd_criterion(output, model_feature["backbone"], target_weight, use_target_weight=False)
                kd_loss["backbone"] = backbone_loss * kd_keys["backbone"]
            if "deconv_layers" in kd_keys:
                deconv_layers_loss = kd_criterion(student_features["deconv_layers"],
                                                  model_feature["deconv_layers"],
                                                  target_weight,
                                                  use_target_weight=False)
                kd_loss["deconv_layers"] = deconv_layers_loss * kd_keys["deconv_layers"]
            if "final_layer" in kd_keys:
                final_layer_loss = kd_criterion(student_features["final_layer"],
                                                model_feature["final_layer"],
                                                target_weight,
                                                use_target_weight=False)
                kd_loss["final_layer"] = final_layer_loss * kd_keys["final_layer"]
            loss_s[name] = sum(kd_loss.values())
        loss = loss_h + sum(loss_s.values())
    # bp_tic = time.time()
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

    return loss, loss_h, loss_s, output


@tf.function
def val_step(model, criterion, images, target, target_weight):
    # forward_tic = time.time()
    y_pred = model(images, training=False)
    # forward_toc = time.time()
    loss_value = criterion(y_pred, target, target_weight)
    return loss_value, y_pred


def train(config, train_loader, model, teacher_model, criterion, kd_criterion, kd_weight,optimizer, epoch,
          output_dir, tb_log_dir, writer_dict, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    # model.train()
    logger.info("kd_keys:{}".format(kd_weight))
    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        teacher_features = get_model_feature(teacher_model, input.to(device), kd_weight)
        # measure data loading time
        data_time.update(time.time() - end)
        input = input.numpy().transpose(0, 2, 3, 1)  # <class 'tuple'>: (16, 256, 192, 3)
        target = target.numpy().transpose(0, 2, 3, 1)  # <class 'tuple'>: (16, 64, 48, 10)
        target_weight = target_weight.numpy()  # <class 'tuple'>: (16, 10, 1)

        # compute output
        # output = model(input)
        loss, loss_h, kd_loss, output = train_step(model, teacher_features, criterion, kd_criterion, optimizer,
                                                   input, target, target_weight, kd_weight)
        # target_weight = target_weight.cuda(non_blocking=True)
        # measure accuracy and record loss

        batch_size = input.shape[0]
        losses.update(loss.numpy(), batch_size)

        output = output.numpy().transpose(0, 3, 1, 2)
        target = target.transpose(0, 3, 1, 2)

        _, avg_acc, cnt, pred = accuracy(output, target)
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            lr = tf.keras.backend.get_value(optimizer.lr)
            loss_info = ["{}:{:.5f}".format(k, v) for k, v in kd_loss.items()]
            loss_info += ["{}:{:.5f}".format("loss_h", loss_h)]
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.avg:.5f} ({loss.val:.5f} {kd_loss})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t' \
                  'lr:{lr:.7f}'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                speed=batch_size / batch_time.val,
                data_time=data_time, loss=losses, kd_loss=loss_info, acc=acc, lr=lr)
            logger.info(msg)

            # writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            # writer.add_scalar('train_loss', losses.val, global_steps)
            # writer.add_scalar('train_acc', acc.val, global_steps)
            # writer.add_scalar('lr_global_steps', lr, global_steps)

            writer_dict['train_global_steps'] = global_steps + 1
            prefix = os.path.join(output_dir, "images", 'train', str(i))
            # save_debug_images(config, input, meta, target, pred * 4, output, prefix)
    writer = writer_dict['writer']
    writer.add_scalar('train_loss', losses.avg, epoch)
    writer.add_scalar('train_acc', acc.avg, epoch)


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    # model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3), dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    image_idx = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            # input = input.to(device)
            # target = target.to(device)
            # target_weight = target_weight.to(device)

            input = input.numpy().transpose(0, 2, 3, 1)
            target = target.numpy().transpose(0, 2, 3, 1)
            target_weight = target_weight.numpy()
            # output = model(input)  # torch.Size([32, 3, 256, 192])->torch.Size([32, 17, 64, 48])
            loss, output = val_step(model, criterion, input, target, target_weight)
            batch_size = input.shape[0]
            output = output.numpy().transpose(0, 3, 1, 2)
            target = target.transpose(0, 3, 1, 2)
            # if config.TEST.FLIP_TEST:
            #     # this part is ugly, because pytorch has not supported negative index
            #     # input_flipped = model(input[:, :, :, ::-1])
            #     input_flipped = np.flip(input.cpu().numpy(), 3).copy()
            #     # input_flipped = torch.from_numpy(input_flipped).cuda()
            #     input_flipped = torch.from_numpy(input_flipped).to(device)
            #     output_flipped = model(input_flipped)
            #     output_flipped = flip_back(output_flipped.cpu().numpy(),
            #                                val_dataset.flip_pairs)
            #     # output_flipped = torch.from_numpy(output_flipped.copy()).cuda()
            #     output_flipped = torch.from_numpy(output_flipped.copy()).to(device)
            #
            #     # feature is not aligned, shift flipped heatmap for higher accuracy
            #     if config.TEST.SHIFT_HEATMAP:
            #         output_flipped[:, :, :, 1:] = \
            #             output_flipped.clone()[:, :, :, 0:-1]
            #         # output_flipped[:, :, :, 0] = 0
            #
            #     output = (output + output_flipped) * 0.5

            # target = target.cuda(non_blocking=True)
            # target_weight = target_weight.cuda(non_blocking=True)

            # loss = criterion(output, target, target_weight)

            num_images = input.shape[0]
            # measure accuracy and record loss
            losses.update(loss.numpy(), num_images)
            _, avg_acc, cnt, pred = accuracy(output, target)

            acc.update(avg_acc, cnt)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()
            image_id = meta['image_id'].numpy().tolist()

            preds, maxvals = get_final_preds(config, output, c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s * 200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])
            image_idx.extend(image_id)
            if config.DATASET.DATASET == 'posetrack':
                filenames.extend(meta['filename'])
                imgnums.extend(meta['imgnum'].numpy())

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time,
                    loss=losses, acc=acc)
                logger.info(msg)

                # prefix = '{}_{}'.format(os.path.join(output_dir, 'val'), i)
                prefix = os.path.join(output_dir, "images", 'val', str(i))
                # save_debug_images(config, input, meta, target, pred * 4, output, prefix)

        name_values, ap = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums, image_idx=image_idx)

        _, full_arch_name = get_model_name(config)
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, full_arch_name)
        else:
            _print_name_value(name_values, full_arch_name)

        if writer_dict:
            writer = writer_dict['writer']
            # global_steps = writer_dict['valid_global_steps']
            writer.add_scalar('valid_loss', losses.avg, epoch)
            writer.add_scalar('valid_acc', acc.avg, epoch)
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars('valid', dict(name_value), epoch)
            else:
                name_values = {n: v for n, v in dict(name_values).items() if v >= 0}
                # name_value = dict(name_value)
                writer.add_scalars('valid', name_values, epoch)
            writer_dict['valid_global_steps'] = epoch

    return ap


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values + 1) + '|')
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
