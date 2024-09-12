# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: tf-yolov3-detection
# @Author : Pan
# @E-mail : 390737991@qq.com
# @Date   : 2020-09-29 18:07:54
# --------------------------------------------------------
"""

import tensorflow as tf


class MultiStepLR(tf.keras.callbacks.Callback):
    def __init__(self, lr_stages, init_lr=0.1, steps_per_epoch=0, warmup_epoch=0,
                 decay_rates=None, log_freq=None, multi_gpu=False):
        '''
        warmup_epoch = 8
        steps_per_epoch = num_images / self.batch_size
        num_batch_warm_up = steps_per_epoch * warmup_epoch
        :param lr_stages:
        :param init_lr:
        :param steps_per_epoch:
        :param warmup_epoch:
        :param decay_rates:
        :param log_freq:
        '''
        super().__init__()
        self.multi_gpu = multi_gpu
        if decay_rates:
            lr_list = [init_lr * decay for decay in decay_rates]
        else:
            lr_list = [init_lr * 0.1 ** decay for decay in range(0, len(lr_stages) + 1)]
        self.init_lr = init_lr
        self.lr_list = lr_list
        self.lr_stages = lr_stages
        self.log_freq = log_freq
        self.warmup_epoch = warmup_epoch
        self.steps_per_epoch = steps_per_epoch
        self.num_batch_warm_up = steps_per_epoch * self.warmup_epoch
        self.epoch = 0
        self.model = None

    def set_model(self, model):
        self.model = model

    def set_optimizer(self, optimizer):
        self.model.optimizer = optimizer

    def on_train_begin(self, logs=None):
        if self.epoch == 0:
            step = 1
            self.__set_warm_up_lr(step, self.init_lr, self.num_batch_warm_up)

    def __set_warm_up_lr(self, step, init_lr, num_batch_warm_up):
        '''
        :param step: step = self.epoch * steps_per_epoch + batch
        :param init_lr: 0.1
        :param num_batch_warm_up: num_batch_warm_up = steps_per_epoch * warmup_epoch
        :return:
        '''
        if step <= self.num_batch_warm_up and self.num_batch_warm_up > 0:
            lr = step * init_lr / num_batch_warm_up
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)

    def __get_lr(self, epoch, lr_stages, lr_list):
        lr = None
        max_stages = 0
        if not lr_stages:
            lr = lr_list[0]
        else:
            max_stages = max(lr_stages)
        for index in range(len(lr_stages)):
            if epoch <= lr_stages[index]:
                lr = lr_list[index]
                break
            if epoch > max_stages:
                lr = lr_list[index + 1]
        return lr

    def __set_stages_lr(self, epoch, lr_stages, lr_list):
        '''
        :param epoch:
        :param lr_stages: [    35, 65, 95, 150]
        :param lr_list:   [0.1, 0.01, 0.001, 0.0001, 0.00001]
        :return:
        '''
        # if epoch in lr_stages:
        #     index = lr_stages.index(epoch) + 1
        #     lr = lr_list[index]
        #     tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        lr = self.__get_lr(epoch, lr_stages, lr_list)
        if lr:
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)

    def on_batch_begin(self, batch, logs=None):
        step = self.epoch * self.steps_per_epoch + batch
        self.__set_warm_up_lr(step, self.init_lr, self.num_batch_warm_up)
        if self.log_freq and batch % self.log_freq == 0:
            print('\nstep:{},lr:{}:'.format(step, tf.keras.backend.get_value(self.model.optimizer.lr)))

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.__set_stages_lr(epoch, self.lr_stages, self.lr_list)
        # print('\nepoch:{},lr:{}:'.format(epoch, tf.keras.backend.get_value(self.model.optimizer.lr)))

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch
        logs = logs or {}
        logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)
