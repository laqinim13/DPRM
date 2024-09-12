# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .mpii import MPIIDataset as mpii
from .custom_mpii import CustomMPIIDataset as custom_mpii
from .coco import COCODataset as coco
from .custom_coco import CustomCOCODataset as custom_coco
from .concat_dataset import ConcatDataset,ConcatDatasetResample

