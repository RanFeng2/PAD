from typing import List

from mmseg.registry import MODELS
from mmseg.models.segmentors import EncoderDecoder
from mmseg.utils import ConfigType, OptConfigType, OptMultiConfig, OptSampleList, SampleList
from mmseg.utils import add_prefix

import torch
import torch.nn as nn
from torch import Tensor


@MODELS.register_module()
class EarlyFusionSegmentorPenalty(EncoderDecoder):

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        img = inputs[:, :3]
        img2 = inputs[:, 3:]
        x = self.backbone(img, img2)
        return x
    
    @staticmethod
    def penalty_loss_dict(loss_value):
        losses = dict()
        losses_penalty = {'loss_penalty': loss_value}
        losses.update(add_prefix(losses_penalty, 'fusion'))
        return losses
    
    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        x = self.extract_feat(inputs)

        losses = dict()

        if self.backbone.stage_penalty_losses:
            loss_penalty = self.penalty_loss_dict(sum(self.backbone.stage_penalty_losses.values()))
            losses.update(loss_penalty)

        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses
 
