import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from mmengine.model import BaseModule
from mmseg.registry import MODELS


from .PAD_module import PAD_module
FFM_TYPE = {
    'PAD': PAD_module,
}

@MODELS.register_module()
class PADPenalty(BaseModule):
    def __init__(self,
                 backbone,
                 ffm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.backbone = MODELS.build(backbone)
        self.backbone2 = MODELS.build(backbone)

        self.backbone_type = backbone['type']

        if self.backbone_type == 'SegformerFusionBackbone':
            self.depths =  self.backbone.depths
            self.channels = self.backbone.embed_dims
            self.num_stages = self.backbone.num_stages
            self.out_indices = self.backbone.out_indices
        else:
            self.depths =  self.backbone.depths
            self.channels = self.backbone.channels
            self.num_stages = self.backbone.num_stages
            self.out_indices = self.backbone.out_indices

        self.ffm_cfg = ffm_cfg
        
        # FFM
        if ffm_cfg is not None:
            self.stage_penalty_losses = {}
            ffm_num_blocks = ffm_cfg['num_blocks'] if 'num_blocks' in ffm_cfg else [1 for _ in range(self.num_stages)]      # each stage must be greater than 0
            if ffm_cfg['type'] in FFM_TYPE:   
                ffm_class = FFM_TYPE[ffm_cfg['type']]                                                                        # used to store the penalty loss of each stage

                sig = inspect.signature(ffm_class.__init__)
                param_names = list(sig.parameters.keys())[1:]

                self.FFMs = nn.ModuleList([
                    nn.ModuleList([
                        self._build_ffm_for_stage(ffm_class, i, param_names, ffm_cfg)
                        for _ in range(ffm_num_blocks[i])
                    ]) if ffm_num_blocks[i] > 0 else None
                    for i in range(self.num_stages)
                ])
            else:
                raise ValueError(f"Invalid FFM type: {ffm_cfg['type']}")
        

    def _build_ffm_for_stage(self, ffm_class, stage_idx, param_names, ffm_cfg):
        """build the FFM module for each stage, according to the cfg and the __init__ method of the FFM class."""
        ffm_params = {}
        # build the parameters for the FFM module
        for param_name in param_names:
            if param_name in ffm_cfg:
                param_value = ffm_cfg[param_name]
                if isinstance(param_value, list) or isinstance(param_value, tuple):
                    if stage_idx < len(param_value):
                        ffm_params[param_name] = param_value[stage_idx]
                    else:
                        ffm_params[param_name] = param_value[-1]
                else:
                    ffm_params[param_name] = param_value
        # special handling of the 'dim', ensure the correct channel number is used
        if 'dim' in ffm_params:
            if isinstance(ffm_params['dim'], list) or isinstance(ffm_params['dim'], tuple):
                if stage_idx < len(ffm_params['dim']):
                    ffm_params['dim'] = ffm_params['dim'][stage_idx]
                else:
                    ffm_params['dim'] = ffm_params['dim'][-1]
            else:
                ffm_params['dim'] = self.channels[stage_idx]
        else:
            ffm_params['dim'] = self.channels[stage_idx]
        
        return ffm_class(**ffm_params)

    def forward(self, x_opt, x_sar):
        outs = []

        for i in range(self.num_stages):
            if self.backbone_type == 'SegformerFusionBackbone':
                B = x_opt.shape[0]
                x_opt, H, W = getattr(self.backbone, f"patch_embed{i+1}")(x_opt)
                x_sar, _, _ = getattr(self.backbone2, f"patch_embed{i+1}")(x_sar)
                for blk in getattr(self.backbone, f"block{i+1}"):
                    x_opt = blk(x_opt, H, W)
                for blk in getattr(self.backbone2, f"block{i+1}"):
                    x_sar = blk(x_sar, H, W)
            else:
                x_opt = self.backbone.downsample_layers[i](x_opt)
                x_opt = self.backbone.stages[i](x_opt)
                x_sar = self.backbone2.downsample_layers[i](x_sar)
                x_sar = self.backbone2.stages[i](x_sar)

            if i in self.out_indices:
                if self.backbone_type == 'SegformerFusionBackbone':
                    opt_norm_layer = getattr(self.backbone, f'norm{i+1}')
                    sar_norm_layer = getattr(self.backbone2, f'norm{i+1}')
                    x_opt = opt_norm_layer(x_opt)
                    x_sar = sar_norm_layer(x_sar)
                    x_opt = x_opt.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                    x_sar = x_sar.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                else:
                    opt_norm_layer = getattr(self.backbone, f'norm{i}')
                    sar_norm_layer = getattr(self.backbone2, f'norm{i}')
                    x_opt = opt_norm_layer(x_opt)
                    x_sar = sar_norm_layer(x_sar)
                
                #! FFM
                if isinstance(self.FFMs[i], nn.ModuleList):
                    x_fused = x_opt
                    penalty_losses = []
                    num_layers = len(self.FFMs[i])
                    layer_weights = torch.linspace(0.5, 1.0, num_layers, device=x_opt.device)
                    
                    for j in range(num_layers):
                        x_fused = self.FFMs[i][j](x_fused, x_sar)
                        if hasattr(self.FFMs[i][j], 'penalty_loss'):
                            penalty_losses.append(self.FFMs[i][j].penalty_loss)

                    # calculate the weighted average of the phase loss
                    if penalty_losses:
                        self.stage_penalty_losses[i] = torch.sum(
                            torch.stack(penalty_losses) * layer_weights.view(-1, 1, 1, 1)
                        ) / layer_weights.sum()
                    else:
                        self.stage_penalty_losses[i] = torch.tensor(0.0, device=x_opt.device)
                    
                else:
                    x_fused = self.FFMs[i](x_opt, x_sar)
                    if hasattr(self.FFMs[i], 'penalty_loss'):
                        self.stage_penalty_losses[i] = self.FFMs[i].penalty_loss
                    else:
                        self.stage_penalty_losses[i] = torch.tensor(0.0, device=x_opt.device)

                outs.append(x_fused)

        return tuple(outs)
