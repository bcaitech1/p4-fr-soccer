import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import _IncompatibleKeys

from x_transformers import *
from x_transformers.autoregressive_wrapper import *
# from timm.models.vision_transformer import VisionTransformer
from .vision_transformer import VisionTransformer
from timm.models.resnetv2 import ResNetV2
from timm.models.layers import StdConv2dSame
from einops import rearrange, repeat
from dataset import START, PAD, END

import numpy as np
import math
import random

from dataset import START, PAD, END

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomARWrapper(AutoregressiveWrapper):
    def __init__(self, *args, **kwargs):
        super(CustomARWrapper, self).__init__(*args, **kwargs)

    @torch.no_grad()
    def generate(self, start_tokens, seq_len, eos_token=None, temperature=1., filter_logits_fn=top_k, filter_thres=0.9,
                 **kwargs):
        device = start_tokens.device
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        self.net.eval()
        out = start_tokens
        mask = kwargs.pop('mask', None)
        if mask is None:
            mask = torch.full_like(out, True, dtype=torch.bool, device=out.device)

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]
            mask = mask[:, -self.max_seq_len:]
            # print('arw:',out.shape)
            logits = self.net(x, mask=mask, **kwargs)[:, -1, :]

            if filter_logits_fn in {top_k, top_p}:
                filtered_logits = filter_logits_fn(logits, thres=filter_thres)
                probs = F.softmax(filtered_logits / temperature, dim=-1)

            elif filter_logits_fn is entmax:
                probs = entmax(logits / temperature, alpha=ENTMAX_ALPHA, dim=-1)

            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)
            mask = F.pad(mask, (0, 1), value=True)

            if eos_token is not None and (torch.cumsum(out == eos_token, 1)[:, -1] >= 1).all():
                break

        out = out[:, t:]

        if num_dims == 1:
            out = out.squeeze(0)

        self.net.train(was_training)
        return out

    def forward(self, x, **kwargs):
        #print(x)
        xi = x[:, :-1]
        xo = x[:, 1:]

        # help auto-solve a frequent area of confusion around input masks in auto-regressive
        # if user supplies a mask that is only off by one from the source sequence, resolve itfor them
        mask = kwargs.get('mask', None)
        if mask is not None and mask.shape[1] == x.shape[1]:
            mask = mask[:, :-1]
            kwargs['mask'] = mask

        out = self.net(xi, **kwargs)
        # print(out.shape)
        # print(out.transpose(1,2).shape)
        # print(xo.shape)
        loss = F.cross_entropy(out.transpose(1, 2), xo, ignore_index=2)
        return loss


class CustomVisionTransformer(VisionTransformer):
    def __init__(self, img_size=224, *args, **kwargs):
        super(CustomVisionTransformer, self).__init__(img_size=img_size, *args, **kwargs)
        self.height, self.width = img_size
        self.patch_size = 16

    def forward_features(self, x):
        #(x.shape)
        B, c, h, w = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        h, w = h // self.patch_size, w // self.patch_size
        pos_emb_ind = repeat(torch.arange(h) * (self.width // self.patch_size - w), 'h -> (h w)', w=w) + torch.arange(
            h * w)

        #print(torch.zeros(1))

        pos_emb_ind = torch.cat((torch.zeros(1,dtype=torch.long), pos_emb_ind + 1), dim=0).long()
        x += self.pos_embed[:, pos_emb_ind]
        # x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x


class ViT(nn.Module):
    def __init__(self, FLAGS, train_dataset, checkpoint=None, temp=.333):
        super(ViT, self).__init__()
        self.bos_token = train_dataset.token_to_id[START]
        self.eos_token = train_dataset.token_to_id[END]
        self.pad_token = train_dataset.token_to_id[PAD]
        self.max_seq_len = FLAGS.ViT.max_seq_len
        self.temperature = temp

        self.encoder = CustomVisionTransformer(img_size=(FLAGS.input_size.height, FLAGS.input_size.width),
                                               patch_size=FLAGS.ViT.encoder.patch_size,
                                               in_chans=FLAGS.ViT.channels,
                                               num_classes=0,
                                               embed_dim=FLAGS.ViT.encoder.dim,
                                               depth=FLAGS.ViT.encoder.depth,
                                               num_heads=FLAGS.ViT.encoder.heads,
                                               hybrid_backbone=ResNetV2(
                                                   layers=FLAGS.ViT.backbone.layers, num_classes=0, global_pool='',
                                                   in_chans=FLAGS.ViT.channels,
                                                   preact=False, stem_type='same', conv_layer=StdConv2dSame)
                                               )

        self.decoder = CustomARWrapper(
            TransformerWrapper(
                num_tokens=len(train_dataset.id_to_token),
                max_seq_len=self.max_seq_len,
                attn_layers=Decoder(
                    dim=FLAGS.ViT.decoder.dim,
                    depth=FLAGS.ViT.decoder.depth,
                    heads=FLAGS.ViT.decoder.heads,
                    attn_on_attn=FLAGS.ViT.decoder.args.attn_on_attn,
                    cross_attend=FLAGS.ViT.decoder.args.cross_attend,
                    ff_glu=FLAGS.ViT.decoder.args.ff_glu,
                    rel_pos_bias=FLAGS.ViT.decoder.args.rel_pos_bias,
                    use_scalenorm=FLAGS.ViT.decoder.args.use_scalenorm
                )),
            pad_value=self.pad_token)

        self.criterion = (
            nn.CrossEntropyLoss()
        )

        if checkpoint is not None:
            self.load_state_dict(checkpoint)


    def load_state_dict(self, state_dict, strict=True):
        total_parameters = 0
        matches = 0
        mismatches = 0
        with torch.no_grad():
            for name,param in self.named_parameters():
                total_parameters+=1
                try:
                    param.copy_(state_dict[name])
                    matches+=1
                except Exception as e:
                    print(f'disable to get parameter : {name}')
                    print(e)
                    mismatches+=1
                    continue
        print(f"Load weights : {matches}/{total_parameters} mathches, {mismatches} mismatches.")

    def forward(self, x: torch.Tensor, expected, is_train, teacher_focing_ratio=None):
        device = x.device
        encoded = self.encoder(x.to(device))
        dec = self.decoder.generate(torch.LongTensor([self.bos_token] * len(x))[:, None].to(device), self.max_seq_len,
                                    eos_token=self.eos_token, context=encoded, temperature=self.temperature)
        return dec
