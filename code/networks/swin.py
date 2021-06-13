import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import _IncompatibleKeys

from x_transformers import *
from x_transformers.autoregressive_wrapper import *
from timm import create_model
from timm.models.layers.patch_embed import PatchEmbed
from copy import deepcopy
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


class CustomSwinTransformer(torch.nn.Module):
    def __init__(self,swin_layer):
        super(CustomSwinTransformer, self).__init__()
        self.patch_embed = PatchEmbed(img_size=(112,448),patch_size=4,in_chans=1,embed_dim=192)
        self.swin_layer = swin_layer
        self.linear = torch.nn.Linear(1536,256)
        self.norm = torch.nn.LayerNorm((256,),eps=1e-06)
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, x):
        x=self.patch_embed(x)
        x=self.swin_layer(x)
        x=self.linear(x)
        x=self.norm(x)
        x=self.avg_pool(x)



class Swin(nn.Module):
    def __init__(self, FLAGS, train_dataset, checkpoint=None, temp=.333):
        super(Swin, self).__init__()
        self.bos_token = train_dataset.token_to_id[START]
        self.eos_token = train_dataset.token_to_id[END]
        self.pad_token = train_dataset.token_to_id[PAD]
        self.max_seq_len = FLAGS.swin.max_seq_len
        self.temperature = temp
        sw = create_model('swin_large_patch4_window7_224',checkpoint_path="/opt/ml/input/data/swin.pth")
        sl = deepcopy(sw.layers)
        del sw
        self.encoder = CustomSwinTransformer(sl)

        self.decoder = CustomARWrapper(
            TransformerWrapper(
                num_tokens=len(train_dataset.id_to_token),
                max_seq_len=self.max_seq_len,
                attn_layers=Decoder(
                    dim=FLAGS.swin.decoder.dim,
                    depth=FLAGS.swin.decoder.depth,
                    heads=FLAGS.swin.decoder.heads,
                    attn_on_attn=FLAGS.swin.decoder.args.attn_on_attn,
                    cross_attend=FLAGS.swin.decoder.args.cross_attend,
                    ff_glu=FLAGS.swin.decoder.args.ff_glu,
                    rel_pos_bias=FLAGS.swin.decoder.args.rel_pos_bias,
                    use_scalenorm=FLAGS.swin.decoder.args.use_scalenorm
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