import sys
sys.path.append('./')

from modules.TextEncoder import BartPhoExtractor
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models import create_model
from torchscale.model.BEiT3 import BEiT3
from torchscale.component.multiway_network import MutliwayEmbedding
from torchscale.component.embedding import PositionalEmbedding
from torchscale.architecture.encoder import Encoder
from torchscale.architecture.config import EncoderConfig
import torch.nn as nn
import torch
import math

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


def _get_base_config(
        img_size=224, patch_size=16, drop_path_rate=0, 
        checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs
):
    return EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True, 
        layernorm_embedding=False, normalize_output=True, no_output_layer=True, 
        drop_path_rate=drop_path_rate, encoder_embed_dim=768, encoder_attention_heads=4, 
        encoder_ffn_embed_dim=int(768 * mlp_ratio), encoder_layers=4, 
        checkpoint_activations=checkpoint_activations, 
    )

class Pooler(nn.Module):
    def __init__(self, input_features, output_features, norm_layer):
        super().__init__()
        self.norm = norm_layer(input_features)
        self.dense = nn.Linear(input_features, output_features)
        self.activation = nn.Tanh()

    def forward(self, x):
        cls_rep = x[:, 0, :]
        cls_rep = self.norm(cls_rep)
        pooled_output = self.dense(cls_rep)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class ViVQABEiT3(BEiT3):
    def __init__(self, args):
        super(ViVQABEiT3, self).__init__(args)
        self.text_embed = BartPhoExtractor()
        self.linear = nn.Linear(1024, 768)
        
        # being consistent with Fairseq, which starts from 2 for position embedding
        num_position_embeddings = 32
        embed_positions = MutliwayEmbedding(
            modules=[
                PositionalEmbedding(num_position_embeddings + 2, args.encoder_embed_dim),
                PositionalEmbedding(args.max_source_positions, args.encoder_embed_dim),
            ],
            dim=1,
        )
        self.encoder = Encoder(
            args,
            embed_tokens=None,
            embed_positions=embed_positions,
            output_projection=None,
            is_encoder_decoder=False,
        )

    def forward(self, textual_tokens, visual_embeded, text_padding_position):
        multiway_split_position = visual_embeded.size(1)
        x2 = self.linear(self.text_embed(textual_tokens, text_padding_position))
        x = torch.cat([visual_embeded, x2], dim=1)
        encoder_padding_mask = torch.cat(
            [
                torch.zeros(visual_embeded.shape[:-1]).to(visual_embeded.device).bool(),
                text_padding_position,
            ],
            dim=1,
        )
        encoder_out = self.encoder(
            src_tokens=None,
            encoder_padding_mask=encoder_padding_mask,
            token_embeddings=x,
            multiway_split_position=multiway_split_position
        )
        encoder_out["multiway_split_position"] = multiway_split_position
        return encoder_out
    
class BEiT3Wrapper(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.beit3 = ViVQABEiT3(args)
        self.apply(self._init_weights)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return self.beit3.encoder.num_layers
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'beit3.encoder.embed_positions.A.weight', 'beit3.vision_embed.cls_token', 'logit_scale'}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    

class BEiT3ForVietnameseVisualQuestionAnswering(BEiT3Wrapper):
    def __init__(
            self, 
            args, 
            num_classes, 
            norm_layer=nn.LayerNorm, 
            **kwargs
    ):
        super(BEiT3ForVietnameseVisualQuestionAnswering, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        self.pooler = Pooler(
            input_features=embed_dim, 
            output_features=embed_dim, 
            norm_layer=norm_layer,
        )
        self.pooler.apply(self._init_weights)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), 
            norm_layer(embed_dim * 2), 
            nn.GELU(), 
            nn.Linear(embed_dim * 2, num_classes), 
        )
        self.head.apply(self._init_weights)

    def forward(self, image, question, padding_mask, **kwargs):
        outputs = self.beit3(
            textual_tokens=question, 
            visual_embeded=image, 
            text_padding_position=padding_mask, 
        )
        x = outputs["encoder_out"]
        cls_rep = self.pooler(x)
        return self.head(cls_rep)
    
@register_model
def beit3_blip2_vivqa(pretrained=False, **kwargs):
    args = _get_base_config(**kwargs)
    model = BEiT3ForVietnameseVisualQuestionAnswering(args, num_classes=353, **kwargs)
    return model

if __name__ == '__main__':
    # lr=3e-5, eps=1e-8
    model = create_model('beit3_blip2_vivqa', pretrained = False, drop_path_rate=0.5)