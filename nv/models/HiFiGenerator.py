from typing import List
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU
from torch.nn.utils import weight_norm

SLOPE = 0.1
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilations) -> None:
        super().__init__()
        blocks = []
        for d_list in dilations:
            block = []    
            for d in d_list:
                block.append(
                    nn.Sequential(
                        LeakyReLU(SLOPE),
                        weight_norm(nn.Conv1d(
                            in_ch, out_ch, kernel_size,
                            stride=1,
                            padding=(kernel_size-1)//2*d,
                            dilation=d
                        )),
                        LeakyReLU(SLOPE),
                        weight_norm(nn.Conv1d(
                            in_ch, out_ch, kernel_size, stride=1,
                            dilation=1,
                            padding=(kernel_size-1)//2
                        ))
                    )
                )
            blocks.append(nn.Sequential(*block))
        self.blocks = nn.ModuleList(blocks)
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x) + x
        return x


class MultiReceptiveFieldFusion(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernels_size: List,
        dilations: List,
    ) -> None:
        super().__init__()
        res_blocks = []
        for k, D in zip(kernels_size, dilations):
            res_blocks.append(ResBlock(in_ch, out_ch, k, D))
        self.res_blocks = nn.ModuleList(res_blocks)

    def forward(self, x):
        out = 0
        for res_block in self.res_blocks:
            out = out + res_block(x)
        return out
        
      
class HiFiGenerator(nn.Module):
    def __init__(
        self,
        in_ch: int,
        h_u: int,
        k_u: List[int],
        k_r: List[int],
        D_r: List[List[List[int]]],
        *args, **kwargs
    ) -> None:
        #TODO: more skip connections?
        super().__init__()
        layers = [weight_norm(nn.Conv1d(
            in_ch, h_u, kernel_size=7, dilation=1, padding=3
        ))]
        prev_ch = h_u
        for l, k in enumerate(k_u):
            new_ch = h_u//(2**(l+1))
            layers.append(nn.Sequential(
                LeakyReLU(SLOPE),
                weight_norm(nn.ConvTranspose1d(
                    prev_ch, new_ch, kernel_size=k, stride=k//2
                )),
                MultiReceptiveFieldFusion(new_ch, new_ch, k_r, D_r)
            ))
            prev_ch = new_ch
        layers.append(LeakyReLU(SLOPE))
        layers.append(weight_norm(nn.Conv1d(
            prev_ch, 1, kernel_size=7, padding=7//2
        )))
        layers.append(nn.Tanh())
        
        self.net = nn.Sequential(*layers)
    def forward(self, mel, *args, **kwargs):
        return {'wav_pred': self.net(mel).squeeze(1)}
    
    