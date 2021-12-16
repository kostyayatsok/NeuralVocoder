from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.conv import Conv2d
from torch.nn.utils import weight_norm

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods: List) -> None:
        super().__init__()
        self.periods = periods
        classifiers = []
        for p in self.periods:
            classifiers.append(MPDClassifier(p))
        self.classifiers = nn.ModuleList(classifiers)
    def forward(self, audio: torch.Tensor) -> List[torch.Tensor]:
        """
            audio -- 1D raw audio of length T
        """
        T = audio.size(1)
        res = []
        for p, classifier in zip(self.periods, self.classifiers):
            x = F.pad(audio, (0, T % p))
            x = x.view(-1, p, T // p)
            x = classifier(x)
            res.append(x)
        return res
    
class MPDClassifier(nn.Module):
    # TODO weight normalization
    def __init__(self, p) -> None:
        super().__init__()
        layers = []
        in_ch = p
        for l in range(1, 5):
            out_ch = 2**(5+l)
            layers.append(nn.Sequential(  
                nn.Conv2d(in_ch, out_ch, (5, 1), stride=(3, 1)),
                nn.LeakyReLU()
            ))
            in_ch = out_ch
        out_ch = 1024
        layers.append(nn.Sequential(  
            nn.Conv2d(in_ch, out_ch, (5, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(out_ch, 1, (3, 1))
        ))
        self.net = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class MultiScaleDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self):