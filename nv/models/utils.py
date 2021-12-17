import torch.nn as nn

SLOPE = 0.1

def init_weights(model):
    def init_fn(m):
        if isinstance(m, nn.Linear) or\
        isinstance(m, nn.Conv2d) or\
           isinstance(m, nn.ConvTranspose2d) or\
           isinstance(m, nn.ConvTranspose1d) or\
           isinstance(m, nn.Conv1d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    model.apply(init_fn)
