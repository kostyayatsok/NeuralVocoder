import argparse
import sys
import os
import torch
import torchaudio
import glob
from nv.models import HiFiGenerator
from nv.data_utils import MelSpectrogram

vocoder_config = {
    "in_ch": 80,
    "h_u": 128,
    "k_u": [16, 16, 4, 4],
    "k_r": [3, 7, 11],
    "D_r": [
        [[1, 1], [3, 1], [5, 1]],
        [[1, 1], [3, 1], [5, 1]],
        [[1, 1], [3, 1], [5, 1]]
    ]
}
mel_config = {
    "sample_rate" : 22050,
    "win_length"  : 1024,
    "hop_length"  : 256,
    "n_fft"       : 1024,
    "f_min"       : 0,
    "f_max"       : 8000,
    "n_mels"      : 80,
    "power"       : 1.0,
    "pad_value"   : -11.5129251
}


weights_path = './vocoder.pt'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
vocoder = HiFiGenerator(**vocoder_config).to(device)
vocoder.load_state_dict(torch.load(weights_path))
vocoder = vocoder.eval()

featurizer = MelSpectrogram(mel_config).to(device)
    
def mel2wav(in_path, out_path):
    waveform, _ = torchaudio.load(in_path)
    waveform = waveform.to(device)
    mel = featurizer(waveform, 0)['mel']
    wav_pred = vocoder(mel).cpu()
    
    torchaudio.save(out_path, wav_pred, 22050)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-o",
        "--output",
        default='./test_outputs/',
        type=str,
        help="config file path",
    )
    args.add_argument(
        "-i",
        "--input",
        default='./test',
        type=str,
        help="test wavs path",
    )
    args = args.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    for i, path in enumerate(glob.glob(f"{args.input}/*.wav")):
        mel2wav(path, f"{args.output}/{path.split('/')[-1]}")