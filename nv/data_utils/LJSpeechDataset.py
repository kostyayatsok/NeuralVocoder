import torch
import torchaudio
import random
import torch.nn.functional as F

class LJSpeechDataset(torchaudio.datasets.LJSPEECH):    
    def __init__(self, root, segment_size):
        super().__init__(root=root)
        self._tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()
        self.segment_size = segment_size
    def __getitem__(self, index: int):
        waveform, _, _, transcript = super().__getitem__(index)
        waveform_length = torch.tensor([waveform.shape[-1]]).int()
        
        tokens, token_lengths = self._tokenizer(transcript)
        
        if waveform.size(1) >= self.segment_size:
            frm = random.randint(0, waveform.size(1) - self.segment_size)
            to = frm+self.segment_size
            waveform = waveform[:, frm:to]
            waveform_length = torch.tensor([self.segment_size]).int()
        else:
            waveform = F.pad(waveform, (0, self.segment_size - waveform.size(1)))

        
        return waveform, waveform_length, transcript, tokens, token_lengths
    
    def decode(self, tokens, lengths):
        result = []
        for tokens_, length in zip(tokens, lengths):
            text = "".join([
                self._tokenizer.tokens[token]
                for token in tokens_[:length]
            ])
            result.append(text)
        return result
                