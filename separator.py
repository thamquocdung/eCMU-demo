

import onnxruntime
import numpy as np
import torch
import time
import soundfile as sf
from pathlib import Path
import librosa


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def padding(x, length):
    offsets = length - x.shape[-1]
    left_pad = offsets // 2
    right_pad = offsets - left_pad

    return left_pad, right_pad, torch.nn.functional.pad(x, (left_pad, right_pad))

class STFT:
    def __init__(self, n_fft, hop_length, dim_f):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(window_length=n_fft, periodic=True)        
        self.dim_f = dim_f
    
    def __call__(self, x):
        window = self.window.to(x.device)
        batch_dims = x.shape[:-2]
        c, t = x.shape[-2:]
        x = x.reshape([-1, t])
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=window, center=True, return_complex=False)
        x = x.permute([0,3,1,2])
        x = x.reshape([*batch_dims,c,2,-1,x.shape[-1]]).reshape([*batch_dims,c*2,-1,x.shape[-1]])
        return x[...,:self.dim_f,:]

    def inverse(self, x):
        window = self.window.to(x.device)
        batch_dims = x.shape[:-3]
        c,f,t = x.shape[-3:]
        n = self.n_fft//2+1
        f_pad = torch.zeros([*batch_dims,c,n-f,t]).to(x.device)
        x = torch.cat([x, f_pad], -2)
        x = x.reshape([*batch_dims,c//2,2,n,t]).reshape([-1,2,n,t])
        x = x.permute([0,2,3,1])
        x = x[...,0] + x[...,1] * 1.j
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=window, center=True)
        x = x.reshape([*batch_dims,2,-1])
        return x
    

class Separator:
    def __init__(self, n_fft=8192, hop_length=2048, dim_f=4096, onxx="unet_conformer.onnx"):
        self.ort_session = onnxruntime.InferenceSession(onxx, providers=["CPUExecutionProvider"])
        self.stft = STFT(n_fft, hop_length, dim_f)
        self.chunk_size = 260096
        self.overlap = 0.5
        self.nb_targets = 4
        self.num_subbands = 4
        self.sources = ["vocals", "drums", "bass", "other"]
        self.OUTPUT_DIR = "static/audio"
        self.target_sr = 44100

    def separate(self, audio_path):
        filename = audio_path.split("/")[-1][:-4]
        mixture, sr = sf.read(audio_path, dtype='float32', fill_value=0.)
        mixture = mixture.T
        print(mixture.shape, sr, self.target_sr)
        mixture = librosa.resample(mixture, orig_sr=sr, target_sr=self.target_sr)
        mixture = torch.as_tensor(mixture[None,...])
        estimated_waves = self.segment_separate(mixture)

        print(estimated_waves.shape)
        OUTPUT_DIR = self.OUTPUT_DIR
        Path(f'{OUTPUT_DIR}/{filename}').mkdir(parents=True, exist_ok=True)
        for i, source in enumerate(self.sources):
            print(f'{OUTPUT_DIR}/{filename}/{source}.wav')
            sf.write(f'{OUTPUT_DIR}/{filename}/{source}.wav', estimated_waves[0][i].T.numpy(), self.target_sr)


    def segment_separate(self, mix):
        segment = self.chunk_size
        stride = int((1-self.overlap) * segment)
        nb_sources = self.nb_targets
        batch, channels, length = mix.shape
        length = mix.shape[-1]
        offsets = range(0, length, stride)


        out = torch.zeros(batch, nb_sources, channels, length)
        sum_weight = torch.zeros(length)
        weight = torch.cat([torch.arange(1, segment // 2 + 1),
                        torch.arange(segment - segment // 2, 0, -1)])
        weight = (weight / weight.max())
        assert len(weight) == segment

        for offset in offsets:
            chunk = mix[..., offset:offset+segment]
            left_pad, right_pad, chunk_pad = padding(chunk, length=segment)
            chunk_out = self.inference(chunk_pad)
            if left_pad > 0:
                chunk_out = chunk_out[...,left_pad:]
            if right_pad > 0:
                chunk_out = chunk_out[...,:-right_pad]

            # chunk_out = chunk_out.cpu().detach()
            chunk_length = chunk_out.shape[-1]
            w = weight[:chunk_length]
            out[..., offset:offset + segment] += (w * chunk_out)
            sum_weight[offset:offset + segment] += w #.to(mix.device)
            offset += segment

        assert sum_weight.min() > 0
        out /= sum_weight

        return out

    def cac2cws(self, x):
        k = self.num_subbands
        b,c,f,t = x.shape
        x = x.reshape(b,c,k,f//k,t)
        x = x.reshape(b,c*k,f//k,t)
        return x

    def cws2cac(self, x):
        k = self.num_subbands
        b,c,f,t = x.shape
        x = x.reshape(b,c//k,k,f,t)
        x = x.reshape(b,c//k,f*k,t)
        return x

    def inference(self, audio):
        """Performing the separation on audio input

        Args:
            audio (Tensor): [shape=(nb_samples, nb_channels, nb_timesteps)]
                mixture audio waveform

        Returns:
            Tensor: stacked tensor of separated waveforms
                shape `(nb_samples, nb_targets, nb_channels, nb_timesteps)`
        """
        X = self.stft(audio)
        X = self.cac2cws(X)

        ort_inputs = {self.ort_session.get_inputs()[0].name: to_numpy(X)}
        ort_outs = self.ort_session.run(None, ort_inputs)
        ort_outs = torch.as_tensor(ort_outs[0])
        spectrogram_hat = self.stft.inverse(ort_outs)
        print(1111, spectrogram_hat.shape)

        return spectrogram_hat
    


if __name__ == "__main__":
    separator = Separator()
    separator.separate("static/audio/G5ERdrjBe40.wav")