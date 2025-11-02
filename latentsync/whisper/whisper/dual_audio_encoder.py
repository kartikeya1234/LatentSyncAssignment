# Implementation of a Wav2Vec2-based audio encoder for phoneme feature extraction

import torch
import torch.nn as nn
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from typing import Tuple

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import scipy.io.wavfile as wavfile
    from scipy.signal import resample
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class Wav2Vec2Encoder(nn.Module):
    
    def __init__(
        self, 
        model_name: str = "facebook/wav2vec2-lv-60-espeak-cv-ft",
        target_dim: int = 128,
        cache_dir: str = "./checkpoints/wav2vec2",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        super().__init__()
        
        self.sample_rate = 16000
        self.target_dim = target_dim
        self.device = device
        self.dtype = dtype
        
        print(f"Loading Wav2Vec2 model: {model_name}")
        
        # Load Wav2Vec2 for phoneme extraction
        self.processor = Wav2Vec2Processor.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        self.model = Wav2Vec2ForCTC.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=dtype
        )
        
        self.model = self.model.to(device)
        
        self.hidden_size = self.model.config.hidden_size
        print(f"  Wav2Vec2 native dimension: {self.hidden_size}")
        
        # Projection layer: 1024 → 256 → 128
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, 256, dtype=self.dtype),
            nn.LayerNorm(256, dtype=self.dtype),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, target_dim, dtype=self.dtype)
        ).to(device)
        
        self.model.eval()
    
    def _load_audio(self, audio_path: str) -> torch.Tensor:
        
        if LIBROSA_AVAILABLE:
            # Use librosa (preferred)
            waveform, sr = librosa.load(audio_path, sr=self.sample_rate)
            waveform = torch.from_numpy(waveform).float()
        elif SCIPY_AVAILABLE:
            # Use scipy
            sr, waveform = wavfile.read(audio_path)
            
            # Convert to float32
            if waveform.dtype == np.int16:
                waveform = waveform.astype(np.float32) / 32768.0
            elif waveform.dtype == np.int32:
                waveform = waveform.astype(np.float32) / 2147483648.0
            
            # Convert stereo to mono
            if len(waveform.shape) > 1:
                waveform = waveform.mean(axis=1)
            
            # Resample if needed
            if sr != self.sample_rate:
                num_samples = int(len(waveform) * self.sample_rate / sr)
                waveform = resample(waveform, num_samples)
            
            waveform = torch.from_numpy(waveform).float()
        else:
            raise ImportError(
                "Neither librosa nor scipy is available. "
                "Please install one: pip install librosa"
            )
        
        return waveform
    
    @torch.no_grad()
    def forward(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        
        # Load audio
        waveform = self._load_audio(audio_path)
        audio_length = waveform.shape[0]
        
        # Convert to numpy for processor
        waveform_np = waveform.cpu().numpy()
        
        # Process with Wav2Vec2
        inputs = self.processor(
            waveform_np,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        input_values = inputs["input_values"].to(device=self.device, dtype=self.dtype)
    
        if "attention_mask" in inputs:
            attention_mask = inputs["attention_mask"].to(device=self.device, dtype=torch.long)
        else:
            attention_mask = None
        
        model_inputs = {
            "input_values": input_values,
        }
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        
        outputs = self.model(**model_inputs, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states[-1]  
        
        features = self.projection(hidden_states)
        
        features = features.squeeze(0)  
        
        return features, audio_length


def load_wav2vec2_encoder(
    model_name: str = "facebook/wav2vec2-lv-60-espeak-cv-ft",
    target_dim: int = 128,
    cache_dir: str = "./checkpoints/wav2vec2",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16
) -> Wav2Vec2Encoder:

    encoder = Wav2Vec2Encoder(
        model_name=model_name,
        target_dim=target_dim,
        cache_dir=cache_dir,
        device=device
    )
    encoder.eval()
    
    print(f"✓ Wav2Vec2 encoder loaded on {device}")
    return encoder


if __name__ == '__main__':
    """Test the Wav2Vec2 encoder"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', type=str, required=True, help='Audio file path')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Testing Wav2Vec2 Encoder")
    print("="*60)
    
    encoder = load_wav2vec2_encoder(device=args.device)
    
    print(f"\nExtracting features from: {args.audio}")
    features, audio_length = encoder(args.audio)

    print(f'\nExtraction successful!')
    print("="*60)