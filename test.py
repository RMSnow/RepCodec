import torchaudio
import torch

bundle = torchaudio.pipelines.HUBERT_LARGE
model = bundle.get_model()
model_gpu = model.cuda()

wav, sr = torchaudio.load("/data/home/xueyao/workspace/dataset/LibriSpeech/dev-clean/84/121123/84-121123-0000.flac")

def extract_features(wav, hubert, output_layer=18):
    with torch.no_grad():
        feats, _ = hubert.extract_features(wav, num_layers=output_layer,)
        feats = feats[-1].squeeze()
    print(feats.shape, torch.isnan(feats).any())
    return feats