import argparse
import librosa
from src.spectrogram import compute_spectrogram
from src.seam_carving import find_seam, extract_if_and_amp
from src.estimation import coarse_search, refine_params

def run_pipeline(audio_path):
    x, sr = librosa.load(audio_path, sr=None)
    S,freq,time,xds,sr_ds = compute_spectrogram(x,sr)
    seam=find_seam(S,penalty=5)
    inst_freq,inst_amp=extract_if_and_amp(S,freq,time,seam)
    coarse=coarse_search(time,inst_freq)
    refined=refine_params(coarse,time,inst_freq)
    print('Estimated parameters:', refined)
    return refined

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--audio_path',required=True)
    args=parser.parse_args()
    run_pipeline(args.audio_path)
