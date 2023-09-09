import argparse
import numpy as np
import soundfile as sf
import torch
from transformers import AutoTokenizer, VitsModel, pipeline


def main(audio_path: str):
    asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en")
    tr_pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ru")

    tts_model = VitsModel.from_pretrained("facebook/mms-tts-rus")
    tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-rus")

    asr_result = asr_pipe(
        audio_path, return_timestamps=True
    )["chunks"]
    translations = []
    for chunk in asr_result:
        translations.append(
            {
                "ts": chunk["timestamp"],
                "text": tr_pipe(chunk["text"], max_length=256)[0]["translation_text"],
            }
        )

    audios = []
    for chunk in translations:
        audios.append(
            {
                "ts": chunk["ts"],
                "audio": generate_waveform(tts_model, tts_tokenizer, chunk["text"]),
            }
        )

    sr = tts_model.config.sampling_rate
    for idx, chunk in enumerate(audios):
        audio = chunk["audio"]
        len_audio = len(audio) / sr
        needed_len = chunk["ts"][1] - chunk["ts"][0]
        # if len_audio > needed_len:
        #     sr = int(np.ceil(len_audio / needed_len * sr))
        # else:
        #     samples_needed = int(sr * needed_len) - len_audio
        #     silence = np.zeros(samples_needed)
        #     audio = np.concatenate([audio, silence])
        sf.write(
            f"./outputs/output_{idx}.wav",
            data=audio,
            samplerate=sr,
        )


def generate_waveform(tts_model, tts_tokenizer, text: str):
    inputs = tts_tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        return np.ravel(tts_model(**inputs).waveform.to("cpu").detach().numpy())


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-p', '--path', help="Path to audio file", type=str)
    args = argparser.parse_args()
    main(args.path)
