import os
import uuid
from pathlib import Path

import ffmpeg
from pydub import AudioSegment 
import numpy as np

np.array

def convert_mp3(path: str):
    target_path = Path(path)

    if not target_path.exists():
        FileExistsError()
        return

    stream = ffmpeg.input(path)
    stream = ffmpeg.output(stream, str(target_path.with_suffix(".mp3")))
    ffmpeg.run(stream)

def extract_audio_truck(file_path: str, start_time:int, end_time:int, new_file_name: str = None):
    target_path = Path(file_path)

    if not target_path.exists():
        FileExistsError()
        return

    audio: AudioSegment = AudioSegment.from_file(target_path, format="mp3")

    if new_file_name is None:
        new_file_name = uuid.uuid4() 
    
    extract_target_audio = audio[start_time*1000:end_time*1000]

    louder_extract_target_audio = extract_target_audio + 10

    Path(new_file_name).parent.mkdir(parents=True, exist_ok=True)

    louder_extract_target_audio.export(new_file_name+".mp3", format="mp3")


def test_whisper(file_path: str):
    import whisper

    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    print(result["text"])



if __name__=='__main__':
    test_audio = "./audio_truck/output/test.mp3"

    test_whisper(test_audio)