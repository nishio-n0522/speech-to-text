import os
import uuid
from pathlib import Path
import json
import time

import torch
import whisper
from pyannote.audio import Pipeline
import ffmpeg
from pydub import AudioSegment 

def convert_to_mp3(path: str):
    target_path = Path(path)

    if not target_path.exists():
        FileExistsError()
        return

    stream = ffmpeg.input(path)
    stream = ffmpeg.output(stream, str(target_path.with_suffix(".mp3")))
    ffmpeg.run(stream)

def convert_to_wav(path: str):
    target_path = Path(path)

    if not target_path.exists():
        FileExistsError()
        return
        
    audio: AudioSegment = AudioSegment.from_file(target_path, format="mp3")
    audio.export("audio_truck/output/"+ target_path.stem + ".wav", format="wav")


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

def exec_whisper(file_path: str, save_dir: str):
    model = whisper.load_model("large", device="cuda")
    result = model.transcribe(file_path)

    result_text_list = result["text"].split(". ")

    with open(save_dir + "speech_to_text_result.txt", "w") as file:
        for result_text in result_text_list:
            file.write(result_text + ".\n")

    print(result["text"])

def annotate_speaker(path):
    recognization_speaker = []

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=os.environ["hugging_face_access_token"]) 
    pipeline.to(torch.device("cuda"))
    diarization = pipeline(path)

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        recognization_speaker.append([speaker,turn.start,turn.end])

    return recognization_speaker

def annotate_speaker_ver1(path):
    recognization_speaker = []

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=os.environ["hugging_face_access_token"]) 
    pipeline.to(torch.device("cuda"))
    diarization = pipeline(path)

    previous_speaker = None
    previous_time = 0

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if previous_speaker is None:
            previous_speaker = speaker

        if speaker is not None and speaker != previous_speaker:
            recognization_speaker.append([previous_speaker, previous_time, turn.end])
            previous_speaker = speaker
            previous_time = turn.end

    recognization_speaker.append([previous_speaker, previous_time, turn.end])

    return recognization_speaker

def save_talk_stream(file_path:str, recognization_speaker: list, save_dir: str):
    target_path = Path(file_path)
    meta_data = []

    if not target_path.exists():
        FileExistsError()
        return

    audio: AudioSegment = AudioSegment.from_file(target_path, format="mp3")
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for i, each_recognization_speaker in enumerate(recognization_speaker):
        extract_target_audio = audio[each_recognization_speaker[1]*1000:each_recognization_speaker[2]*1000]
        save_path = save_dir + f"sentence{i}" + ".wav"
        extract_target_audio.export(save_path, format="wav")
        meta_data.append({"speaker": each_recognization_speaker[0], "file_path": save_path, "start_time": each_recognization_speaker[1], "end_time": each_recognization_speaker[2]})

    with open(save_dir+"meta_data.json", "w") as file:
        json.dump(meta_data, file, indent=4)


def speech_to_text_step_by_step(target_dir: str):
    t0 = time.perf_counter()
    print("start_process")

    with open(target_dir + "meta_data.json", "r") as file:
        meta_data = json.load(file)

    model = whisper.load_model("large", device="cuda")
    speech_to_text_result = []

    for each_meta_data in list(meta_data):
        result = model.transcribe(each_meta_data["file_path"])
        speech_to_text_result.append([each_meta_data["speaker"], result["text"]])

    with open(save_dir+"speech_to_text_result.txt", "w") as file:
        for each_result in speech_to_text_result:
            file.write(f"{each_result[0]}: {each_result[1]}\n")

    print("finish_this_process: ", time.perf_counter() - t0)


if __name__=='__main__':
    test_audio = "./audio_truck/output/test.wav"
    save_dir = "./audio_truck/test_talk_without_annote/"


    exec_whisper(test_audio, save_dir)

    # recognization_speaker = annotate_speaker_ver1(test_audio)
    # save_talk_stream(test_audio, recognization_speaker, save_dir)

    # speech_to_text_step_by_step(target_dir=save_dir)

