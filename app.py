import numpy as np
import PIL
from PIL import Image, ImageDraw
import gradio as gr
import torch
import easyocr
import os
from pathlib import Path
import cv2


#torch.hub.download_url_to_file('https://github.com/AaronCWacker/Yggdrasil/blob/main/images/BeautyIsTruthTruthisBeauty.JPG', 'BeautyIsTruthTruthisBeauty.JPG')
#torch.hub.download_url_to_file('https://github.com/AaronCWacker/Yggdrasil/blob/main/images/PleaseRepeatLouder.jpg', 'PleaseRepeatLouder.jpg')
#torch.hub.download_url_to_file('https://github.com/AaronCWacker/Yggdrasil/blob/main/images/ProhibitedInWhiteHouse.JPG', 'ProhibitedInWhiteHouse.JPG')

torch.hub.download_url_to_file('https://raw.githubusercontent.com/AaronCWacker/Yggdrasil/master/images/20-Books.jpg','20-Books.jpg')
torch.hub.download_url_to_file('https://github.com/JaidedAI/EasyOCR/raw/master/examples/english.png', 'COVID.png')
torch.hub.download_url_to_file('https://github.com/JaidedAI/EasyOCR/raw/master/examples/chinese.jpg', 'chinese.jpg')
torch.hub.download_url_to_file('https://github.com/JaidedAI/EasyOCR/raw/master/examples/japanese.jpg', 'japanese.jpg')
torch.hub.download_url_to_file('https://i.imgur.com/mwQFd7G.jpeg', 'Hindi.jpeg')

def draw_boxes(image, bounds, color='yellow', width=2):
    draw = ImageDraw.Draw(image)
    for bound in bounds:
        p0, p1, p2, p3 = bound[0]
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
    return image

def inference(video, lang, time_step):
    # output = f"{Path(video).stem}_detected{Path(src).suffix}"
    output = 'results.mp4'
    
    reader = easyocr.Reader(lang)
    bounds = []   
    vidcap = cv2.VideoCapture(video)
    success, frame = vidcap.read()
    count = 0
    frame_rate = vidcap.get(cv2.CAP_PROP_FPS)
    output_frames = []
    while success:
        if count % (int(frame_rate * time_step)) == 0:
            bounds = reader.readtext(frame)
            im = PIL.Image.fromarray(frame)
            draw_boxes(im, bounds)
            output_frames.append(np.array(im))
        success, frame = vidcap.read()
        count += 1
    
    # Default resolutions of the frame are obtained. The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frames_total = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object.
    temp = f"{Path(output).stem}_temp{Path(output).suffix}"
    output_video = cv2.VideoWriter(
        temp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    # output_video = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in output_frames:
        output_video.write(frame)
    output_video.release()
    vidcap.release()

    # Compressing the video for smaller size and web compatibility.
    os.system(
        f"ffmpeg -y -i {temp} -c:v libx264 -b:v 5000k -minrate 1000k -maxrate 8000k -pass 1 -c:a aac -f mp4 /dev/null && ffmpeg -y -i {temp} -c:v libx264 -b:v 5000k -minrate 1000k -maxrate 8000k -pass 2 -c:a aac -movflags faststart {output}"
    )
    os.system(f"rm -rf {temp} ffmpeg2pass-0.log ffmpeg2pass-0.log.mbtree")
    return output


title = '🖼️Video to Multilingual OCR👁️Gradio'
description = 'Multilingual OCR which works conveniently on all devices in multiple languages.'
article = "<p style='text-align: center'></p>"

examples = [
#['PleaseRepeatLouder.jpg',['ja']],['ProhibitedInWhiteHouse.JPG',['en']],['BeautyIsTruthTruthisBeauty.JPG',['en']],
['20-Books.jpg',['en']],['COVID.png',['en']],['chinese.jpg',['ch_sim', 'en']],['japanese.jpg',['ja', 'en']],['Hindi.jpeg',['hi', 'en']]
]

css = ".output_image, .input_image {height: 40rem !important; width: 100% !important;}"
choices = [
    "ch_sim",
    "ch_tra",
    "de",
    "en",
    "es",
    "ja",
    "hi",
    "ru"
]


gr.Interface(
    inference,
    [
        # gr.inputs.Image(type='file', label='Input Image'),
        gr.inputs.Video(label='Input Video'),
        gr.inputs.CheckboxGroup(choices, type="value", default=['en'], label='Language'),
        gr.inputs.Number(label='Time Step (in seconds)', default=1.0)
    ],
    [
        gr.outputs.Video(label='Output Video'),
        # gr.outputs.Dataframe(headers=['Text', 'Confidence'])
    ],
    title=title,
    description=description,
    article=article,
    # examples=examples,
    css=css,
    enable_queue=True
).launch(debug=True)