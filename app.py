import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
import torch
import easyocr
import os
from pathlib import Path
import cv2
import pandas as pd


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

def box_size(box):
    points = box[0]
    if len(points) == 4:
        x1, y1 = points[0]
        x2, y2 = points[2]
        return abs(x1 - x2) * abs(y1 - y2)
    else:
        return 0

def box_position(box):
    return (box[0][0][0] + box[0][2][0]) / 2, (box[0][0][1] + box[0][2][1]) / 2


def inference(video, lang, time_step, full_scan=False):
    output = 'results.mp4'
    reader = easyocr.Reader(lang)
    bounds = []   
    vidcap = cv2.VideoCapture(video)
    success, frame = vidcap.read()
    count = 0
    frame_rate = vidcap.get(cv2.CAP_PROP_FPS)
    output_frames = []
    temporal_profiles = []
    compress_mp4 = True

    # Get the positions of the largest boxes in the first frame
    bounds = reader.readtext(frame)
    im = PIL.Image.fromarray(frame)
    im_with_boxes = draw_boxes(im, bounds)
    largest_boxes = sorted(bounds, key=lambda x: box_size(x), reverse=True)
    positions = [box_position(b) for b in largest_boxes]
    temporal_profiles = [[] for _ in range(len(largest_boxes))]
    
    # Match bboxes to position and store the text read by OCR
    while success:
        if count % (int(frame_rate * time_step)) == 0:
            if full_scan:
                bounds = reader.readtext(frame)
                for box in bounds:
                    bbox_pos = box_position(box)
                    for i, position in enumerate(positions):
                        distance = np.linalg.norm(np.array(bbox_pos) - np.array(position))
                        if distance < 50:
                            temporal_profiles[i].append((count / frame_rate, box[1]))
                            break
            else:
                for i, box in enumerate(largest_boxes):
                    x1, y1 = box[0][0]
                    x2, y2 = box[0][2]
                    box_width = x2 - x1
                    box_height = y2 - y1
                    ratio = 0.2
                    x1 = max(0, int(x1 - ratio * box_width))
                    x2 = min(frame.shape[1], int(x2 + ratio * box_width))
                    y1 = max(0, int(y1 - ratio * box_height))
                    y2 = min(frame.shape[0], int(y2 + ratio * box_height))
                    cropped_frame = frame[y1:y2, x1:x2]
                    text = reader.readtext(cropped_frame)
                    if text:
                        temporal_profiles[i].append((count / frame_rate, text[0][1]))
            
            im = PIL.Image.fromarray(frame)
            im_with_boxes = draw_boxes(im, bounds)
            output_frames.append(np.array(im_with_boxes))

        success, frame = vidcap.read()
        count += 1
    
    # Default resolutions of the frame are obtained. The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frames_total = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object.
    if compress_mp4:
        temp = f"{Path(output).stem}_temp{Path(output).suffix}"
        output_video = cv2.VideoWriter(
            temp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
    else:
        output_video = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in output_frames:
        output_video.write(frame)

    # Draw boxes with box indices in the first frame of the output video
    im = Image.fromarray(output_frames[0])
    draw = ImageDraw.Draw(im)
    font_size = 30
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    for i, box in enumerate(largest_boxes):
        draw.text((box_position(box)), f"Box {i+1}", fill='red', font=ImageFont.truetype(font_path, font_size))
    
    output_video.release()
    vidcap.release()

    if compress_mp4:
        # Compressing the video for smaller size and web compatibility.
        os.system(
            f"ffmpeg -y -i {temp} -c:v libx264 -b:v 5000k -minrate 1000k -maxrate 8000k -pass 1 -c:a aac -f mp4 /dev/null && ffmpeg -y -i {temp} -c:v libx264 -b:v 5000k -minrate 1000k -maxrate 8000k -pass 2 -c:a aac -movflags faststart {output}"
        )
        os.system(f"rm -rf {temp} ffmpeg2pass-0.log ffmpeg2pass-0.log.mbtree")
    
    # Format temporal profiles as a DataFrame
    df_list = []
    for i, profile in enumerate(temporal_profiles):
        for t, text in profile:
            df_list.append({"Box": f"Box {i+1}", "Time (s)": t, "Text": text})
        df_list.append({"Box": f"", "Time (s)": "", "Text": ""})
    df = pd.concat([pd.DataFrame(df_list)])
    return output, im, df


title = '🖼️Video to Multilingual OCR👁️Gradio'
description = 'Multilingual OCR which works conveniently on all devices in multiple languages. Adjust time-step for inference and the scan mode according to your requirement. For `Full Scan`, model scan the whole image if flag is ture, while scan only the box detected at the first video frame; this save computation cost; noting that the box is fixed in this case.'
article = "<p style='text-align: center'></p>"

examples = [
['test.mp4',['en'],10,False]
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
        gr.inputs.Video(label='Input Video'),
        gr.inputs.CheckboxGroup(choices, type="value", default=['en'], label='Language'),
        gr.inputs.Number(label='Time Step (in seconds)', default=1.0),
        gr.inputs.Dropdown(['True', 'False'], label='Full Scan', default='False')
    ],
    [
        gr.outputs.Video(label='Output Video'),
        gr.outputs.Image(label='Output Preview', type='numpy'),
        gr.outputs.Dataframe(headers=['Box', 'Time (s)', 'Text'], type='pandas')
    ],
    title=title,
    description=description,
    article=article,
    examples=examples,
    css=css,
    enable_queue=True
).launch(debug=True)