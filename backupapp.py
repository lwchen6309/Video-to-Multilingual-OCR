import pandas as pd
import PIL
from PIL import Image
from PIL import ImageDraw
import gradio as gr
import torch
import easyocr

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

def inference(img, lang):
    reader = easyocr.Reader(lang)
    bounds = reader.readtext(img.name)
    im = PIL.Image.open(img.name)
    draw_boxes(im, bounds)
    im.save('result.jpg')
    return ['result.jpg', pd.DataFrame(bounds).iloc[: , 1:]]

title = '🖼️Image to Multilingual OCR👁️Gradio'
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
    [gr.inputs.Image(type='file', label='Input'),gr.inputs.CheckboxGroup(choices, type="value", default=['en'], label='language')],
    [gr.outputs.Image(type='file', label='Output'), gr.outputs.Dataframe(headers=['text', 'confidence'])],
    title=title,
    description=description,
    article=article,
    examples=examples,
    css=css,
    enable_queue=True
    ).launch(debug=True)