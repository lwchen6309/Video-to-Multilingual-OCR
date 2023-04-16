# Video number detector

This app utilize these models for text tracking in video
- [easyocr](https://github.com/JaidedAI/EasyOCR) for fast but whole screen detection
- [trocr-large](https://huggingface.co/microsoft/trocr-large-printed) for accurate detection



# Huggingface api
See out the hugggingface api [here](https://huggingface.co/spaces/stupidog04/Video-to-Multilingual-OCR?logs=build)

# Installation

Clone the source code
```
git clone git@github.com:lwchen6309/Video-to-Multilingual-OCR.git
```
Install ffmpeg
```

```
Setup conda enviroment
```
conda create -n video_ocr python=3.10
conda activate video_ocr
conda install pip
```
Install pytorch with cuda if available
```
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
```
Install the library
```
cd ./Video-to-Multilingual-OCR
pip install -r requirements.txt
```
Run the script
```
python app.py
```


# Acknowledgement
This project is revied from [Image-to-Multilingual-OCR](https://huggingface.co/spaces/awacke1/Image-to-Multilingual-OCR)



