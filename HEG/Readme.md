# HOI Prediction with Emotion Detection and Gaze Following

This repository focuses on human-object interaction (HOI), gaze detection, and emotion detection on a video file. It combines vision models for these tasks from different sources.

## Vision Models Used

1. [HOI Interaction and Gaze Following](https://github.com/nizhf/hoi-prediction-gaze-transformer)
2. [Emotion Detection](https://huggingface.co/dima806/facial_emotions_image_detection)

## Usage

To run inference on arbitrary videos, follow these steps:

1. Unzip the zip file and cd into the MAIN folder.

2. Download the weights folder and place it like HEG/weights/. Link to the folder: https://mega.nz/fm/1nsERKDK

3. Then to perform inference, replace the placeholders in the following command with directory pathss in your system and run:
`python run.py --source /Path/To/Source/Video --out /Path/To/Output --hoi-thres 0.17 --show_bb --show_emotions --print`

#### Parameters

1. hoi-thres: The HOI score threshold to log the interaction.
2. show_bb: Set this flag to show the bounding boxes in the resultant video.
3. show_emotions: Set this flag to show the detected emotions.
4. print: Set this flag to print the interactions to the console.

## Acknowledgments
This repository is built on top of the existing works of another repository, linked as https://github.com/nizhf/hoi-prediction-gaze-transformer.
