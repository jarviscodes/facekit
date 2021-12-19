# Facekit

![Version 1.0.0](https://img.shields.io/badge/Version-1.0.0-blue)
![MIT License](https://img.shields.io/badge/License-MIT-success)

Face kit is a library that uses DNNs to ease data collection for other neural networks such as face recognition or face analysis.
It currently has a MTCNN based face extractor for image directories as well as video directories with `vidsnap`.

## Installation

Note: For some reason I can't get setup.py to pickup tensorflow, so after installing facekit, please also install tensorflow.

### From PyPi

```
pip3 install facekit
pip3 install tensorflow
```

### From Source
```
git pull https://github.com/jarviscodes/facekit
setup.py install
pip3 install tensorflow
```

## Usage
```
Usage: python -m facekit [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  extract-faces
  extract-faces-video
```

### extract-faces
```
(env) E:\Users\Jarvis\PycharmProjects\Facekit>python -m facekit extract-faces --help
Facekit v1.0.0
Usage: python -m facekit extract-faces [OPTIONS]

Options:
  -i, --in_path TEXT    Path where detector will pick up images.
  -o, --out_path TEXT   Path where detector will store images.
  -a, --accuracy FLOAT  Minimum detector threshold accuracy.
  --preload             Preload images in memory. More memory intensive, might
                        be faster on HDDs!
  --help                Show this message and exit.
```

### extract-faces-video

```
(env) E:\Users\Jarvis\PycharmProjects\Facekit>python -m facekit extract-faces-video --help
Facekit v1.0.0
Usage: python -m facekit extract-faces-video [OPTIONS]

Options:
  -v, --video_in TEXT
  --video_interval INTEGER
  -i, --detector_in TEXT
  -o, --detector_out TEXT
  -a, --accuracy FLOAT      Minimum detector threshold accuracy.
  --preload                 Preload images in memory. More memory intensive,
                            might be faster on HDDs!
  --help                    Show this message and exit.

```

### Demo
![Video Extractor Gif](video-extractor.gif)