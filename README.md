# A Python script to split audio with AI

## Installation

* Clone the source of this library: `git clone https://github.com/timendum/ciccio.git`
* Install dependencies: `pip install -r requirements.txt`
* FFMPEG needs to be in the path

## How to use:

### Training

1. Manualy split audio in two folders:
   * in `data\ok` put what you want to survive (es: speech parts)
   * in `data\ko` put the rest of the audio (es: ads or music parts)
1. Train the AI with `python main.py train`

The output will be the model in `data\svmSM` folder.

### One time split

You can split a file with `python main.py split <source.mp3>`

The program will produce many `source_n.mp3` files in the same folder as the original mp3.


### Podcast

The logic and details are in the `podcast.py` file.

It will download an mp3 for a specific show, split it and then produce an XML for the podcast.

To allow the processing on smaller machine, the input file is splitted in smaller chunks
and every chunk is parsed and analyzed.

Use the `BASE_URL` env var to output full paths for the mp3s.


## Remarks

This module is based on
[pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis/)
by Theodoros Giannakopoulos, under Apache License.  
I only removed unused parts, simplified others and automated a little bit more.

