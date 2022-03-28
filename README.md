
# A Python script to split audio with AI

## Installation

* Clone the source of this library: `git clone https://github.com/timendum/ciccio.git`
* Install dependencies: `pip install -r ./requirements.txt`
* FFMPEG needs to be in the path

## How to use:

1. Manualy split audio in two folders:
   * in `data\ok` put what you want to survive (es: speech parts)
   * in `data\ko` put the rest of the audio (es: ads or music parts)
1. Train the AI with `python main.py train`
1. Now you can split a file with `python main.py split <source.mp3>`

The program will produce many `source_n.mp3` files.

## Remarks

These scripts are based on
[pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis/)
by Theodoros Giannakopoulos, under Apache License.  
I only removed unused parts, simplified others and automated the split.

