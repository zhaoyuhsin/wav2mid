# PyTorch Implementation of Onsets and Frames

This is a [PyTorch](https://pytorch.org/) implementation of Google's [Onsets and Frames](https://magenta.tensorflow.org/onsets-frames) model, using the [Maestro dataset](https://magenta.tensorflow.org/datasets/maestro) for training and the Disklavier portion of the [MAPS database](http://www.tsi.telecom-paristech.fr/aao/en/2010/07/08/maps-database-a-piano-database-for-multipitch-estimation-and-automatic-transcription-of-music/) for testing.

## Instructions

This project is quite resource-intensive; 32 GB or larger system memory and 8 GB or larger GPU memory is recommended. 

### Convert WAV file into MIDI
* Convert wav format file into flac by ffmpeg
```bash
ffmpeg -y -loglevel fatal -i a.wav -ac 1 -ar 16000 a.flac
```
* Put the flac file into data/MAPS/flac(for example a.flac)
* Rename the t.tsv and put the tsv file into data/MAPS/tsv/matched (for example a.tsv)
* run evaluate.py 
```bash
python3 evaluate.py model.pt --save-path output/ 
```
* The result a.mid is placed in output/




