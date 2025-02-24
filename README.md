### Managing dependencies

<a href="https://docs.astral.sh/uv/">uv</a> is used for Python dependencies management

To create a virtual environment after cloning this repository, you need to run the following commands:

- ```curl -LsSf https://astral.sh/uv/install.sh | sh```

- ```uv venv```


 ### Linters

```pre-commit run --all-files```


### repository structure 
```
root/
  |
  | ------ .misk/tests/
  |             | ----- theory answers
  |
  | ------ notebooks/
  |           | ------ hw1.ipynb
  |           | ------ audio-lab1.ipynb
  |
  | ------ src/                                            | ------ energy.py  
  |         |                     |  ----- vad  ---------- | ------ k_means.py
  |         | ----- models ------ |                        | ------ silero.py
  |         | ----- utils  ------ |  ----- metrics.py      | ------ speechbrain.py 
  |                               |  ----- singleton.py    | ------ zff.py 
  |                               |  ----- wav_utils.py                         
  |  
  | --- dependencies  
```
___
models:

- energy.py  - counts the energy of the frame and classifies whether there is speech or not based on its amount
- k_means.py - For each chunk, it is determined whether there is a signal or not based on the features “mfcc”, “spectral_centroid”, “zcr”, “spectral_flux”, “rms”, and their average value
- silero.py - the code uses the Silero model to determine whether there is speech in the audio segment or not
- speechbrain.py - uses a speechbrain model that handles small and large windows. First, it calculates the probabilities for segments, whether there is speech or not, then it calculates the potential speech --boundaries, calculates the energy for a part of the audio, and if it is sufficient, it is a chunk with speech. Then all the chunks with speech are merged. 
- zff.py - minimizes or eliminates the zero frequency
___

___
utils:

- metrics.py - calculates metrics to verify correct speech detection in audio
- singleton.py - guarantees that only one instance of the object will exist during the entire program execution time
- wav_utils.py - returns sample rate, duration, normalizes audio, and filters it
___

___
features:

- mfcc - mel-frequency cepstal coefficients
- spectral_centroid - mean value of audio spectra 
- zcr - zero crossing rate
- spectral_flux  - rate of change of the signal spectrum 
- rms - root mean square - calculates mean energy of a signal
___

### results (audio-lab1.ipynb)

- Method 1 - Cutting out the desired parts of the audio from the left and right of our chunk to get the full context of the fragment, Wiener filter (window = 5)
- Method 2 - Averaging the audio to the left and right of our chunk to get the full context of the fragment, Wiener filter (window = 5)

| Method | Butterworth filter frequency | Result
|------ | ----------------------------|--------------|
| 1 |  60 Гц | 8.2% (8.2068) |
| 1 |  80 Гц | 7.7% (7.6729) | 
| 1 |  90 Гц | 8.2% (8.2068) | 
| 1 |  100 Гц | 8.1% (8.0870)|
| 2 |  60 Гц | 30.5% (30.4734) |
| 2 |  80 Гц | 8.0% (7.9940) |
| 2 |  90 Гц | 8.2% (8.1589) |
| 2 |  100 Гц | 9.5% (9.4925) |


As we can see from the results of the Average Detection Error Rate across all files for the folder p225,
the best result is shown by the First method with a Butterworth filter frequency of 80 Hz. The purpose of the 
Butterworth filter is to reject frequencies that are too quiet, silence, and retain speech frequencies. 
This combination of method and frequency has the lowest error value. 
Why method 2 at 60 Hz has such a large error result is unknown, because we started everything from scratch and 
cleaned the cache of the process. Perhaps it was not cleared properly or something. 
