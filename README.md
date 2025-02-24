### Managing dependencies

<a href="https://docs.astral.sh/uv/">uv</a> is used for Python dependencies management

To create a virtual environment after cloning this repository, you need to run the following commands:

- ```curl -LsSf https://astral.sh/uv/install.sh | sh```

- ```uv venv```


___

# HW1 report

To evaluate the researched unsupervised approaches, <a href="https://github.com/kashperova/asp-iasa/blob/main/src/models/vad/silero.py">Silero VAD</a> was used as the main model.
<a href="https://github.com/kashperova/asp-iasa/blob/main/src/models/vad/speechbrain.py">Speechbrain VAD</a> (model from speechbrain <a href="https://huggingface.co/speechbrain/vad-crdnn-libriparty">toolkit</a>) was also used as an additional validation.


Initially, two approaches were tested - energy-based and zero-frequency filtering.

In the energy-based approach (implementation <a href="https://github.com/kashperova/asp-iasa/blob/main/src/models/vad/energy.py">here</a>) 
the signal is divided into frames of 10 ms with 5ms overlaps. Energy is calculated for each frame. If the energy is below the specified threshold, the audio chunk is classified as noise. 
Then we create the speech boundaries (segments) from classified frames. If the distance between speech segments is greater than 450 ms, they are merged into one speech segment (450 ms was chosen empirically).

Zff is a filter that removes the constant component (zero frequency) and slow changes in the signal.
At the beginning, we integrate the signal twice (sum the signal values over time), and then apply the moving average.
After that, we differentiate the result twice. 

We didn't have time to implement zff, therefore, for testing, we used a ready-made <a href="https://github.com/kashperova/asp-iasa/blob/main/src/models/vad/zff.py">solution</a> from the authors of the <a href="https://arxiv.org/abs/2206.13420">paper</a>.

Also we add validation on full speech prediction (just to see that our methods much better than this naive approach)


| Method      | Target      | Avg DER | Avg Precision | Avg Recall | Avg F1
|-------------|-------------|---------|---------------|------------|---------
| Energy      | Silero      | 0.4179  |     0.9983    | 0.5833     | 0.7149
| Energy      | Speechbrain | 0.4216  |     0.9996    | 0.5785     | 0.7112 
| Zff         | Silero      | 1.0148  |     0.5117    | 0.9972     | 0.6715
| Zff         | Speechbrain | 1.0069  |     0.5143    | 0.9972     |0.6736
| Full speech | Silero      | 1.0247  |     0.5091    | 0.9999     | 0.6703
| Full speech | Speechbrain | 1.0168  |     0.5117    | 0.9999     | 0.6724


Experiments presented at this <a href="https://github.com/kashperova/asp-iasa/blob/main/notebooks/hw1.ipynb">notebook</a>.

Zff predicted a lot of false positives (segments where the reference actually didn't contain speech), that's why the precision was so low and DER is greater than 1
(we checked the pyannote source, this can indeed happen, since the denominator is the total speech duration in the reference, and if there were a lot of false positives, the numerator is much larger)

### Clustering 

The best approach in this work was K-means clustering.

Here is the list of features we used:

- `mfcc` - mel-frequency cepstal coefficients
- `spectral_centroid` - mean value of audio spectra 
- `zcr` - zero crossing rate
- `spectral_flux`  - rate of change of the signal spectrum 
- `rms` - root mean square - calculates mean energy of a signal
___

### results (audio-lab1.ipynb)

- Method 1 - Cutting out the desired parts of the audio from the left and right of our chunk to get the full context of the fragment, Wiener filter (window = 5)
- Method 2 - Averaging the audio to the left and right of our chunk to get the full context of the fragment, Wiener filter (window = 5)

| Method | Butterworth filter frequency | Avg DER
|------ |------------------------------|--------------|
| 1 | 60 Hz                        | 8.2% (8.2068) |
| 1 | 80 Hz                        | 7.7% (7.6729) | 
| 1 | 90 Hz                        | 8.2% (8.2068) | 
| 1 | 100 Hz                       | 8.1% (8.0870)|
| 2 | 60 Hz                        | 30.5% (30.4734) |
| 2 | 80 Hz                        | 8.0% (7.9940) |
| 2 | 90 Hz                        | 8.2% (8.1589) |
| 2 | 100 Hz                       | 9.5% (9.4925) |


As we can see from the results of the Average Detection Error Rate across all files for the folder p225,
the best result is shown by the First method with a Butterworth filter frequency of 80 Hz. The purpose of the 
Butterworth filter is to reject frequencies that are too quiet, silence, and retain speech frequencies. 
This combination of method and frequency has the lowest error value. 
Why method 2 at 60 Hz has such a large error result is unknown, because we started everything from scratch and 
cleaned the cache of the process. Perhaps it was not cleared properly or something. 

---

<b>Contributors</b>: Vadym Vilhurin, Kateryna Kovalchuk, Sofiia Kashperova.