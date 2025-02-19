**Q: The two main consecutive procedures for transforming analog signal to a digital one are:**
A: sampling & quantization

**Q: Why are "Window" waveforms much less complex than "Full" waveforms?**
A: They represent only a small portion of the signal. The full signal is dynamically changeable structure (its properties like amplitude or phase significantly change within time).
But if we took short window of signal, it can be stationary (not changeable). Therefore, when expanding into a Fourier series, we need less harmonics for approximation.

**Q: Why do we usually use only Amplitude from the STFT transform?**
A: ?? phase data is unstructured and not used in analysis (spectrogram)

**Q: Which type of filter would you use to attenuate frequencies above a certain threshold?**
A: Low-pass filter

**Q: Why do we need augmentations in the modern Deep Learning era? Where can we use them? (propose as many applications, as possible?**
A: To increase training dataset size i.e. to prevent overfitting, to improve model robustness, to reduce biases for imbalanced datasets. Also it's a cheap method to collect more data.

**Q: What are the advantages and disadvantages of using higher order filters?**
A: ?? Higher order filters have steeper roll-off for better frequency attenuation.
However,  they introduce more phase shift, leading to potential signal distortion.
