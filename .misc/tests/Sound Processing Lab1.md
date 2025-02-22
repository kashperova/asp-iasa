The two main consecutive procedures for transforming analog signal to a digital one are:
Sampling, Quantization

***
- The process of converting continuously varying signals into digital signals is called analog-to-digital conversion. 
    
- Generally, analog-to-digital conversion is carried out in the order of sampling-quantization-coding. 
- ### Sampling

[Sampling](https://resources.pcb.cadence.com/blog/2020-anti-aliasing-filter-design-and-applications-in-sampling) is the process in which continuous-time signals or analog signals are converted into  discrete-time signals. The samples of the analog signals are taken at discrete time instants to obtain discrete-time signals. Sampling is associated with a term called sampling rate, which gives the number of samples per second, or sampling frequency.

The sampling frequency is significant in converting the analog signal to a digital signal without losing any special information. An adequate sampling frequency should be selected for recovering the spectrum of the analog signal from the spectrum of the discrete-time signal.
 Quantization

The conversion of a discrete-time continuous-valued signal into a discrete-time discrete-value signal is called quantization. In the quantization process, each signal sample is represented by a value chosen from the finite set of possible values. The possible values are collectively called quantization levels. The difference between the unquantized sample and the quantized output is called the quantization error. The analog-to-digital conversion effectiveness can be improved by minimizing the quantization error. 

### Coding or Encoding

The process in which the discrete value samples are represented by an n-bit binary sequence or code is called coding.
- According to the sampling theorem, the sampling rate is selected such that it is greater than or equal to 2fmsamples per second, where fmis the highest frequency of the continuous time or analog signal.


**TODO**: Implement Butterworth Filter from scratch. 
***



Why are "Window" waveforms much less complex than "Full" waveforms?
Because a window is a means to capture a part of the signal, so depending on the size of the window, we can capture a very small part of the signal and the complexity of this part we capture depends on the duration of the window, although this part will still not be as complex as the full signal, which is dynamically changable structure. Also, if we took short window of signal, it can be stationary (not changeable). Therefore, when expanding into a Fourier series, we need less harmonics for approximation.


***
A: They represent only a small portion of the signal. The full signal is dynamically changeable structure (its properties like amplitude or phase significantly change within time).

But if we took short window of signal, it can be stationary (not changeable). Therefore, when expanding into a Fourier series, we need less harmonics for approximation.

  

Тому що вікно - це засіб для захоплення частини сигналу, тому залежно від розміру вікна ми можемо захопити як дуже маленьку частину сигналу, так і більшу, і від цього залежить комплексність частини сигналу, яку ми захоплюємо, хоча ця частина все одно буде не така складна, як повний сигнал. Віконна форма демонструє частину сигналу, яка може не містити якихось різких спадів чи скачків, які присутні в повному сигналі.
***
*** 
Перетворення Фур'є (і його аватари) є прототипом дуальності. Дуальність тут означає, що ви можете представити сигнал у деякій первинній області (час) у подвійній області (тут частота). Таке перетворення має володіти корисними властивостями, зберігати інформацію про сигнал і додавати розуміння, наприклад, дещо простішу інтерпретацію.

Сигнали традиційно представляють як «амплітуди чогось для кожного моменту часу». Ці амплітуди не завжди абсолютні. Але ми можемо сподіватися, що вони лінійно пов'язані з реальними фізичними величинами. І що коефіцієнт лінійності не змінюється з часом. За таких передумов синуси і косинуси, або комплексні експоненти (цисоїди), є дуже хорошим способом (можливо, найкращим) моделювання системи. І, в свою чергу, вони можуть бути використані для представлення даних, як лінійна комбінація зсунутих синусів.

Оскільки синусоїди ортогональні, квадратична величина коефіцієнтів, що зважують синусоїди, пропорційна енергії кожної відповідної частоти синусоїди.

Коефіцієнт пропорційності залежить від способу обчислення коефіцієнтів, часто до масштабного коефіцієнта. Амплітуду певної частотної складової можна безпосередньо обчислити за допомогою кореляції. Зауважте, однак, що цей коефіцієнт слід сприймати з обережністю, оскільки зовнішні умови відіграють певну роль: обмежений горизонт спостереження функції/сигналу, дискретизація амплітуди, шуми тощо.
*** 


Why do we usually use only Amplitude from the STFT transform?  
This is convenient and understandable for humans, because we understand the concept of amplitude, which is the largest deviation of a periodically changing value from a certain value that is conventionally assumed to be zero.  We are more accustomed to the concept of understanding the change in amplitude, and it is also convenient to work with, representing the strength of different frequency components over time.  Another result of the Fourier transform, the phase of a signal, is convenient to work with in specific cases when we are interested in how long a signal is delayed or what, but this concept is not as familiar as the amplitude and is not used as often as it is and it is unsrtuctured and not used in analysis (spectrogram). 

***
Це зручно та зрозуміло для людини, тому що ми розуміємо концепт амплітуди — найбільше відхилення величини, яка періодично змінюється від деякого значення, умовно прийнятого за нульове.  Ми більш звичні до концепту розуміння зміни амплітуди, до того ж із нею зручно працювати, вона репрезентує силу різних частотних компонент у часі.  Із ще одним результатом перетворення Фур'є - фазою сигналу зручно працювати у конкретних випадках, коли нам цікаво, наскільки сигнал затримується або що, але цей концепт не такий звичний, як амплітуда і застосовується не так часто, як вона. 

A: ?? phase data is unstructured and not used in analysis (spectrogram)

*** 

Which type of filter would you use to attenuate frequencies above a certain threshold?
Low-pass filter


Why do we need augmentations in the modern Deep Learning era? Where can we use them? (propose as many applications, as possible)

When training a model, a problem may arise when it is trained on a “clean” dataset but used on noisy data, which worsens the quality of the performance. In addition, there may be insufficient data in the dataset, or there may be little amount of noisy data, which leads to training on an unbalanced dataset. If it is balanced with noisy data, then there will be no overfitting effect and the model will react normally to deviations in the data, so it will help to move more smoothly from the training to the working environment, also this will omprove model robustness. 
This can be used in Computer Vision, with flipping, dimming, rotating, contrasting images, changing lighting, adding filters (like in an instagram), in Audio Processing (like unexpected), adding noise to sound, increasing the duration of the noise or sound, compressing or enhancing the sound, recording the same sound in different environments (studio, street, bus, classroom, by the river, in the forest), speeding up the sound by 1. 25, 2 times the sound, and so on. It can also be used in Natural Language Processing - synonyms, sentence structure changes, different personality, incorrect accents, loanwords, anglicisms. 

***
Під час навчання моделі може виникати проблема, що її натренували на "чистому" датасеті, але застосовують на зашумлених даних, що погіршує якість виконання поставленої задачі. До того ж, даних у датасеті може бути недостатньо, може бути мало зашумлених даних, що призводить до тренування на незбалансованому датасеті. Якщо його збалансувати зашумленими даними, то не буде overfitting ефекту і модель буде нормально реагувати на відхилення в даних, тому це буде допомагати більш плавно переходити із навчального до робочого середовища. 
Це можна використовувати у сфері Комп'ютерного зору, із перевертанням, затемнянням, поворотом, контрастністю картинок, зміною освітлення, додавання фільтрів (як в інстраграмі), в Аудіо Обробці (як несподівано), додаванням шумів до звуку, збільшенням тривалості шуму чи звукового сигналу, стишенням чи підвищенням звукового сигналу, записами одного і того ж звукового сигналу в різних середовищах (студія, вулиця, автобус, аудиторія, біля річки, в лісі), пришвидшувати в 1.25, 2 рази звучання, тощо. Також це можна застосовувати у Обробці Природньої мови - синоніми, зміна структури речення, різна особовість, неправильні наголоси, запозичені слова, англіцизми. 

A: To increase training dataset size i.e. to prevent overfitting, to improve model robustness, to reduce biases for imbalanced datasets. Also it's a cheap method to collect more data.
*** 


What are the advantages and disadvantages of using higher order filters?

The lower the filter order, the more the boundary between the passband and the stopband will be blurred, which means that the filter will not effectively filter out unwanted frequencies and this will not lead to a “clean” result, but if you increase the filter order, it will sharply filter out unwanted frequencies, i.e. the result will be “cleaner” and better.  However,  they introduce more phase shift, leading to potential signal distortion.

***
Чим нижчий порядок фільтра, тим більше буде стиратись межа між смугою пропускання і стоп-смугою, що означає, що фільтр не буде ефективно відсіювати непотрібні частоти і це не призведе до "чистого" результату, проте, якщо підвищити порядок фільтра, то він буде різкіше відсіювати непотрібні частоти, тобто результат буде "чистішим" і кращим. 

A: ?? Higher order filters have steeper roll-off for better frequency attenuation.

However,  they introduce more phase shift, leading to potential signal distortion.
*** 
