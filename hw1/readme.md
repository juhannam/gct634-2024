# Homework #1: Music Classification (Due date: Mar 31)


## Part 1: Musical Instrument Recognition with Traditional Machine Learning 
Musical instrument recognition is a fundamental task in understanding music by computers. Your first mission is developing your own algorithm based on the traditional machine learning approach. Specifically, the goals of this homework are as follows: 
- Experiencing the whole pipeline of a machine learning task: data preparation, feature extraction, training learning models and evaluation 
- Using the Librosa and Scikit-learn libraries in practice 
- Analyzing different characteristics of musical instrument tones and extracting them in a numerical form

### Dataset
We use a subset of the [NSynth dataset](https://magenta.tensorflow.org/datasets/nsynth) which is a large collection of musical instrument tones from the Google Magenta project. The subset has 10 classes of different musical instruments, including bass, brass, flute, guitar, keyboard, mallet, organ, reed, string and vocal. For our expriment, it is split into training, validation and test sets. For each class, the training set has 110 audio samples and the validation set have 30 audio samples. You can download the subset [here](). 

Make './dataset/' directory, and download the dataset to './dataset/'.
Once you downloaded the dataset, make sure that you have the following files and folders.  

```
$ cd dataset
$ ls
train train_list.txt valid valid_list.txt
```

### Baseline Code
The Python notebook file for the baseline algorithm ([GCT634-HW1-part1.ipynb](https://colab.research.google.com/drive/1ljALxWTA0qaMAQgew_0JRUR4XJ4i0KrN?usp=sharing)) is provided so that you can easily start with the homework and also compare your own algorithm to it in performance. The baseline model extracts MFCC, summarizes them by taking temporal average for each audio file and use a linear SVM model for classification.  

If the run is successful, it will display the validation accuracy values.  

```
alpha=0.0001, validation acc=69.33
alpha=0.0010, validation acc=74.00
alpha=0.0100, validation acc=72.00
alpha=0.1000, validation acc=68.00
alpha=1.0000, validation acc=60.67
```

### Improving Algorithms
Now it is your turn. You should improve the baseline code by developing your own algorithm. There are many ways to improve it. The followings are possible ideas: 

* Check the difference between musical instrument sounds carefully by observing sounds as a waveform and spectrogram and at the same time listening to them. This will give you a lot of insight. Audacity is a convenient tool for this purposes. 
* Think about the length of audio clips that you will use (You don't have use the entire length of audio files) 
* Try different MFCC parameter settings: mel-bin size and DCT size.
* Add delta and double-delta of MFCCs (time-wise difference of MFCC)
* Add other audio features: spectral statitsics, temporal envelope (e.g. ADSR), and so on.
* Codebook-based feature summarization.
* Try different classifiers: k-NN, SVM with nonlinear kernels, MLP, GMM, ...

## Part 2: Music Auto-Tagging with Deep Learning (Will be released by March 17)






## Deliverables
You should submit your Python code (.py files) and homework report (.pdf file) to KLMS. The report should include:
* Algorithm Description
* Experiments and Results
* Discussion
