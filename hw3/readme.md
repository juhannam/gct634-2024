# Homework #3: Symbolic Music Generation


Symbolic music generation is a task that generates a score-level music data using a generative model.  In this homework, you will implement the [PerformanceRNN model](https://magenta.tensorflow.org/performance-rnn), a simple RNN-based musical language model with the following goals:

* Experiencing the pipeline of training and evaluate a deep learning model for symbolic music generation
* Learning a MIDI-like tokenization method to handle performance MIDI files 
* Implementing a simple language model and sampling methods including temporature control 


## Dataset
We use [Saarland Music Data](https://resources.mpi-inf.mpg.de/SMD/SMD_MIDI-Audio-Piano-Music.html) where 50 piano performances are recorded as audio-MIDI pairs. We will use the MIDI files only for this homework. 

## Baseline: a LSTM-based language model 
We provide a Python notebook file [GCT634-HW3.ipynb](https://colab.research.google.com/drive/1pGIaIs33swOilhIw8zjB-VGcVQxuweph?usp=sharing) which includes all components for data preparation, building, training, and evaluting a baseline model. The baseline model is a simple language model with two LSTM layers that uses a MIDI-like tokenization method. You can train and evalute the model by simply executing the cells one by one in the Python notebook file. 

## Question 1: Review the MIDI-like tokenization method (5pts)
Your first task is to understand the MIDI-like tokenization method. It is implemented in the following class (See the MIDI-Like Tokenizer cell).

```
class EventSeq:
    pitch_range = range(21, 109)
    velocity_range = range(21, 109)
    velocity_steps = 32
    time_shift_bins = 1.15 ** np.arange(32) / 65

    @staticmethod
    def from_note_seq(notes):
        note_events = []
        if USE_VELOCITY:
            velocity_bins = EventSeq.get_velocity_bins()

...

```
Read the EventSeq class and the following execution code that encodes the MIDI file to token indices and decodes the token indices back to MIDI. Write a detailed description of the MIDI-like tokenization including your answers to the following questions (but not limited to them):
- What is the size of the entire token vocabulary ?
- Why is the timeshift tokens implemented in an exponentially increasing way? (time_shift_bins = 1.15 ** np.arange(32) / 65 )
- The decoded MIDI from the tokens are somewhat different from the original MIDI. Why?

## Question 2: Implement the softmax temporature (2pts)
You can train the LSTM-model with the baseline code and generate the music. Note that the baseline model was designed to overfit to a single piece of the piano music and generate a new piece similar to that. Run the code and get the generated music. How does it sound like? We will control the output by adding the temporatue to the softmax output, which is supposed to control the diversity. Modify the following snippet of code (in the "Build the Model" cell) to add the softemax temporatue.  
```
def forward(self, x, hidden, tau):
        # TO DO: incoporate tau into the following code 
        x_encoder = self.encoder(x)
        x_encoder, x_hidden = self.rnn(x_encoder)
        x_decoder = self.decoder(x_encoder)
        x_pred = self.log_softmax(x_decoder)

        return x_pred, x_hidden
```
Try different values of tau (for example, tau = 0.1, 0.5, 0.8, 1.0, 1.2, 1,5, 2,0) and compare the results by listening to the output and observing the statistics of pitch, steps, and duration. 

## Question 3: Implement the primed generation (3pts)
You can use a primer to continute the music generation from the given piece of music. Implement the primed music generation by filling in the following code. 
```
def test_primer(self, prime, sequence, tau):
    # TO DO: implement the primed generation

    with torch.no_grad():
        self.model.eval()
        preds = []

        return preds
```
Note that the continuation result of the baseline code is not so good. Don't be disappointed with it.  

## Question 4: Improve the result (5pts + extra pts)
Now, you can use improve the quality of music generation in your own way. The followings are possible ideas that you can try: 

- Search better hyperparameters (RNN configuration, optimizers) 
- Use more MIDI files [Saarland Music Data](https://resources.mpi-inf.mpg.de/SMD/SMD_MIDI-Audio-Piano-Music.html). You can simple upload MIDI files into the "gct634-SMD-MIDI" folder 
- Use MIDI data augmentation techniques such as transpose or tempo change  
- Use different random sampling methods such as Top-K, Top-p sampling
- Renovate the MIDI-like tokenization, particularly to address the bad note duration estimation  
- Find bugs in the baseline code :)

