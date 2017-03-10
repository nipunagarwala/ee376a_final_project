# EE376a Final Project

MRI Undersampled dataset link: http://mridata.org/undersampled

Data repo: /afs/ir.stanford.edu/users/n/i/nipuna1/EE376A

## algorithms.py
Basic framework for your algorithm implementation. The basic equation is `y=Ax`.

### Parameters
  * `self.input_func` : Data generation function
  * `self.input_func_args` : Tuple passed in as the argument 'params' to @input_func
  * `self.A_func` : Defines how the `A` matrix is generated
  * `self.N` : Size of `x`
  * `self.M` : Size of `y`

### Functions
  * `generateData()` : Generates the data (`self.x`) using `self.input_func` and `self.input_func_args`. Also generates `self.A` using `self.A_func`. Then calculates `self.y` by `np.matmul(self.A, self.x)`.
  * `predict_long_wav_data()` : Applies prediction on sliding windows of the loaded real music. The results are concatenated later and outputted to a file.
  * `predict()` : The implementation of the algorithm.
  * `predict_perf()` : A wrapper around `predict()` that measures performance metrics like accuracy and time elapsed.
  * `plot()` : Plots `self.x` and `self.x_pred`.
  * `eval_error()` : Calculates the MSE and sparsity error metrics.

## test.py
Used to test the algorithms.

  * `test_sound()` : Generates sound according to `soundType`, and applies the algorithm on it. Saves the result in .wav format. Testing is repeated `repNum` times.
  * `test_real_song()` :  Loads 10 seconds of 8000Hz music (`dataset/wind_lq_predicted.wav`), applies algorithm on windows of size alg.N, and outputs the result in a .wav file. 
  * `test_random_noise_img()` : Uses the data generation function as defined prior to this function call (set `alg.input_func` and `alg.input_func_args`), and creates an image of it for easier inspection of the performance. Testing is repeated `repNum` times.
  * `test_real_img()` : Loads a real image as defined in @img_name, and applies the algorithm. Set `use_fft` to True/False to convert the image to frequency domain (i.e. apply np.fft.fft2()).
  * `test_any()` : General testing module. Manually set the data generation method. (set `alg.input_func` and `alg.input_func_args`) Testing is repeated `repNum` times.

## data_gen.py
Generates the data. `size` and `params` are passed in for all functions, where `size` is an integer defining the size of the generated output, and `params` characterizes the distribution to be sampled from.

  * `gen_sound_dataset()` : Generates a sound dataset with notes as defined in `params`.
    * `params` = `(note_list, fs)`
      * `note_list` : an array of tuples of (`amplitude`, `note frequency`, `phase`)
      * `fs` : sampling frequency

  * `gen_random_sound_dataset()` : Generates a sound dataset with 1 to 4 random notes between C3-B4 (defined in `NOTES`). 
    * `params` = `fs`
      * `fs` : sampling frequency

  * `gen_verdu_dataset()` : Generates a dataset in which each element's distribution as defined by Verdu et al. Basically a Bernouilli distribution with normal distribution instead of 1. 
    * `params` = `(gamma,sigma,mean)`
      * `gamma` : Bernouilli probability 
      * `sigma, mean` : Parametrizes the normal distribution
 
  * `gen_basic_dataset()` : Generates a dataset of 0 and 1, with `K` 1's. 
    * `params` = `K`
      * `K` : Number of 1's 
 
  * `gen_random_dataset()` : Generates a dataset in which each element is drawn from a Gaussian distribution. Not sparse. 
    * `params` = `(mu, sig)`
      * `mu, sig` : Parametrizes the normal distribution