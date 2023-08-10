# DreamText

We introduce DreamText, an image-to-3D generative
model that leverages text descriptors as an intermediary
step for 3D reconstruction. Without 3D training data. Our
approach involves training a neural radiance field, which
subsequently enables the rendering of novel views of the
reconstructed 3D object which appears in the image. To
accomplish this we employ a pretrained diffusion model to
generate diverse object views, which serve as training data
for this neural radiance field. To condition the generated
views to closely resemble the image object, we pretrain the
conditioning input, the text, of the diffusion model. Thus as
the 3D regeneration comes from ”dreaming” novel views
using pretrained text, we call our model DreamText.

Samples can bee seen at this [page](https://nazarpuriy.github.io/).

## Code replication
This repository is based on this other [repository](https://github.com/ashawkey/stable-dreamfusion). Before working on Stable Diffusion, one should ensure that their computer has the CUDA libraries installed and a graphics card with at least 12 GB of memory in order to successfully run the code. Some libraries have dependencies on the others and the requirements.txt, obtained with pip freeze, might not work properly. We recommend to run it one time and then install the missing libraries manually. Also  the exact version of CUDA should be checked, and torch libraries that you need for your GPU. Our experiments were performed using Ubuntu 22.04 wich Nvidia driver version: NVIDIA-SMI 525.125.06 driver CUDA Version: 12.0 
\newline

There are two scripts that has to be executed to run our code: to train the text embedding and to make the 3D reconstruction once the text embedding is trained.  The following code contains two imperative variables: \textit{WORKSPACE} and \textit{PATH}. First is used to indicate where all the generated data will be stored, second is used to indicate where the training image is located. Besides, for DreamText, we recommend to add noun and quantifier parameters to specify this values to the network, although they can be automatically predicted. The following snipet of code generates the sentence that describes the image and trains it:

~~~python
python ./descriptor.py --workspace WORKSPACE --noun watch --quantifier a --image_path PATH
~~~

Once the sentence is generated and trained the 3D reconstruction can be made. It is imperative to provide the same workspace directory for this. The following snipet of code reconstructs the 3D object after training the descriptive sentence:

~~~python
python main.py --workspace WORKSPACE -O --h 256 --w 256 --iters 10100 --load_token 1
~~~


Samples can be visualized in a graphical user interface using:

~~~python
python main.py -O --test --gui --workspace WORKSPACE
~~~

We have uploaded 3 samples: examples/example1, examples/example2 and examples/example3. Change the WORKSPACE in the code above by one of these two directories to visualize them.
