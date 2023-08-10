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

Samples can bee seen at this [page](https://nazarpuriy.github.io/)
