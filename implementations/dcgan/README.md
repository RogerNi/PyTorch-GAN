# How to run

The following arguments (a subset of all arguments) can be used to control the behavior of training.

|Parameters|Explanations|Default values|
|---|---|---|
|sample_saving_delay|number of epochs to delay before starting to save samples|0 (no delay)|
|new_weight_ratio|This argument is used to control how the new weights are initialized. Details can be found from line 82 to 85 in codes. Valid range is [0, 1], where 0 means new initialized weights for saved samples are very low and the weight for generator does not change. 1 means that the weight for generator will be completely distributed to new saved samples. 0.5 means that the weight for generator will be evenly distributed to saved samples such that the weight for generator and the weights for new saved samples are the same.|0.5 (evenly distributed)|
|skip_weights|Whether to skip optimizing weights (weights do not change)|False (optimizing weights)|
|min_gen_weight|The lower bound of raw sampling weight of the generator|`torch.finfo().min` (minimal possible value that can be represented in a PyTorch tensor)|
|min_gen_norm_weight|The lower bound of normalized sampling weight of the generator, valid range: [0, 1)|0|
|policy_loss|Whether to use policy gradient loss instead of binary cross entropy loss|False|

Here are some sample commands to run:

- Enable all features: `python gan.py`
- Assign very low value to new weights: `python dcgan.py --new_weight_ratio 0`
- Assign very low value to new weights and stop optimizing sampling weights: `python dcgan.py --new_weight_ratio 0 --skip_weights`
- Disable GPU for better debugging messages: `python dcgan.py --disable_gpu`
- Do not save samples until the 20th epoch: `python dcgan.py --sample_saving_delay 20`

# Some default training hyper-parameters:

|Parameters|Default values|
|---|---|
|Number of epochs|200|
|Batch size|64|
|Learning rate (for all optimizations)|0.0002|
|Adam $\beta_1$|0.5|
|Adam $\beta_2$|0.999|

# Outputs

The following files will be created after completion of training:

|Files|Descriptions|
|---|---|
|`[TIMESTAMP]-images-sampled-on-prob`|Folder containing saved samples|
|`[TIMESTAMP]-loss_lists`|Pickled list of losses of each epoch (`[g_loss_list, d_real_loss_list, d_fake_loss_list, w_loss_list, w_grad_sum_list]`)|
|`[TIMESTAMP]-loss_plot.svg`|Plot showing how losses change over epochs|
|`[TIMESTAMP]-weights.pkl`|Pickled list of final weights for all saved samples and the generator. The generator weight is at index 0|

# Experiment results (so far)

Training using these two commands output very similar results as the original GAN:- Assign very low value to new weights and stop optimizing sampling weights: `python dcgan.py --new_weight_ratio 0 --skip_weights`

Run our method:
```
python dcgan.py --new_weight_ratio 0.001 --policy_loss True
```