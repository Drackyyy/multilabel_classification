<!--
 * @Author: your name
 * @Date: 2021-05-30 14:51:36
 * @LastEditTime: 2021-05-30 15:13:31
 * @LastEditors: your name
 * @Description: In User Settings Edit
 * @FilePath: /pg-task11/project/README.md
-->
# Code of semantic type detection
This is code for multilabel semantic type detection subtask for V-NLI. Predicted labels will be used for selecting proper forms of visualization downstream.

## Prepare Dataset
Training and validation set have been neatened to use. For inference, dataset should be organized in cvs format as training and validation set. To arrange raw data into desired one, run following script:
```
python data.py
```
## Model Training
We organize training process in transformer learner pipeline. Learning rate and training epoch are two hyperparameters to optimize. Since save_model function hasn't been embedded into training process (and can be complicated to do it), only model at the end of training will be saved. Thus a recommended training strategy is starting with few epochs and save, then restart at that saved model incrementally.
To train the model, run following script:
```
python -m torch.distributed.launch --nproc_per_node= #GPU main.py
```
Note #GPU needs to be consistent with os.environ['CUDA_VISIBLE_DEVICES'] in main.py.

## Training outcomes
Training outcomes, including curves and saved model, will be stored at ./learner_cl_output.

## Inference
For inference, just run following script:
```
python prediction.py
```
Just substitute contents to be classified in the script.


## WARNING: NEVER TRY TO MODIFY SCRIPTS RELATED TO PARALLEL COMPUTING IN LEARNER_CLS.PY!!!
Since it has cost me half a day to correct this code and the relevant orders. 
