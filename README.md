# SemEval2023_QUST

This is the code for the [SemEval 2023 shared task: Detecting the categroy, the framing, and the persuasion techniques in online news in a multi-lingual setups](https://propaganda.math.unipd.it/semeval2023task3/teampage.php?passcode=6a5411b7a5bcecd3654a611b6b1b667d) by the team ``QUST``:

* [Ye Jiang](https://ye-jiang.com)

He is an Assistant Professor of the Qingdao University of Science and Technology (QUST).

The system created with this was the 2nd best system in Italian and Spanish on subtask 1, see the public [leaderboard](https://propaganda.math.unipd.it/semeval2023task3/leaderboard.php)

# Preparation / Requirements

* python 3.9 or above
* pytorch 1.13.0.dev20220730	
* transformers	4.21.0
* datasets	2.4.0
* pandas	1.4.3
* scikit-learn	1.1.2
* tqdm	4.64.0

# To run

1. `Utils` folder contains data preprocessing scripts for each subtask separately. e.g., `train_data_task1.py`is used for the training and dev data in the subtask-1. (**Note that the training and dev data are merged as we implement 10-fold cross validation.**)

2. `train_pred` folder contains train and predict scripts for each subtask separately. e.g., `t1_kfold.py` will train the preprocessed data from above step through a 10-fold cross validation setup. We also applies early stopping and only save the best model checkpoint from the 10-fold.

3. after training, the prediction scripts is combining the top 3 best checkpoints to make a average ensemble for the test data. e.g., 't1_pred.py' will load the top 3 best checkpoint (the selection of the top 3 checkpoints are made manually by checking the training log once the training phase is done), and generate the prediction `.txt` file for each language in each subtask.
