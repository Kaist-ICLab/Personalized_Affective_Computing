This is a supplementary repository for the paper titled **Systematic Evaluation of Personalized Deep Learning Models for Affect Recognition**.

## RUN
1. Execute `requirements.txt` and `setup.py` to install the necessary packages.
2. Save the data in the `archives` folder.
3. Run `ar_dataset_preprocessing.py` for the desired dataset preprocessing. The processed data will be saved in `mts_archive`.
4. Run `./datasetnametuning.sh X` in the desired folder (X: id of GPU).
5. Execute `datasetnameresults.py`.

## STRUCTURE

#### archives Folder
- You need to create a `data` folder manually.
  -  The top-level folders contain raw data, and `mts_archive` contains data after each preprocessing.

#### DATASET_PREPARE Folder
- We have to format all datasets into the same structure as the WESAD dataset.
  - In each `Si` folder, have a file for each participant in `.pkl`.
  - In each `.pkl` file, label and sensor signal data are in `numpy.array` format.

#### arpreprocessing Folder
- When you run `ar_dataset_preprocessing.py`, the codes inside this folder will be executed.
- The main files are `datasetname.py`, which perform winsorization, filtering, resampling, normalization, and windowing and also formatting the dataset for deep learning models.
  - For datasets without user labels, we use `preprocessor.py` and `subject.py`, while those with labels, `preprocessorlabel.py` and `subjectlabel.py` are used.

#### GeneralizedModel Folder
- Functions in the `multimodal_classifiers` folder are used for model training.
  - For each deep learning structure (i.e., Fully Convolutional Network (FCN), Residual Network (ResNet), and Multi-Layer Perceptron with LSTM (MLP-LSTM)), non-personalized models are implemented.
- For a detailed explanation of model implementation, please refer to section 3.3 Non-Personalized Model.

#### PersonalizedModel_FineTuning Folder
- Functions in the `multimodal_classifiers_finetuning` folder are used for model training.
  - For each deep learning structure, personalized models with fine-tuning are implemented.
- For a detailed explanation of model implementation, please refer to section 3.4.1 Unseen User-Dependent Fine-Tuning part.

#### PersonalizedModel_Hybrid Folder
- Functions in the `multimodal_classifiers_hybrid` folder are used for model training.
  - For each deep learning structure, hybrid (partially-personalized) models are implemented.
- For a detailed explanation of model implementation, please refer to section 3.4.1 Unseen User-Dependent Hybrid part.

#### PersonalizedModel_ClusterSpecific Folder
- Functions in the `multimodal_classifiers` folder and `clustering` folder are used for model training.
  - As explained in section 3.4.2 Unseen User-Independent, the difference between generalized model and cluster-specific personalized model is the data used for training, not the model itself.
    - Therefore, we use the same functions in the `multimodal_classifiers` folder as in generalized models.
  - Using functions in the `clustering` folder, trait-based clustering is done and its result is used for model training.

#### PersonalizedModel_MTLNN Folder
- Functions in the `multimodal_classifiers_mtl` folder and `clustering` folder are used for model training.
  - As explained in section 3.4.2 Unseen User-Independent, multi-task learning personalized models differ from generalized models in both the data used for training and the model itself.
  - Therefore, we use the functions in the `multimodal_classifiers_mtl` folder.
- Also, using functions in the `clustering` folder, trait-based clustering is done for multi-task learning models.

## Acknowledgments
Codes for non-personalized models, i.e., `arpreprocessing`, `GeneralizedModel`, and `multimodal_classifiers` folder, are based on code provided at the "dl-4-tsc" GitHub repository. [https://github.com/Emognition/dl-4-tsc](https://github.com/Emognition/dl-4-tsc)

The datasets used are as follows, and they can be downloaded from the provided links:
  1. **AMIGOS**: [AMIGOS: A Dataset for Affect, Personality and Mood Research on Individuals and Groups](http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/)
  2. **ASCERTAIN**: [ASCERTAIN: Emotion and Personality Recognition Using Commercial Sensors](https://ascertain-dataset.github.io/)
  3. **CASE**: [A dataset of continuous affect annotations and physiological signals for emotion analysis](https://gitlab.com/karan-shr/case_dataset)
  4. **WESAD**: [WESAD: Multimodal Dataset for Wearable Stress and Affect Detection](https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/)
  5. **K-EmoCon**: [K-EmoCon, a multimodal sensor dataset for continuous emotion recognition in naturalistic conversations](https://zenodo.org/records/3814370)
  6. **K-EmoPhone**: [K-EmoPhone, A Mobile and Wearable Dataset with In-Situ Emotion, Stress, and Attention Labels](https://doi.org/10.5281/zenodo.7606611)
