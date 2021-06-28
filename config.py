# │   config.py
# │   dataset.py
# │   engine.py
# │   model.py
# │   test.py
# │   train.py
# ├───checkpoints
# ├───input
# │   │   PotholeDataset.pdf
# │   │   train_df.csv
# │   └───Dataset 1 (Simplex)
# │       └───Dataset 1 (Simplex)
# │           ├───Test data
# │           └───Train data
# │               ├───Negative data
# │               └───Positive data
# ├───test_predictions

ROOT_PATH = 'input/Dataset 1 (Simplex)/Dataset 1 (Simplex)'
TEST_PATH = 'input/Dataset 1 (Simplex)/Dataset 1 (Simplex)/Test data'
PREDICTION_THRES = 0.8
EPOCHS = 5
MIN_SIZE = 800
BATCH_SIZE = 2
DEBUG = False # to visualize the images before training
