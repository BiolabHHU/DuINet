import os
from .dataset_rgb import DataLoaderTrain, DataLoaderVal, DataLoaderTest, DataLoaderTestSR, DataLoaderTrainGT, DataLoaderValGT, DataLoaderTestGT

def get_training_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, img_options, None)

def get_validation_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, None)


def get_test_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, None)


def get_test_data_SR(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderTestSR(rgb_dir, None)

def get_train_clean_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrainGT(rgb_dir, img_options, None)

def get_val_clean_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderValGT(rgb_dir, None)

def get_test_clean_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderTestGT(rgb_dir, None)




