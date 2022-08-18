%run -i setup_libraries
%run -i setup_device
%run -i setup_parameters

%run -i train/build_utils/generic_utils
%run -i train/build_utils/data_augmentation
%run -i train/build_utils/data_utils
%run -i train/build_utils/training_utils

%run -i train/setup_data
%run -i train/setup_train
%run -i train/train