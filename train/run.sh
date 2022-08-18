%run -i setup_libraries
%run -i setup_device
%run -i setup_parameters

%run -i build_utils/generic_utils
%run -i build_utils/data_utils
%run -i build_utils/training_utils

%run -i setup_data
%run -i setup_train
%run -i train