'''
Created on Sep. 18, 2020
@author: hmo (hyunho.mo@unitn.it)
'''

import sys
import random
import numpy as np
import pandas as pd
import json
import time
from utils import network_launcher
from datetime import date
from network_training import network_train
log_path = 'log/'
fitness_mode_list = ["val_rmse", "aic", "val_score", "rmse_combined", "score_combined"]
fitness_mode = 2

log_file_path = log_path + 'log_temp.csv'
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only use the first GPU
#   try:
#     tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
#   except RuntimeError as e:
#     # Visible devices must be set before GPUs have been initialized
#     print(e)

print(sys.path)
# Assign CSV file path & name
tmp_path = 'tmp/'
# Assign columns name
# The columns indicate the type of measured physical properties in csv file
# It can be an input argument for later use (application)
num_sensors = 26
test_engine_idx = 45
cols = ['unit_nr', 'cycles', 'os_1', 'os_2', 'os_3']
cols += ['sensor_{0:02d}'.format(s + 1) for s in range(num_sensors)]
cols_non_sensor = ['unit_nr', 'cycles', 'os_1', 'os_2', 'RUL']
model_path = tmp_path + 'network/trained_opt_model-temp.h5'

# Other parameters
# window_length = None #GA parameter
# n_window = None #Internal variable calculated by n_window = int((sequence_length - window_length) / (stride) + 1)
# window_length = 2
# n_filters = 5
# kernel_size = 2
# n_conv_layer = 1
# LSTM1_ref = 290
# LSTM2_ref = 232

# window_length = 5
# n_filters = 8
# kernel_size = 5
# n_conv_layer = 1
# LSTM1_ref = 156
# LSTM2_ref = 104

pop_size = 50 # toy example
n_generations = 50 # toy example

# window_length = 1
# n_filters = 1
# kernel_size = 1
# n_conv_layer = 1
# LSTM1_ref = 4
# LSTM2_ref = 4

window_length = 5
n_filters = 10
kernel_size = 5
n_conv_layer = 2
LSTM1_ref = 20
LSTM2_ref = 15


# csv_filename = 'train_FD001.csv'  # user defined
# csv_test_filename = 'test_FD001.csv'  # user defined
# csv_rul_filename = 'RUL_FD001.txt'  # user defined
# csv_path = tmp_path + csv_filename
# RUL_FD_path = tmp_path + csv_rul_filename
# # Other parameters
# # window_length = None #GA parameter
# # n_window = None #Internal variable calculated by n_window = int((sequence_length - window_length) / (stride) + 1)
# sequence_length = 30
# n_channel = 1
# strides_len = 1
# n_outputs = 1
# cross_val = False
# k_value_fold = 10
# val_split = 0.1
# max_epoch = 20
# patience = 10
# bidirec = False
# stride = 1
# piecewise_lin_ref = 125
# batch_size = 400
# dropout = 0
#
# seed = 0
# individual_seed = [3, 2, 2, 10, 5]
#
#
# train_FD, cols_sensors = network_launcher().preprocessing_main(csv_filename, cols, cols_non_sensor)
# test_FD, cols_sensors = network_launcher().preprocessing_main(csv_test_filename, cols, cols_non_sensor, train=False)
#
#
#
#
# start = time.time()
#
#
# training_input, training_input_label = network_launcher().opt_network_input_generator(
#     dataframe_norm=train_FD,
#     cols_non_sensor=cols_non_sensor,
#     sequence_length=sequence_length,
#     stride=stride,
#     window_length=window_length,
#     piecewise_lin_ref=piecewise_lin_ref
# )
#
# test_input, test_input_label = network_launcher().rmse_test_input_generator(
#     dataframe_norm=test_FD,
#     cols_non_sensor=cols_non_sensor,
#     rul_file_path = RUL_FD_path,
#     sequence_length=sequence_length,
#     stride=stride,
#     window_length=window_length,
#     piecewise_lin_ref=piecewise_lin_ref
# )
#
# ## Generate the network and run the training and save the trained model into the file
# cnnlstm = network_launcher().network_training(training_input, training_input_label, cols_sensors, model_path,
#                                               fitness_mode, log_file_path,
#                                               test=False, test_engine_idx=None,
#                                               n_channel=1, n_filters=n_filters, strides_len=1,
#                                               kernel_size=kernel_size,
#                                               n_conv_layer=n_conv_layer, LSTM1_ref=LSTM1_ref,
#                                               LSTM2_ref=LSTM2_ref,
#                                               n_outputs=1, cross_val=False, k_value_fold=k_value_fold,
#                                               val_split=val_split,
#                                               batch_size=batch_size, max_epoch=max_epoch,
#                                               patience=patience, bidirec=False, dropout=0.5, experiment=True)
#
# rmse1, score1 = network_train().opt_network_test_rmse(cnnlstm, test_input, test_input_label, model_path,
#                                                     window_length,
#                                                     n_filters, kernel_size, n_conv_layer, LSTM1_ref, LSTM2_ref)
#
#
#
# print (rmse1, score1)
# end = time.time()
# print(end - start)

#
# csv_filename = 'train_FD002.csv'  # user defined
# csv_test_filename = 'test_FD002.csv'  # user defined
# csv_rul_filename = 'RUL_FD002.txt'  # user defined
# csv_path = tmp_path + csv_filename
# RUL_FD_path = tmp_path + csv_rul_filename
# # Other parameters
# # window_length = None #GA parameter
# # n_window = None #Internal variable calculated by n_window = int((sequence_length - window_length) / (stride) + 1)
# sequence_length = 21
# n_channel = 1
# strides_len = 1
# n_outputs = 1
# cross_val = False
# k_value_fold = 10
# val_split = 0.1
# max_epoch = 20
# patience = 10
# bidirec = False
# stride = 1
# piecewise_lin_ref = 125
# batch_size = 400
# dropout = 0
#
# seed = 0
# individual_seed = [4, 4, 1, 20, 10]
#
#
# train_FD, cols_sensors = network_launcher().preprocessing_main(csv_filename, cols, cols_non_sensor)
# test_FD, cols_sensors = network_launcher().preprocessing_main(csv_test_filename, cols, cols_non_sensor, train=False)
#
# training_input, training_input_label = network_launcher().opt_network_input_generator(
#     dataframe_norm=train_FD,
#     cols_non_sensor=cols_non_sensor,
#     sequence_length=sequence_length,
#     stride=stride,
#     window_length=window_length,
#     piecewise_lin_ref=piecewise_lin_ref
# )
#
# test_input, test_input_label = network_launcher().rmse_test_input_generator(
#     dataframe_norm=test_FD,
#     cols_non_sensor=cols_non_sensor,
#     rul_file_path = RUL_FD_path,
#     sequence_length=sequence_length,
#     stride=stride,
#     window_length=window_length,
#     piecewise_lin_ref=piecewise_lin_ref
# )
#
# ## Generate the network and run the training and save the trained model into the file
# cnnlstm = network_launcher().network_training(training_input, training_input_label, cols_sensors, model_path,
#                                               fitness_mode, log_file_path,
#                                               test=False, test_engine_idx=None,
#                                               n_channel=1, n_filters=n_filters, strides_len=1,
#                                               kernel_size=kernel_size,
#                                               n_conv_layer=n_conv_layer, LSTM1_ref=LSTM1_ref,
#                                               LSTM2_ref=LSTM2_ref,
#                                               n_outputs=1, cross_val=False, k_value_fold=k_value_fold,
#                                               val_split=val_split,
#                                               batch_size=batch_size, max_epoch=max_epoch,
#                                               patience=patience, bidirec=False, dropout=0.5, experiment=True)
#
# rmse2, score2 = network_train().opt_network_test_rmse(cnnlstm, test_input, test_input_label, model_path,
#                                                     window_length,
#                                                     n_filters, kernel_size, n_conv_layer, LSTM1_ref, LSTM2_ref)
#
#
#
#
#
# csv_filename = 'train_FD003.csv'  # user defined
# csv_test_filename = 'test_FD003.csv'  # user defined
# csv_rul_filename = 'RUL_FD003.txt'  # user defined
# csv_path = tmp_path + csv_filename
# RUL_FD_path = tmp_path + csv_rul_filename
# # Other parameters
# # window_length = None #GA parameter
# # n_window = None #Internal variable calculated by n_window = int((sequence_length - window_length) / (stride) + 1)
# sequence_length = 38
# n_channel = 1
# strides_len = 1
# n_outputs = 1
# cross_val = False
# k_value_fold = 10
# val_split = 0.1
# max_epoch = 20
# patience = 10
# bidirec = False
# stride = 1
# piecewise_lin_ref = 125
# batch_size = 400
# dropout = 0
#
# seed = 0
#
#
# train_FD, cols_sensors = network_launcher().preprocessing_main(csv_filename, cols, cols_non_sensor)
# test_FD, cols_sensors = network_launcher().preprocessing_main(csv_test_filename, cols, cols_non_sensor, train=False)
#
# training_input, training_input_label = network_launcher().opt_network_input_generator(
#     dataframe_norm=train_FD,
#     cols_non_sensor=cols_non_sensor,
#     sequence_length=sequence_length,
#     stride=stride,
#     window_length=window_length,
#     piecewise_lin_ref=piecewise_lin_ref
# )
#
# test_input, test_input_label = network_launcher().rmse_test_input_generator(
#     dataframe_norm=test_FD,
#     cols_non_sensor=cols_non_sensor,
#     rul_file_path = RUL_FD_path,
#     sequence_length=sequence_length,
#     stride=stride,
#     window_length=window_length,
#     piecewise_lin_ref=piecewise_lin_ref
# )
#
# ## Generate the network and run the training and save the trained model into the file
# cnnlstm = network_launcher().network_training(training_input, training_input_label, cols_sensors, model_path,
#                                               fitness_mode, log_file_path,
#                                               test=False, test_engine_idx=None,
#                                               n_channel=1, n_filters=n_filters, strides_len=1,
#                                               kernel_size=kernel_size,
#                                               n_conv_layer=n_conv_layer, LSTM1_ref=LSTM1_ref,
#                                               LSTM2_ref=LSTM2_ref,
#                                               n_outputs=1, cross_val=False, k_value_fold=k_value_fold,
#                                               val_split=val_split,
#                                               batch_size=batch_size, max_epoch=max_epoch,
#                                               patience=patience, bidirec=False, dropout=0.5, experiment=True)
#
# rmse3, score3 = network_train().opt_network_test_rmse(cnnlstm, test_input, test_input_label, model_path,
#                                                     window_length,
#                                                     n_filters, kernel_size, n_conv_layer, LSTM1_ref, LSTM2_ref)
#
#
#
#


start = time.time()
csv_filename = 'train_FD004.csv'  # user defined
csv_test_filename = 'test_FD004.csv'  # user defined
csv_rul_filename = 'RUL_FD004.txt'  # user defined
csv_path = tmp_path + csv_filename
RUL_FD_path = tmp_path + csv_rul_filename
# Other parameters
# window_length = None #GA parameter
# n_window = None #Internal variable calculated by n_window = int((sequence_length - window_length) / (stride) + 1)
sequence_length = 21
n_channel = 1
strides_len = 1
n_outputs = 1
cross_val = False
k_value_fold = 10
val_split = 0.1
max_epoch = 20
patience = 10
bidirec = False
stride = 1
piecewise_lin_ref = 125
batch_size = 400
dropout = 0

seed = 0



train_FD, cols_sensors = network_launcher().preprocessing_main(csv_filename, cols, cols_non_sensor)
test_FD, cols_sensors = network_launcher().preprocessing_main(csv_test_filename, cols, cols_non_sensor, train=False)

training_input, training_input_label = network_launcher().opt_network_input_generator(
    dataframe_norm=train_FD,
    cols_non_sensor=cols_non_sensor,
    sequence_length=sequence_length,
    stride=stride,
    window_length=window_length,
    piecewise_lin_ref=piecewise_lin_ref
)

test_input, test_input_label = network_launcher().rmse_test_input_generator(
    dataframe_norm=test_FD,
    cols_non_sensor=cols_non_sensor,
    rul_file_path = RUL_FD_path,
    sequence_length=sequence_length,
    stride=stride,
    window_length=window_length,
    piecewise_lin_ref=piecewise_lin_ref
)

## Generate the network and run the training and save the trained model into the file
cnnlstm = network_launcher().network_training(training_input, training_input_label, cols_sensors, model_path,
                                              fitness_mode, log_file_path,
                                              test=False, test_engine_idx=None,
                                              n_channel=1, n_filters=n_filters, strides_len=1,
                                              kernel_size=kernel_size,
                                              n_conv_layer=n_conv_layer, LSTM1_ref=LSTM1_ref,
                                              LSTM2_ref=LSTM2_ref,
                                              n_outputs=1, cross_val=False, k_value_fold=k_value_fold,
                                              val_split=val_split,
                                              batch_size=batch_size, max_epoch=max_epoch,
                                              patience=patience, bidirec=False, dropout=0.5, experiment=True)

rmse4, score4 = network_train().opt_network_test_rmse(cnnlstm, test_input, test_input_label, model_path,
                                                    window_length,
                                                    n_filters, kernel_size, n_conv_layer, LSTM1_ref, LSTM2_ref)

print (rmse4, score4)
end = time.time()
print(end - start)

#
#
#
#
# print (rmse1, score1)
# print (rmse2, score2)
# print (rmse3, score3)
# print (rmse4, score4)
#
# dataset = np.array(['fd001', 'fd002', 'fd003', 'fd004'])
# rmse_array = np.array([rmse1, rmse2, rmse3, rmse4])
# score_array = np.array([score1, score2, score3, score4])
# results_file_path = log_path + 'results_%s_pop-%s_gen-%s_(%s, %s, %s, %s, %s).csv' % (fitness_mode_list[fitness_mode], pop_size, n_generations, window_length, n_filters, n_conv_layer, LSTM1_ref, LSTM2_ref)
# results_col = ['dataset', 'rmse' ,'score']
# results_df = pd.DataFrame(columns=results_col, index=None)
# results_df['dataset'] = dataset
# results_df['rmse'] = rmse_array
# results_df['score'] = score_array
# results_df.to_csv(results_file_path, index=False)