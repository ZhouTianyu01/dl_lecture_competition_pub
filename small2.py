# make smaller dataset to test
import torch
import numpy as np


full_X_train = torch.load('data/train_X.pt')
full_y_train = torch.load('data/train_y.pt')
full_subject_idxs_train = torch.load('data/train_subject_idxs.pt')

full_X_val = torch.load('data/val_X.pt')
full_y_val = torch.load('data/val_y.pt')
full_subject_idxs_val = torch.load('data/val_subject_idxs.pt')

full_X_test = torch.load('data/test_X.pt')
full_subject_idxs_test = torch.load('data/test_subject_idxs.pt')


print(f"Number of training samples: {len(full_X_train)}, shape: {full_X_train.shape}")
print(f"Number of validation samples: {len(full_X_val)}, shape: {full_X_val.shape}")
print(f"Number of test samples: {len(full_X_test)}, shape: {full_X_test.shape}")


num_classes = 1854  
num_samples_per_class_train = 1  

small_X_train = []
small_y_train = []
small_subject_idxs_train = []

for cls in range(num_classes):
    cls_indices = np.where(full_y_train.numpy() == cls)[0]
    selected_indices = np.random.choice(cls_indices, num_samples_per_class_train, replace=False)
    small_X_train.extend(full_X_train[selected_indices].clone())
    small_y_train.extend(full_y_train[selected_indices].clone())
    small_subject_idxs_train.extend(full_subject_idxs_train[selected_indices].clone())

small_X_train = torch.stack(small_X_train)
small_y_train = torch.tensor(small_y_train)
small_subject_idxs_train = torch.tensor(small_subject_idxs_train)


num_samples_per_class_val = 1 

small_X_val = []
small_y_val = []
small_subject_idxs_val = []

for cls in range(num_classes):
    cls_indices = np.where(full_y_val.numpy() == cls)[0]
    selected_indices = np.random.choice(cls_indices, num_samples_per_class_val, replace=False)
    small_X_val.extend(full_X_val[selected_indices].clone())
    small_y_val.extend(full_y_val[selected_indices].clone())
    small_subject_idxs_val.extend(full_subject_idxs_val[selected_indices].clone())

small_X_val = torch.stack(small_X_val)
small_y_val = torch.tensor(small_y_val)
small_subject_idxs_val = torch.tensor(small_subject_idxs_val)


small_X_test = full_X_test[:50, :, :].clone() 
small_subject_idxs_test = full_subject_idxs_test[:50].clone() 

torch.save(small_X_train, 'data/small_train_X.pt')
torch.save(small_y_train, 'data/small_train_y.pt')
torch.save(small_subject_idxs_train, 'data/small_train_subject_idxs.pt')

torch.save(small_X_val, 'data/small_val_X.pt')
torch.save(small_y_val, 'data/small_val_y.pt')
torch.save(small_subject_idxs_val, 'data/small_val_subject_idxs.pt')

torch.save(small_X_test, 'data/small_test_X.pt')
torch.save(small_subject_idxs_test, 'data/small_test_subject_idxs.pt')
