import os
import shutil
data_dir='data'
try:
    shutil.rmtree(rf'.\{data_dir}\train')
except: 
    pass
os.makedirs(rf'.\{data_dir}\train')
try:
    shutil.rmtree(rf'.\{data_dir}\val')
except: pass
os.makedirs(rf'.\{data_dir}\val')
from random import sample
val_ppt=0.2
for subdir in os.walk(rf'.\{data_dir}'):
    if len(subdir[2])==0 or 'train' in subdir[0] or 'val' in subdir[0]:
        continue
    rel_path = os.path.relpath(subdir[0], data_dir)
    train_dir = os.path.join(data_dir, 'train', rel_path)
    val_dir = os.path.join(data_dir, 'val', rel_path)
    os.makedirs(train_dir)
    os.makedirs(val_dir)
    num=len(subdir[2])
    val_idx=set(sample(range(num),int(num*val_ppt)))
    for idx, file in enumerate(subdir[2]):
        if idx in val_idx:
            shutil.copy(os.path.join(subdir[0],file),os.path.join(val_dir,file))
        else:
            shutil.copy(os.path.join(subdir[0],file),os.path.join(train_dir,file))


        
    