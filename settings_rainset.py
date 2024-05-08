import os
import logging

patch_size = 64
# true when train, false when test.
pic_is_pair = False # input picture is pair or single

ssim_loss = True
aug_data = False

lr = 0.001

data_dir = 'your\\training\\dataset\\path\\'
test_data_dir = 'your\\testing\\dataset\\path\\'
if pic_is_pair is False:
    data_dir =  'your\\testing\\real\\dataset\\path\\'
log_dir = './logdir_MPRain'
model_dir = './trained_model/'

log_level = 'info'
model_path = os.path.join(model_dir, 'net_520_epoch')
save_steps = 113

num_workers = 8

num_GPU = 1
device_id = '0, 1'

epoch = 520
batch_size = 32

if pic_is_pair:
    root_dir = os.path.join(data_dir, 'Rain')
    mat_files = os.listdir(root_dir)
    num_datasets = len(mat_files)
    l1 = int(3/5 * epoch * num_datasets / batch_size) # 90000  # 6300
    l2 = int(4/5 * epoch * num_datasets / batch_size) # 120000 # 8400
    one_epoch = int(num_datasets/batch_size)
    total_step = int((epoch * num_datasets)/batch_size)

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


