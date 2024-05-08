import os
import logging

patch_size = 128
pic_is_pair = False  # input picture is pair or single
ssim_loss = True
aug_data = False

lr = 0.0001
data_dir = 'your\\training\\dataset\\path\\'
test_data_dir = 'your\\testing\\dataset\\path\\'
real_dir = 'your\\training\\real\\dataset\\path\\'
if pic_is_pair is False:
    data_dir =  'your\\testing\\real\\dataset\\path\\'
log_dir = './logdir_MPRain_Real'
model_dir = './trained_model/'
show_dir_feature = '../showdir_feature'

log_level = 'info'
model_path = os.path.join(model_dir, 'net_latest')
save_steps = 645 
num_workers = 8

num_GPU = 2
device_id = '0,1'

epoch = 20
batch_size = 16

if pic_is_pair:
    root_dir = os.path.join(data_dir, '\\Rain')
    # root_dir = data_dir
    mat_files = os.listdir(root_dir)
    num_datasets = len(mat_files)
    l1 = int(3/5 * epoch * num_datasets / batch_size) # 90000
    l2 = int(4/5 * epoch * num_datasets / batch_size) # 120000
    one_epoch = int(num_datasets/batch_size)
    total_step = int((epoch * num_datasets)/batch_size)

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


