from scipy import misc
import numpy as np
import os

root_path = '/data/wei/dataset/MDetection/ICPR2012/training_data/scanner_A'
raw_img_path = os.path.join(root_path, 'bmp')
raw_img_files = sorted(os.listdir(raw_img_path))

label_path = os.path.join(root_path, 'label_plot')
label_files = sorted(os.listdir(label_path))

# print(raw_img_files)
# print(label_files)
path_size = [96, 96]
step_size = [16, 16]


for idx, _  in enumerate(raw_img_files):
    
    fid = 0
    name_prefix = raw_img_files[idx].split('.')[0]
    raw = misc.imread(os.path.join(raw_img_path, raw_img_files[idx]))
    label = misc.imread(os.path.join(label_path, label_files[idx]))
    raw_shape = raw.shape
    
    for xx in range(0, raw_shape[0] - path_size[0], step_size[0]):
        
        x_start = xx
        x_end = xx + path_size[0]
        for yy in range(0, raw_shape[1] - path_size[1], step_size[1]):

            y_start  = yy
            y_end = yy +  path_size[1]
            img_patch = raw[x_start:x_end, y_start:y_end,:]
            label_patch = label[x_start:x_end, y_start:y_end]
            islabeled = True if np.mean(np.mean(label_patch)) else False
           
            if islabeled:
                save_Path = './dataset/ICPR2012/abnormal'
            else:
                save_Path = './dataset/ICPR2012/normal'
            
            filename = name_prefix + '_'+ str(fid) + '.bmp'
            save_Path_img = os.path.join(save_Path, 'data', filename)
            save_Path_label = os.path.join(save_Path, 'label', filename)

            misc.imsave(save_Path_img, img_patch)
            misc.imsave(save_Path_label, label_patch)

            fid+=1
            print('save path:{}'.format(fid))

            