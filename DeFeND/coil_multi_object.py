
import os

import numpy as np
from PIL import Image

def generate_dataset(src_path, out_path, normal_class_grp, num_generated_samples=10000, total_categories=72):
    """
    This function generates the dataset for the multi-object classification task.
    It takes the path to the source dataset and the path to the output directory as input.
    It also takes the normal class group and the total number of categories as input.
    It returns the path to the generated dataset.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Create the directories for the normal and anomalous classes
    if not os.path.exists(out_path + '/train'):
        os.makedirs(out_path + '/train')
    if not os.path.exists(out_path + '/train/normal'):
        os.makedirs(out_path + '/train/normal')
    if not os.path.exists(out_path + '/test'):
        os.makedirs(out_path + '/test')
    if not os.path.exists(out_path + '/test/normal'):
        os.makedirs(out_path + '/test/normal')
    if not os.path.exists(out_path + '/test/anomalous'):
        os.makedirs(out_path + '/test/anomalous')
    train_output_path = out_path + '/train'

    test_output_path = out_path + '/test'
    
    abnormal_class_grp = [i for i in range(1, total_categories + 1) if i not in normal_class_grp]
    generate_samples(src_path, train_output_path, normal_class_grp, abnormal_class_grp, len(normal_class_grp), num_generated_samples, total_categories)
    generate_samples(src_path, test_output_path, normal_class_grp, abnormal_class_grp, 2, num_generated_samples // 2, total_categories)
    generate_samples(src_path, test_output_path, normal_class_grp, abnormal_class_grp, 0, num_generated_samples // 2, total_categories)


def generate_samples(src_path, out_path, normal_class_grp, abnormal_class_grp, num_normal, num_generated_samples, total_categories):
    locations = [(0, 0), (0, 128), (128, 0), (128, 128)]
    for i in range(num_generated_samples):
        image_list = []
        combined_image = Image.new('RGB', (256, 256), (0, 0, 0))
        sampled_normal_classes = np.random.choice(normal_class_grp, num_normal, replace=False)
        sampled_abnormal_classes = np.random.choice(abnormal_class_grp, 4 - num_normal, replace=False)
        for j in sampled_normal_classes:
            rand_num = np.random.randint(0, total_categories)
            sample_name = "obj" + str(j) + "__" + str(rand_num * 5) + ".png"
            sample_path = os.path.join(src_path, sample_name) 
            image = Image.open(sample_path)
            image_list.append(image)
        for j in sampled_abnormal_classes:
            rand_num = np.random.randint(0, total_categories)
            sample_name = "obj" + str(j) + "__" + str(rand_num * 5) + ".png"
            sample_path = os.path.join(src_path, sample_name) 
            image = Image.open(sample_path)
            image_list.append(image)
        ## permute the order of the images and put them in any of 128*128 squares in the combined image
        np.random.shuffle(image_list)
        np.random.shuffle(locations)
        for k in range(len(image_list)):
            combined_image.paste(image_list[k], locations[k])
        if num_normal == 0:
            save_path = os.path.join(out_path + "/anomalous", str(i) + '.png')
        else:
            save_path = os.path.join(out_path + "/normal", str(i) + '.png')
        combined_image.save(save_path)


generate_dataset('data/coil-100/coil-100', 'data/coil-100', [1, 2, 3], 10000, 72)
    
    
        

