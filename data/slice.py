import os
import nibabel as nib
import numpy as np
from scipy.io import savemat

input_folder_1 = '/data5/wgw/pGAN-cGAN-master/train/t1'
input_folder_2 = '/data5/wgw/pGAN-cGAN-master/train/t2'

# 输出.mat文件的路径和名称
output_mat_file = '/data5/wgw/pGAN-cGAN-master/train/output/output_test_normal.mat'  # 自定义输出文件位置和名称

# 选择要处理的切片范围（例如，从第5层到第15层）
start_slice = 30  # 起始层索引，注意Python索引从0开始
end_slice =  110 # 结束层索引，包含在范围内


# 初始化一个空数组来存储切片
slices_1 = []
slices_2 = []

result_slices_1 = []
result_slices_2 = []

file_list_1 = os.listdir(input_folder_1)
file_list_1.sort()

file_list_2 = os.listdir(input_folder_2)
file_list_2.sort()

for filename in file_list_1:
    if filename.endswith('.nii.gz'):
        input_path = os.path.join(input_folder_1, filename)
        nii_img = nib.load(input_path)
        nii_data = nii_img.get_fdata()
        
        print(input_path)
        print(nii_data.shape)

        # 切片并归一化每一层
        num_slices = end_slice - start_slice + 1
        
        for i in range(num_slices):
            single_slice = nii_data[:, :, start_slice + i]
            
           
            mean_value = np.mean(single_slice)
            std_value = np.std(single_slice)
            single_slice[single_slice > mean_value + 3 * std_value] =  mean_value + 3*std_value
            max_value = np.max(single_slice)
            normalized_slice = single_slice / max_value
            normalized_slice=normalized_slice.astype(np.float32)
            
            result_slices_1.append(normalized_slice)
            
        for i in range(num_slices - 2):
            combined_slice = np.stack([result_slices_1[i], result_slices_1[i + 1], result_slices_1[i + 2]], axis=-1)
            slices_1.append(combined_slice)
            
        print(len(slices_1))
        

            
            
        
for filename in file_list_2:
    if filename.endswith('.nii.gz'):
        input_path = os.path.join(input_folder_2, filename)
        nii_img = nib.load(input_path)
        nii_data = nii_img.get_fdata()
        
        print(input_path)
        print(nii_data.shape)

        # 切片并归一化每一层
        num_slices = end_slice - start_slice + 1
        
        for i in range(num_slices):
            single_slice = nii_data[:, :, start_slice + i]
            
            mean_value = np.mean(single_slice)
            print(mean_value)
            std_value = np.std(single_slice)
            print(std_value)
            single_slice[single_slice > mean_value + 3 * std_value] =  mean_value + 3*std_value
            max_value = np.max(single_slice)
            normalized_slice = single_slice / max_value
            normalized_slice=normalized_slice.astype(np.float32)
            
            result_slices_2.append(normalized_slice)
    
        for i in range(num_slices - 2):
            combined_slice = np.stack([result_slices_2[i], result_slices_2[i + 1], result_slices_2[i + 2]], axis=-1)
            slices_2.append(combined_slice)
            
        print(len(slices_2))     
        

slices_1=np.transpose(slices_1,(1,2,0,3))
slices_2=np.transpose(slices_2,(1,2,0,3))

data_x= np.array(slices_1)
data_y= np.array(slices_2)


data_dict = {'data_x': data_x, 'data_y': data_y}

savemat(output_mat_file, data_dict)

print("切块已保存为", output_mat_file)
