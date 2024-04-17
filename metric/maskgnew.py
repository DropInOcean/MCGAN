import os
import scipy.io
import numpy as np
from skimage import io


#Name of transfer value
Tr_value = 'T1_T2'
#Name of Generate value folder
Gv_folder = 'Simons'

#Name of GT value folder
#GT_folder = 'HCP'



# 输入文件夹路径，包含图片的.mat文件
#image_folder = '/data1/wgw/test_one_data_pgan_cgan/results/pgan_result/HCP_T1_T2star/test_latest/images/'
# 输入文件夹路径，包含掩码的.mat文件
#mask_folder = '/DATA2024/wsl/2024Metrics/Matmask/HCP/csf/'
# 输出文件夹路径，用于保存.mat文件和处理后的图像
#output_folder = '/DATA2024/wsl/2024Metrics/resultapgan/HCP/T1_T2star/csf/'

image_folder = '/data1/wgw/test_one_data_pgan_cgan/results/cgan_result/' + Gv_folder +'_' + Tr_value + '/test_latest/images/'
mask_folder = '/DATA2024/wsl/2024Metrics/Matmask/' + Gv_folder + '/csf/'
output_folder = '/DATA2024/wsl/2024Metrics/result_cgan/' + Gv_folder + '/' + Tr_value + '/csf/'

print(image_folder)
print(mask_folder)
print(output_folder)


'''Remember to modify the field below'''

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 获取文件夹中的文件列表，并按文件名排序
image_files = sorted(os.listdir(image_folder))
mask_files = sorted(os.listdir(mask_folder))

for image_file in image_files:
    for mask_file in mask_files:
        #if image_file[:10] == mask_file[:10] and image_file.endswith('.mat') and mask_file.endswith('.mat'):
        #if image_file[1:41].replace('T2star_', '') == mask_file[:37].replace('T1w_','') and image_file.endswith('.mat') and mask_file.endswith('.mat'):
        #if image_file[:32].replace('-PD', '') == mask_file[:29] and image_file.endswith('.mat') and mask_file.endswith('.mat'):
        #if image_file[:27].replace('-PD', '') == mask_file[:24] and image_file.endswith('.mat') and mask_file.endswith('.mat'):
        #if image_file[:29] == mask_file[:29] and image_file.endswith('.mat') and mask_file.endswith('.mat'):
        #print(image_file[:-11])
        if image_file == mask_file and image_file.endswith('.mat') and mask_file.endswith('.mat'):
            # 加载图像和掩码
            image_data = scipy.io.loadmat(os.path.join(image_folder, image_file))['data']
            #mask_data = scipy.io.loadmat(os.path.join(mask_folder, mask_file))['HCPcsf']
            mask_data = scipy.io.loadmat(os.path.join(mask_folder, mask_file))[Gv_folder + 'csf']

            # 在这里进行你的处理，例如将图像和掩码相乘
            processed_image = image_data * mask_data

            # 保存为.mat文件
            mat_output_file = os.path.join(output_folder, f'processed_{image_file}')
            #scipy.io.savemat(mat_output_file, {'T1_T2star_csf': processed_image})
            scipy.io.savemat(mat_output_file, {Tr_value + '_csf': processed_image})
            print(f"处理后的.mat文件已保存为 {mat_output_file}")

            # 将浮点数图像转换为8位图像（0-255范围）
            processed_image_uint8 = (processed_image * 255).astype(np.uint8)

            # 构建输出图像文件路径
            png_output_file = os.path.join(output_folder, f'processed_{image_file[:-4]}.png')

            # 保存为 PNG 文件
            io.imsave(png_output_file, processed_image_uint8)
            print(f"处理后的PNG文件已保存为 {png_output_file}")


