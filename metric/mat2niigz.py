import os
import numpy as np
from scipy.io import loadmat
import nibabel as nib

def mat_to_nii(mat_path, nii_path):
    # Load .mat file
    mat_data = loadmat(mat_path)

    # Assuming your image data is stored in a variable called 'image'
    image_data = mat_data['T1']

    # Create a Nifti image
    nii_img = nib.Nifti1Image(image_data, np.eye(4))

    # Save the Nifti image
    nib.save(nii_img, nii_path)

def batch_convert_mat_to_nii(mat_folder, nii_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(nii_folder):
        os.makedirs(nii_folder)

    # List all .mat files in the input folder
    mat_files = [f for f in os.listdir(mat_folder) if f.endswith('.mat')]

    # Convert each .mat file to .nii.gz
    for mat_file in mat_files:
        mat_path = os.path.join(mat_folder, mat_file)

        # Construct the corresponding .nii.gz file path
        nii_file = os.path.splitext(mat_file)[0] + '.nii.gz'
        nii_path = os.path.join(nii_folder, nii_file)

        # Perform the conversion
        mat_to_nii(mat_path, nii_path)

# Example usage
mat_folder_path = '/DATA2023/wgw/DATASETS_all/IXI_data/test/T1_rotated'
nii_folder_path = '/DATA2024/wsl/T1/IXI/'
batch_convert_mat_to_nii(mat_folder_path, nii_folder_path)