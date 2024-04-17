# MCGAN

## Data
Prepare your dataset under the directory , data is '.mat', if you data is '.nii' you can use /data/slice to slice data
  * Directory structure on new dataset needed for training and testing, such as T1/T2:
    * datasets/train/T1
    * data/train/T2
    * data/test/T1
    * data/test/T2 

## Domain Adaptation

### Training
sh DA.sh
<br />

## Image to Image

### Training
sh I2I.sh
<br />

## Testing

### Training
sh DA_infer.sh
<br />

## Image to Image

### Training
sh I2I_infer.sh
