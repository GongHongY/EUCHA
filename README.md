# Embedded U-shaped Network with Cross Hierarchical Feature Adaptation Fusion for Remote Sensing Image Dehazing
### Dependences
1.Pytorch 1.8.0  
2.Python 3.7.1  
3.CUDA 11.7  
4.Ubuntu 18.04    
### Datasets Preparation
> datasets_train
>> clean  
>> hazy

> datasets_test 
>> clean  
>> hazy

> output_result

### Pretrained Weights and Dataset  
Download our model weights on Baidu cloud disk:  
https://pan.baidu.com/s/1baDpj5YtT5DoP1tbe3DRTw?pwd=GHYU

Download our test datasets on Baidu cloud disk:  
https://pan.baidu.com/s/1TPIUhKPw4HwN52kLYgFGxQ?pwd=GHYU

### Train  
 `python train.py --type 0 -train_batch_size 4 --gpus 0 `

### Test
Put models in the `EUCHA/output_result` folder.   
`python test.py --type 0 --gpus 0  `
