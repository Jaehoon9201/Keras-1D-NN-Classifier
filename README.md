# Keras-1D-NN-Classifier

This code is based on the reference codes linked below.

[reference 1](https://keras.io/getting_started/intro_to_keras_for_researchers/), [reference 2](https://www.linkedin.com/pulse/multi-task-supervised-unsupervised-learning-code-ibrahim-sobh-phd/)

This code is for **1-D array data** classification.

The given data in 'data' directory is simple data for training and testing this code.

## About this code

This code is iterated by changing the'learn rate' variable to find the optimal learning rate.
The related part is the code below.

```python
var = [ 4e-5,8e-5, 12e-5]
for i in range(len(var)):

    var_str = 'lr replay %d th' % i
    dense1 = 16
    dense2 = 16
    train_epoch = 160
    batch_size = 300
    classes = 7
    learn_rate = var[i]
    
```

In addition, it monitors during learning through the code below and stops learning when there is no improvement in accuracy.
```python
checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='auto', period=1)
earlystopping = EarlyStopping(monitor='val_acc', patience=50)
```
## How the model is saved

![image](https://user-images.githubusercontent.com/71545160/117923585-40630c80-b32f-11eb-99fd-e6cc752835e2.png)

![image](https://user-images.githubusercontent.com/71545160/117923568-33deb400-b32f-11eb-9041-c4bfc14a5e4e.png)

![image](https://user-images.githubusercontent.com/71545160/117923512-1ad60300-b32f-11eb-9d4a-144446eb0e21.png)

![image](https://user-images.githubusercontent.com/71545160/117923537-26292e80-b32f-11eb-9748-f14e4e1756fc.png)

**Size of each data class imbalance should be modified**
