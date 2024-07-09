EfficientNet model ('efficientnet-b0') has been used in the assignment. 

It is a pretrained model which follows a compound scaling method, where the depth, width, and resolution of the network are scaled simultaneously. 'efficientnet-b0' is the base model in the EfficientNet family.

The num_classes=1 parameter used to initialize the model indicates that the final layer is adapted for regression tasks, outputting a single value, in the assignment, the predicted age. The learning rate has been set as 0.001.

The following command is required in the terminal to install the EfficientNet model:

pip install efficientnet-pytorch