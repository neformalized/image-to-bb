# image-to-bb

A simple and fast pipeline for training models that convert images to bounding box coordinates.
The pipeline is easy to configure, you can quickly and easily swap in your own dataset and model. Just edit a few settings, and it's ready to run, even in Google Colab.


Try it yourself: [Google Colab Example](https://colab.research.google.com/drive/197-yMBIJNYKqiZvkiz-WuZR2u5u5-XUr?usp=sharing)

#

## Requirements

    tensorflow, numpy, cv2

#

## Configuration

Edit the `/source/init.py` file for pipeline-specific configuration:

    dataset_path_train = ["path-to-train-txt", "path-to-train-folder"]  # Paths to the training dataset
    dataset_path_validation = ["path-to-validation-txt", "path-to-validation-folder"]  # Paths to the validation dataset
    #dataset_path_validation = False  # Uncomment if no validation dataset is available
    buffer_size = 2000  # Size of the training buffer; set to 0 to load the whole dataset at once (requires robust hardware)
    input_image = (139, 132)  # Input image shape
    output_labels = 4  # Number of output labels
    loss_target = 0.05  # Target loss value to stop training
    save_path = "/content/"  # Path to save the model and logs

#

## Learning Loop

The learning loop is straightforward to modify:

`/source/hood/handler.py - start()`

    while True:
        self.fit()                # Fit the full training dataset
        self.evaluate()           # Evaluate the validation dataset
        self.checkpoint()         # Save model weights
        self.write_log()          # Write epoch log
        if self.is_done(): break  # Check break condition
        self.updates()            # Update variables, model, etc.

#

## Pipeline Modules

The pipeline modules are easy to modify:

    /source/hood/buffer.py   # Transformed x, y data holder class
    /source/hood/builder.py  # Keras model structure
    /source/hood/dataset.py  # Dataset holder class
    /source/hood/handler.py  # Main pipeline class
    /source/hood/image.py    # Model input encoder
    /source/hood/label.py    # Model output encoder
    /source/hood/log.py      # Logging class

#

## Report

`/source/report.py` simply visualizes the model's predictions using the validation dataset (requires matplotlib):

#

[LinkedIn Profile](https://www.linkedin.com/in/sergey-syschenko-027b01318/)

Feel free to contact me if you have any questions!
