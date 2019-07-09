# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
'''
    Keras is an amazing library for deep learning and computer vision, it contains
    tricks and tools to import images in an efficient way.
    We only need to have a good structure for the images and Keras will understand
    how to differentiate the labels of the different variables. 
'''
from keras.models import Sequential #initialises nn bc there are 2 ways of initialising (sequence of layers or graph)
from keras.layers import Conv2D # first step, convolution step where we add the convolutional layers, 2D for images, 3D for videos (time)
from keras.layers import MaxPooling2D #step 2: pooling, add pooling layers
from keras.layers import Flatten #step 3: flattening, convert pooled feature maps into a large feature vector that the becomes the input of the fully connected layers
from keras.layers import Dense #add fully connected layers in a classic ANN

# Initialising the CNN
classifier = Sequential() 

# Step 1 - Convolution
'''
    Input image, applying several feature detectors, slide it on top of the whole image, 
    get a feature map where the highers number represents that the featre exists. 
    Input image x feature detector = feature map
    Many feautre detectors --> Many feature maps
    
    .add a convolutional layer
    function to use: Conv2D
        - 32: number of filter/number of feature maps, common practice to start with 32 in the first layer
        - (3, 3): number of rows and columns of the feature detector table
        - input_shape: the shape of the input image on which the feature detectors are applied through the convolution operation
          (64, 64, 3) coloured images (3 - red, blue, green) of 64x64 pixels, the bigger the number the longer it takes to process
        - rectivier activation function to get non-linearity, remove negatives
'''
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])





# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)