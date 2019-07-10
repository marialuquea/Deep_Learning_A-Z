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
'''
    Taking the max number of the feature map to get a reduced feature map
    size of feature map is divided by 2 
    apply max pooling on each feature map 
    use to reduced the number of nodes for the fully connected layers, 
    reduce complexity and time execution without losing the performance
    bc by taking the max, we are keeping the information, keep track of the part of the image that
    contain the feature
    
    .add a max pooling layer
    pool_size: factors by which to downscale, recommended to use 2,2
'''
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))



# Step 3 - Flattening
'''
    Apply flattening step to get a huge single vector that contains the different cells
    of all the feature maps, this vector will be the input layer of the future ANN that has 
    the fully connected layers
    
'''
classifier.add(Flatten())



# Step 4 - Full connection
'''
    We converted the input image into a 1D vector containing infor about the spatial
    structure of the image.
    1. Use this input vector as input layer of classic ANN.
    2. Fully connected layer = hidden layer
    3. Add an output layer of just 1 node because this is a binary outcome (dog/cat)
    Dense is the function used to add a fully connected layer
    
    Choose the number of units between number of inputs and number of outputs
    The choice of number results from experimentation, 
        - if too small, classifier will be a bad model
        - if too big, highly computational intensive
    Pick a power of 2 (not 100 but 128)
    
    Activation function: the nodes in the hidden layer that we're adding are like neurons
    that we need to activate according to how much the can pass on the signal, according 
    on how much they are relevant to pass on their contribution to the final vote.
    
    For the output layer uses the sigmoid function becuase we have a binary outcome. If it 
    has more outcomes, use the Softmax activation function. 1 node that is the predicted 
    probability of 1 class.
    
'''
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid')) # output layer



# Compiling the CNN
'''
    - optimizer: to choose the stochastic gradient descent algorithm, adam algorithm
    - loss: to choose the loss function, log loss for classification problems 
    - metrics: to choose the performance metric
'''
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])










# Part 2 - Fitting the CNN to the images
# taken from keras documentation Image Prepocessing
from keras.preprocessing.image import ImageDataGenerator

'''
    image augmentation part, where we apply several transformations like the rescale
    rescale is always compulsory, corresponds to the feature scanning part of the data preprocessing
    phase: rescale all pixel values between 0 and 1 bc pixels take values between 0 and 255 
    other transformations like shear range: shearing, geometrical transformation that is also called
    transvection
    zoom range: random zoom
    horizontal flip or vertical flip
'''
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

'''
    create the training/test set composed of the augmented images extracted from ImageDataGenerator
'''
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64), #size of images expected in the CNN model
                                                 batch_size = 32, # number of images going through the CNN after which the weights will be updated
                                                 class_mode = 'binary') # if ouput is binary or more outputs

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
'''
    fit CNN model to the training set and test its performance on the test set
'''
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000, # bc we have 8000 images in training dataset
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000) # bc we have 2000 images in test dataset