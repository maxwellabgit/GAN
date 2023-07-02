#lets draw some numbers
import keras
import matplotlib.pyplot as plt
import random
import numpy as np

(trainX, trainy), (testX, testy) = keras.datasets.mnist.load_data()

#Some random cases if you want to take a look:
"""
c = 1
for i in random.sample(range(0, len(trainX)), 25):
    plt.subplot(5, 5, c)
    plt.axis('off')
    plt.imshow(trainX[i], cmap='gray_r')
    c = c + 1
plt.show()
"""

#Discriminator
def discriminatingModel():
    m = keras.models.Sequential()
    m.add(keras.layers.Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=(28,28,1)))
    m.add(keras.layers.LeakyReLU(alpha=0.2))
    m.add(keras.layers.Dropout(0.4))
    m.add(keras.layers.Conv2D(64, (3,3), strides=(2, 2), padding='same'))
    m.add(keras.layers.LeakyReLU(alpha=0.2))
    m.add(keras.layers.Dropout(0.4))
    m.add(keras.layers.Flatten())
    m.add(keras.layers.Dense(1, activation='sigmoid'))
    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    m.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return(m)

discModel = discriminatingModel()
discModel.summary()

#Generator

#latent space at 100 dimensions, an opportunity for fine-tuning
def generateLatentSpace(nSamples, dim=100):
    sampledPoints = np.random.randn(dim * nSamples)
    #Reshape for the net.
    reshape = sampledPoints.reshape(nSamples, dim)
    return reshape

def generatorModel(dim=100):
    model = keras.models.Sequential()
	
    #Starting with a 7x7x128 vector, sampled to 28x28x1 at the end with a convolution.
    model.add(keras.layers.Dense(128 * 7 * 7, input_dim=dim))
	
    #The discriminator backwards
    model.add(keras.layers.LeakyReLU(alpha=0.2))
    model.add(keras.layers.Reshape((7, 7, 128)))
	
    #A transposed 2D Convolution - a filter with weights that vary, that are used to
    #"upsample" each input pixel. Weights are fit to create more resolvable images.
    model.add(keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(keras.layers.LeakyReLU(alpha=0.2))

    model.add(keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(keras.layers.LeakyReLU(alpha=0.2))
    model.add(keras.layers.Conv2D(1, (7,7), activation='sigmoid', padding='same'))
    return model

genModel = generatorModel()
genModel.summary()

#Stick the generator and discriminator networks together
def GAN(genModel, discModel):
	#Turn off the ability to update the weights in the discriminating model
    discModel.trainable = False
	
    m = keras.models.Sequential()
    
    m.add(genModel)
    m.add(discModel)

    #Note - some of the images that are fake are labeled as real to get our loss function working properly.
    #i.e., loss depends on whether or not we fooled the discriminator
    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    m.compile(loss='binary_crossentropy', optimizer=opt)
    return(m)

ganModel = GAN(genModel, discModel)
ganModel.summary()

#Load our Real Data:
def loadData():
    (trainX, _), (_, _) = keras.datasets.mnist.load_data()
    #Add a dimension to the data - we need 28,28,1
    X = np.expand_dims(trainX, axis=-1)

    #Next, we're going to convert from integers to floats.
    #Ultimately we want data scaled between 0 and 1 for input into our net.
    X = X.astype('float32')

    #Now we scale - our image data is from 0 to 255 in it's raw form.
    X = X / 255.0

    return(X)

#Admin steps:
def generateRealSamples(dta, nSamples):
    #Choose random indices from input data
    rnd = np.random.randint(0, dta.shape[0], nSamples)
    X = dta[rnd]
    #1 for "real" images
    y = np.ones((nSamples, 1))
    return(X,y)

def generateFakeSamples(genModel, nSamples, dim=100):
    #Sample our latent space (the 100 dim vector)
    inputData = generateLatentSpace(dim=dim, nSamples=nSamples)

    #Predict the output at that space
    X = genModel.predict(inputData)

    #0 for "fake" images
    y = np.zeros((nSamples, 1))
    return(X,y)

#helper for viz
def saveResults(genModel, epoch, dim=100, size=5):
    #Generate a few fake samples
    fakeImages, _ = generateFakeSamples(genModel, dim=dim, nSamples=size*size)

    for i in range (size*size):
        plt.subplot(size, size, 1+i)
        plt.axis('off')
        #Inverse the colors to make it match
        plt.imshow(fakeImages[i,:,:,0], cmap="gray_r")
    outFile = "./mnistGen/epoch_" + str(epoch) + ".png"
    plt.savefig(outFile)
    plt.close()

#Start drawing
def train(genModel, discModel, ganModel, dta, dim=100, epochs=20, batchSize=1024):
    #Calculate the number of batches required to complete an epoch until model runs through all data
    totalBatches = int(dta.shape[0] / batchSize)

    for i in range(epochs):
        print("Starting Epoch " + str(i))
        for j in range(totalBatches):
            print("       Batch " + str(j))
            #Generate real samples
            xReal, yReal = generateRealSamples(dta, batchSize)

            #Generate fake samples
            xFake, yFake = generateFakeSamples(genModel, nSamples = batchSize, dim=100)

            #Stack for discriminator training
            X, y = np.vstack((xReal, xFake)), np.vstack((yReal, yFake))

            #train
            discLoss, _ = discModel.train_on_batch(X,y)

            #Update generator
            inputSample = generateLatentSpace(nSamples=batchSize, dim=100)

            #Remember - the GAN trainer thinks all the data is real, so that when the discriminator labels them all as true,
            #we know that we've fooled it (0 loss). If it labels them all as false, we get maximum loss.
            inputY = np.ones((batchSize, 1))

            #Train the generator via the GAN (discriminating model is frozen here)
            ganLoss = ganModel.train_on_batch(inputSample, inputY)

        #Save outputs every epoch
        saveResults(genModel, i)

#%reset to clear past model states, if you run this cell more than once
train(genModel, discModel, ganModel, dta=loadData(), dim=100, epochs=100, batchSize=256)
