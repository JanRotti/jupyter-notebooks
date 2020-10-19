class GAN():
  def __init__(self,config = None):
    self.Name = "EBGAN" # Energy Based GAN
    self.m = 5 # Ressource https://github.com/wiseodd/generative-models/blob/master/GAN/ebgan/ebgan_tensorflow.py
    # Input Shape
    self.ImgHeight = 28 
    self.ImgWidth = 28
    self.ImgChannels = 1
    self.ImgShape = (self.ImgHeight, self.ImgWidth, self.ImgChannels)
    
    # Hyperparameter Assignment
    if config != None:
      self.GenFMaps = config["GenFMaps"]          # Base Feature Maps in Generator
      self.DisFMaps = config["DisFMaps"]          # Base Feature Maps in Discriminator
      self.Dropout = config["Dropout"]            # Dropout Rate
      self.Epochs = config["Epochs"]              # Training Epochs
      self.BatchSize = config["BatchSize"]        # Batch Size
      self.GDimension = config["GDimension"]      # Initial Reconstruction Dimension in Generator
      self.LatentDim = config["LatentDim"]        # Latent Input Vector Size for Generator
      self.LearningRate = config["LearningRate"]  # Learning Rate for Optimizers
    else:
      self.GenFMaps = 256 # must be divisible by 4
      self.DisFMaps = 64
      self.Dropout = 0.3
      self.Epochs = 30
      self.BatchSize = 64
      self.GDimension = 7 # Generator latent dimension
      self.LatentDim = 128
      self.LearningRate = 0.001

    # Visualization Parameters
    self.SampleInterval = 10 
    self.ControlSizeSQRT = 6
    self.SavePath = '/content/drive/My Drive/GeneratedImages'
    self.ImagesSavedCount = 0
    
    # Creating Generator and Discriminator Attribute
    self.Generator = None
    self.Discriminator = None
    
    # Loss Logging
    self.DiscLoss = 0
    self.GenLoss = 0
    self.DiscLosses = []
    self.GenLosses = []

    # Create Generator and Discriminator Models
    self.Generator = self.CreateGenerator()
    self.Discriminator = self.CreateDiscriminator() 

    # Define Optimizers
    self.GenOptimizer = tf.keras.optimizers.Adam(self.LearningRate)
    self.DiscOptimizer = tf.keras.optimizers.Adam(self.LearningRate)
    

  def CreateDiscriminator(self):
    if self.Discriminator:
      return self.Discriminator
    
    # Encoder
    self.Discriminator = Sequential(name="Discriminator")
    
    self.Discriminator.add(Conv2D(self.DisFMaps, (4, 4),
                                  strides=(2, 2), padding="same",
                                  input_shape = self.ImgShape))
    self.Discriminator.add(LeakyReLU())
    self.Discriminator.add(Dropout(self.Dropout))

    self.Discriminator.add(Conv2D(2 * self.DisFMaps, (4, 4),
                                  strides=(2, 2), padding="same"))
    self.Discriminator.add(LeakyReLU())
    self.Discriminator.add(Flatten())

    # Latent Representation
    self.Discriminator.add(Dense(self.LatentDim))
    
    # Decoder
    self.Discriminator.add(Dense(self.GDimension * self.GDimension * self.GenFMaps,
                             input_shape=(self.LatentDim,)))
    
    self.Discriminator.add(BatchNormalization())
    self.Discriminator.add(LeakyReLU())
    self.Discriminator.add(Reshape((self.GDimension, self.GDimension, self.GenFMaps)))
    self.Discriminator.add(Conv2DTranspose(self.GenFMaps / 2, 5, strides = 1,
                                  padding = "same"))
    
    self.Discriminator.add(BatchNormalization())
    self.Discriminator.add(LeakyReLU(alpha = 0.2))
    self.Discriminator.add(Conv2DTranspose(self.GenFMaps / 4, 5, strides = 2,
                                  padding = "same"))
    
    self.Discriminator.add(BatchNormalization())
    self.Discriminator.add(LeakyReLU(alpha = 0.2))

    self.Discriminator.add(Conv2DTranspose(1, 5, strides = 2,
                                  padding = "same", activation = "sigmoid"))
    assert self.Discriminator.output_shape == (None, 28, 28, 1)
    
    return self.Discriminator

  
  def CreateGenerator(self):
    if self.Generator:
      return self.Generator
    
    self.Generator = Sequential(name="Generator")
    self.Generator.add(Dense(self.GDimension * self.GDimension * self.GenFMaps,
                             input_shape=(self.LatentDim,)))
    
    self.Generator.add(BatchNormalization())
    self.Generator.add(LeakyReLU())
    self.Generator.add(Reshape((self.GDimension, self.GDimension, self.GenFMaps)))
    self.Generator.add(Conv2DTranspose(self.GenFMaps / 2, 5, strides = 1,
                                  padding = "same"))
    
    self.Generator.add(BatchNormalization())
    self.Generator.add(LeakyReLU(alpha = 0.2))
    self.Generator.add(Conv2DTranspose(self.GenFMaps / 4, 5, strides = 2,
                                  padding = "same"))
    
    self.Generator.add(BatchNormalization())
    self.Generator.add(LeakyReLU(alpha = 0.2))

    self.Generator.add(Conv2DTranspose(1, 5, strides = 2,
                                  padding = "same", activation = "sigmoid"))
    assert self.Generator.output_shape == (None, 28, 28, 1)
    return self.Generator

  # Loss Functions
  def GeneratorLoss(self, PredLabels):
    """
    https://arxiv.org/pdf/1609.03126.pdf%C2%A0
    """
    return tf.reduce_mean(PredLabels)

  def DiscriminatorLoss(self, RealOutput, FakeOutput):
    return tf.reduce_mean(RealOutput + tf.math.maximum(0.,self.m - FakeOutput))
  
  # Help Function for Autoencoder Output
  def MSE(self, Input, Reconstruction):
    return tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(Input, Reconstruction)), axis = 1))
  
  # Training Procedure
  @tf.function
  def TrainingStep(self,RealImages):
    # Generate Latent Vectors for Generator
    Noise = np.random.random(size = (self.BatchSize, self.LatentDim))

    # Automatic Gradient Calculation with GradientTape:    
    with tf.GradientTape() as GenTape, tf.GradientTape() as DiscTape:
      # Generate Images from Generator with Noise
      GeneratedImages = self.Generator(Noise,training = True)

      # Train Discriminator Model on Real and Generated
      RealOutput = self.MSE(RealImages, self.Discriminator(RealImages, training = True))
      FakeOutput = self.MSE(GeneratedImages, self.Discriminator(GeneratedImages, training = True))

      # Calculate Discriminator Loss    
      DiscLoss = self.DiscriminatorLoss(RealOutput, FakeOutput)

      # Calculate Generator Loss
      GenLoss = self.GeneratorLoss(FakeOutput) 

    # Calculate Gradients from "Recorded Operations"
    GradGenerator = GenTape.gradient(GenLoss, self.Generator.trainable_variables)
    GradDiscriminator = DiscTape.gradient(DiscLoss, self.Discriminator.trainable_variables)

    # Apply Gradients on Models
    self.GenOptimizer.apply_gradients(zip(GradGenerator,self.Generator.trainable_variables))
    self.DiscOptimizer.apply_gradients(zip(GradDiscriminator,self.Discriminator.trainable_variables))

    return (DiscLoss,GenLoss)


  def Training(self, RealDataset, Normalization = "norm"):
    TrainingData = tf.cast(RealDataset,'float32')
    
    # Normalizing Dataset
    if Normalization == True:
      TrainingData = TrainingData / 127.5 - 1 
    if Normalization == "norm":
      TrainingData = TrainingData / 255 

    # Adversarial Ground Truths for notion
    # When Ground Truths are changed the Loss functions have to be re-thought!!!
    # ValidLabels = np.ones((self.BatchSize, 1)) # Labels for Real
    # GeneratedLabels = np.zeros((self.BatchSize, 1)) # Labels for Fake

    # Training Loop
    for Epoch in tqdm(range(self.Epochs)):
      for BatchNumber in range(len(RealDataset)// self.BatchSize):      
        # Select Random Training Batch from Real Data
        Indexes = np.random.randint(0, TrainingData.shape[0], self.BatchSize)
        RealImages = tf.gather(TrainingData,Indexes)
        
        #  Combined Training Step
        self.DiscLoss, self.GenLoss = self.TrainingStep(RealImages)

        # Log Losses to Attributes
        self.DiscLosses.append(self.DiscLoss.numpy())
        self.GenLosses.append(self.GenLoss.numpy())

      # Sampling Images at Epoch End
      if (Epoch+1) % self.SampleInterval == 0:
        self.SampleImages()



  def SampleImages(self, seed = None):
    
    # Sampling Function Creating a ControlSizeSQRT * ControlSizeSQRT Gird Image
    if self.SavePath == None:
      return
    if not os.path.exists(self.SavePath):
      os.mkdir(self.SavePath)
    
    # Create Placeholder Image Canvas
    ControlImage = np.zeros((self.ImgWidth * self.ControlSizeSQRT,
                              self.ImgHeight * self.ControlSizeSQRT,
                              self.ImgChannels))
    
    # Generate Latent Vectors for Image Generation
    ControlVectors = np.random.normal(size = (self.ControlSizeSQRT**2, self.LatentDim))
    
    # Generate Control Images
    TestImages = self.Generator.predict(ControlVectors)
    
    # Positioning Images on Image Canvas
    for ImgNumber in range(self.ControlSizeSQRT**2):
      XOffset = ImgNumber % self.ControlSizeSQRT
      YOffset = ImgNumber // self.ControlSizeSQRT
      ControlImage[XOffset * self.ImgWidth : (XOffset + 1) *self.ImgWidth,
                   YOffset * self.ImgHeight : (YOffset + 1) *self.ImgHeight, :] = TestImages[ImgNumber, :, :, :]
    
    # Fix for Shape Problem with Images depending on Channel
    NewImage = Image.fromarray(np.squeeze(np.uint8(ControlImage * 255),axis=2)) # Problem with Shape
    
    # Saving Image to SavePath
    NewImage.save('%s/generated_%s_%d.png' % (self.SavePath,self.Name,self.ImagesSavedCount))      
    
    # Update Counter for Image Naming
    self.ImagesSavedCount += 1  
