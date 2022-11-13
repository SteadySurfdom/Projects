import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class modVGG(tf.keras.Model):
    def __init__(self):
        super(modVGG,self).__init__()
        self.choose = "style"
        # name of the layers you want output of for style loss computation
        self.chosen_style = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']
        # name of the layer you want output of for content loss computation
        self.chosen_content = 'block5_conv2'
        self.model = tf.keras.applications.vgg19.VGG19(weights='imagenet',include_top=False)
        self.model.trainable = False
    def call(self,image):
        '''
        Modified `VGG19` model instance.
        
        Args:
            `image`: The input image.

        Returns:
            If `self.choose` attribute is set to `'style'`, the model returns a list of outputs of 5 Convolutional layers from each block.
            If `self.choose` attribute is set to `'content'`, the model returns the output of the Convolutional layer in the last block.
        '''
        features = []
        # Check if the input image is the combination image
        if isinstance(image,tf.Variable):
            img = image
            # Make sure the combination image has atleast 4 dimensions
            if len(img.get_shape()) == 3:
                img = tf.expand_dims(img,axis=0)
        # This else block means the input image is either content img or style img
        else:
            img = image
            # Make sure content/style image has atleast 4 dimensions
            if img.ndim == 3:
                img = tf.expand_dims(img,axis=0)
        
        # choose the type of output (Varies for style image or content image)
        if self.choose=='style':
            for l in self.model.layers:
                img = l(img)
                # if the name of the layer matches the chosen layers for saving their output, append their output to the features list
                if l.name in self.chosen_style:
                    features.append(img)
        elif self.choose=='content':
            for l in self.model.layers:
                img = l(img)
                # if the name of the layer matches the chosen layer for saving its output, re-assign the features variable as the output of this layer
                if l.name is self.chosen_content:
                    features = img
        return features

class NST_loss(tf.keras.losses.Loss):
    def __init__(self):
        super(NST_loss,self).__init__()
        self.model = modVGG()

    @staticmethod
    def content_fn(base_img,combination_img):
        '''
        Computes the Content Loss between the `base_img` and the `combination_img`.

        Args:
            `base_img`: The Content image.
            `combination_img`: The Combination image.
        '''
        return tf.reduce_sum(tf.square(base_img-combination_img))
    
    @staticmethod
    def gram_matrix(img):
        '''
        Computes the Gram Matrix for a given `img`.
        (Used for computing Style Loss)

        Args:
            `img`: The image you you want to compute the gram matrix of.
        '''
        # transposing so that (224,224,3) becomes (3,224,224)
        img = tf.transpose(img,(2,0,1))
        # opening the image along the height and width dimensions
        img = tf.reshape(img, shape=(tf.shape(img)[0],-1))
        matrix = tf.matmul(img,img,transpose_b=True)
        return matrix

    @staticmethod
    def style_fn(style_img,combination_img):
        '''
        Computes the Style Loss between the `style_img` and the `combination_img`.

        Args:
            `style_img`: The Style image.
            `combination_img`: The Combination image.
        '''
        style_gram = NST_loss.gram_matrix(style_img)
        combination_gram = NST_loss.gram_matrix(combination_img)
        return tf.reduce_sum(tf.square(style_gram-combination_gram)) / (4*9*(400**4)) # Dividing by a normalising constant, is a hyperparameter

    def __call__(self, content_img,style_img,combination_img,content_weight=2e-6,style_weight=2e-10):
        '''
        Computes the `Content Loss` and the `Style Loss` for the Combination image, given the Content and the Style image.

        Args:
            `content_img`: The image you want to retain the content from.
            `style_img`: The image you want to transfer the style from.
            `combination_img`: The final image, consisting of the content and style from the content and style images respectively.
            `content_weight`: The amount of content you want in your final image.
            `style_weight`: The amount of style you want to be transfered to your final image.

        Returns:
                Two numeric quantities which are:
                `Content_loss`: The loss between the content image and the combination image.
                `Style_loss`: The loss between the style image and the combination image.
        '''
        # set the 'choose' attribute to make the model ready to receive the appropriate kind of input
        self.model.choose = 'content'
        # send in the content image
        combination_features_wrt_content = self.model(combination_img)
        content_features = self.model(content_img)
        # set the 'choose' attribute to make the model ready to receive the appropriate kind of input
        self.model.choose = 'style'
        # send in the style image
        combination_features_wrt_style = self.model(combination_img)
        style_features = self.model(style_img)
        # Compute the Content Loss...
        content_loss = content_weight*self.content_fn(content_features,combination_features_wrt_content)
        # Compute the Style Loss...
        style_loss = 0
        for style_feat,combination_feat in zip(style_features,combination_features_wrt_style):
            style_feat = tf.squeeze(style_feat)
            combination_feat = tf.squeeze(combination_feat)
            style_loss += style_weight*self.style_fn(style_feat,combination_feat)
        return content_loss,style_loss

class NST(tf.keras.Model):
    def __init__(self,training_img):
        super(NST,self).__init__()
        self.epoch = 1
        self.training_image = training_img
    def compile(self,optimizer=tf.keras.optimizers.Adam(learning_rate=10),loss=NST_loss()):
        '''
        Custom compile for NST model.

        Args (Modified):
            `optimizer`: The optimizer for the backpropogation algorithm. Defaults to Adam.
            `loss`: The loss function for the NST model. Defaults to NST_loss.
        '''
        super(NST,self).compile(run_eagerly=True)
        self.optimizer = optimizer
        self.initial_lr_of_optim = self.optimizer.learning_rate
        self.loss = loss
    def train_step(self,data):
        # unpack the data as tuple
        x,y = tuple(data)
        self.epoch += 1
        # Compute the gradients on loss
        with tf.GradientTape() as tape:
            tape.watch(self.training_image)
            content_loss, style_loss = self.loss(x,y,self.training_image)
            total_loss = content_loss + style_loss
        grads = tape.gradient(total_loss,[self.training_image])
        # apply the gradients to the training image
        self.optimizer.apply_gradients(zip(grads,[self.training_image]))
        # update the learning rate of the optimizer
        self.optimizer.learning_rate = int(self.initial_lr_of_optim)*(tf.pow(0.96,self.epoch/100))
        return {'content loss' : content_loss, 'style loss': style_loss, 'total loss': total_loss}
    def result(self):
        return self.training_image