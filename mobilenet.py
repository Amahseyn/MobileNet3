
import os
import cv2
import collections
import random
import time
import numpy as np
from os.path import join
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.losses import BinaryCrossentropy,\
    SparseCategoricalCrossentropy
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras.layers  import Layer, InputSpec
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Dense,\
    GlobalAveragePooling2D, Layer
from tensorflow.keras.layers import Activation, BatchNormalization, Add,\
    Multiply, Reshape, AveragePooling2D
from tensorflow.image import ResizeMethod
from collections.abc import Iterable

cv2.setNumThreads(0)


class ImageTargetDataset(Sequence):
    def __init__(self,images, image_dir, labels, label_dir,
                 batch_size,
                 shuffle=True,
                 device='GPU:0',
                 transform=None,
                 image_transform=None,
                 target_transform=None):
        super().__init__()

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.transform = transform
        self.target_transform = target_transform
        self.labels = labels
        self.label_dir = label_dir
        self.image_transform = image_transform
        self.images = images
        self.image_dir = image_dir
        self.on_epoch_end()
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [k for k in indexes]

        # Generate augmented data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.images))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_imgs_temp):
        """Generates data containing batch_size samples."""
        # X : (n_samples, *dim, n_channels)
        # Initialization
        input_list = []
        target_list = []
        # Generate data

        for i in list_imgs_temp:
            image = cv2.imread(self.image_dir + self.images[i])
            image = image[:, :, ::-1]
            # Replace 'images' with 'masks' in the path to get the target path
            target = cv2.imread(self.label_dir + self.labels[i], cv2.IMREAD_GRAYSCALE)
            target = (target >= 10).astype(np.uint8)

            if self.transform is not None:
                image, target = self.transform(image, target)
            if self.target_transform is not None:
                target = self.target_transform(target)
            if self.image_transform is not None:
                image = self.image_transform(image)
            input_list.append(image)
            target_list.append(target)

        return tf.stack(input_list), tf.stack(target_list)



def img_size(image: np.ndarray):
    """
    Return images width and height.
    :param image: nd.array with image
    :return: width, height
    """
    return (image.shape[1], image.shape[0])


def img_crop(img, box):
    img_new = img[box[1]:box[3], box[0]:box[2]]
    return img_new


def img_saturate(img):
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    return img


class MaskToTensor:
    def __init__(self, add_background=False):
        self.add_background = add_background

    def __call__(self, mask):
        mask[mask > 0] = 1
        if self.add_background:
            background_mask = np.ones_like(mask) - mask
            mask = np.stack([mask, background_mask], axis=2)
        mask = tf.convert_to_tensor(mask)
        mask = tf.dtypes.cast(mask, 'float32')
        if len(mask.shape) < 3:
            mask = tf.expand_dims(mask, 2)
        return mask


class UseWithProb:
    """Apply a given transform with probability or return input unchanged."""
    def __init__(self, transform, prob=.5):
        self.transform = transform
        self.prob = prob

    def __call__(self, image, mask=None):
        if self.prob > 0 and random.uniform(0, 1) < self.prob:
            image, mask = self.transform(image, mask)
        return image, mask


class OutputTransform:
    def __init__(self, segm_thresh=0.5):
        self.segm_thresh = segm_thresh

    def __call__(self, mask):
        mask = mask > self.segm_thresh
        return mask


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask=None):
        if mask is None:
            for trns in self.transforms:
                image = trns(image)
            return image
        else:
            for trns in self.transforms:
                image, mask = trns(image, mask)
            return image, mask


class Scale(object):
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        assert isinstance(size, Iterable) and len(size) == 2
        self.size = tuple(size)
        self.interpolation = interpolation

    def __call__(self, img, mask=None):
        img = cv2.resize(img, self.size, interpolation=self.interpolation)
        if mask is not None:
            mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        return img, mask


class RandomCrop(object):
    def __init__(self, scale=0.1):
        self.scale = scale

    def __call__(self, img, mask=None):
        w, h = img_size(img)
        tw, th = int(self.scale*w), int(self.scale*h)

        # Top-left corner
        x1 = random.randint(0, tw)
        y1 = random.randint(0, th)

        # Bottom-right corner
        x2 = random.randint(w-tw, w)
        y2 = random.randint(h-th, h)
        img = img_crop(img, (x1, y1, x2, y2))
        if mask is not None:
            mask = img_crop(mask, (x1, y1, x2, y2))
        return img, mask


class SquareCrop(object):
    def __call__(self, img, mask=None):
        w, h = img_size(img)
        if w > h:
            shift = int((w-h)/2)
            box = (shift, 0, shift+h, h)
        else:
            shift = int((h-w)/2)
            box = (0, shift, w, shift+w)

        img = img_crop(img, box)
        if mask is not None:
            mask = img_crop(mask, box)
        return img, mask


def generate_new_crop(x, y, w, h, image_height, image_width,
                      width_limit=250, height_limit=125):

    start_horizontal = max(0, x - width_limit)
    new_x = random.randint(start_horizontal, x)
    start_vertical = max(0, y - height_limit)
    new_y = random.randint(start_vertical, y)
    finish_horizontal = min(image_width, x + w + width_limit)
    new_w_x = random.randint(x + w, finish_horizontal)
    finish_vertical = min(image_height, y + h + height_limit)
    new_h_y = random.randint(y + h, finish_vertical)
    if new_h_y - new_y > new_w_x - new_x\
            and new_x + new_h_y - new_y < image_width:
        new_w_x = new_x + new_h_y - new_y
    return new_x, new_y, new_w_x, new_h_y


class RandomMaskCrop(object):
    def __init__(self, width_limit=250, height_limit=125):
        self.width_limit = width_limit
        self.height_limit = height_limit

    def __call__(self, img, mask):
        height, width, channels = img.shape
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            random_contour_id = random.randint(0, len(contours)-1)
            x, y, w, h = cv2.boundingRect(contours[random_contour_id])
            new_x, new_y, new_w_x, new_h_y = generate_new_crop(
                x, y, w, h, height, width, self.width_limit, self.height_limit)
            img = img[new_y:new_h_y, new_x:new_w_x, :]
            mask = mask[new_y:new_h_y, new_x:new_w_x]

        return img, mask


def central_crop(img, mask, part=0.1):
    h, w, c = img.shape
    img = img[int(part * h):h - int(part * h),
              int(part * w):w - int(part * w), :]
    mask = mask[int(part * h):h - int(part * h),
                int(part * w):w - int(part * w)]
    return img, mask


class RandomRotation(object):
    def __init__(self, ang_range=15, crop_part=0.1, probability=0.1):
        self.ang_range = ang_range
        self.crop_part = crop_part

    def __call__(self, img, mask):
        ang_rot = random.uniform(-self.ang_range, self.ang_range)
        rows, cols, ch = img.shape
        Rot_M = cv2.getRotationMatrix2D((cols/2, rows/2), ang_rot, 1)
        img = cv2.warpAffine(img, Rot_M, (cols, rows))
        mask = cv2.warpAffine(mask, Rot_M, (cols, rows))
        img, mask = central_crop(img, mask, self.crop_part)
        return img, mask


class Flip(object):
    def __init__(self, flip_code):
        self.flip_code = flip_code

    def __call__(self, imgs, trgs_mask):
        flip_imgs = cv2.flip(imgs, self.flip_code)
        trgs_mask = cv2.flip(trgs_mask, self.flip_code)
        return flip_imgs, trgs_mask


class HorizontalFlip(Flip):
    def __init__(self):
        super().__init__(1)


class ToTensorColor(object):
    def __call__(self, img):
        assert isinstance(img, np.ndarray)
        tensor = tf.convert_to_tensor(img)
        tensor = tf.dtypes.cast(tensor, 'float32')
        return tf.divide(tensor, 255.0)


class AugmentImage(object):
    def __init__(self, augment_parameters):
        self.gamma_low = augment_parameters[0]  # 0.8
        self.gamma_high = augment_parameters[1]  # 1.2
        self.brightness_low = augment_parameters[2]  # 0.5
        self.brightness_high = augment_parameters[3]  # 2.0
        self.color_low = augment_parameters[4]  # 0.8
        self.color_high = augment_parameters[5]  # 1.2

    def __call__(self, img, mask=None):
        random_gamma = random.uniform(self.gamma_low, self.gamma_high)
        random_brightness = random.uniform(
            self.brightness_low, self.brightness_high)
        random_colors = np.array(
            [random.uniform(self.color_low, self.color_high)
             for _ in range(3)]) * random_brightness

        img = img.astype(float)
        # randomly shift gamma
        img = img ** random_gamma
        # randomly shift brightness and color
        for i in range(3):
            img[:, :, i] = img[:, :, i] * random_colors[i]
        # saturate
        img = img_saturate(img)
        return img, mask


class RandomGaussianBlur:
    """Apply Gaussian blur with random kernel size
    Args:
        max_ksize (int): maximal size of a kernel to apply, should be odd
        sigma_x (int): Standard deviation
    """
    def __init__(self, max_ksize=5, sigma_x=35):
        assert max_ksize % 2 == 1, "max_ksize should be odd"
        self.max_ksize = max_ksize // 2 + 1
        self.sigma_x = sigma_x

    def __call__(self, img, mask=None):
        kernal_size = (2 * random.randint(0, self.max_ksize) + 1,
                       2 * random.randint(0, self.max_ksize) + 1)
        img = cv2.GaussianBlur(img, kernal_size, self.sigma_x)
        return img, mask


class BasicNoise:
    """Apply Gauss or speckle noise to an image.

    Args:
        sigma_sq (float): Sigma squared to generate a noise matrix
        speckle (bool): False - Gauss noise, True - speckle
    """
    def __init__(self, sigma_sq, speckle=False):
        self.sigma_sq = sigma_sq
        self.speckle = speckle

    def __call__(self, img, mask=None):
        if self.sigma_sq > 0.0:
            w, h, c = img.shape
            sigma_to_use = random.uniform(0, self.sigma_sq)
            gauss = np.random.normal(0, sigma_to_use, (w, h, c))
            img = img.astype(np.int32)
            if self.speckle:
                img = img * gauss
            else:
                img = img + gauss
            img = img_saturate(img)
        return img, mask


class ComposeSegDet(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, trgs_mask):

        if trgs_mask is None:
            for t in self.transforms:
                img = t(img)
            return img
        else:
            for t in self.transforms:
                img, trgs_mask = t(img, trgs_mask)
            return img, trgs_mask


def train_transforms(dataset='coco', scale_size=(512, 512), sigma_g=25,
                     ang_range=15, width_limit=250, height_limit=150,
                     augment_params=[0.8, 1.2, 0.8, 1.2, 0.8, 1.2],
                     crop_scale=0.2, add_background=False):
    transforms_dict = dict()

    transforms_dict['transform'] = ComposeSegDet([
        UseWithProb(RandomRotation(ang_range), 0.5),
        RandomMaskCrop(width_limit, height_limit),
        SquareCrop(),
        Scale(scale_size),
        UseWithProb(HorizontalFlip(), 0.5),
        UseWithProb(AugmentImage(augment_params), 0.5),
        UseWithProb(RandomGaussianBlur(), 0.2),
        UseWithProb(BasicNoise(sigma_g), 0.3)])

    transforms_dict['image_transform'] = ToTensorColor()
    transforms_dict['target_transform'] = MaskToTensor(
        add_background=add_background)
    return transforms_dict


def test_transforms(dataset='coco', scale_size=(512, 512),
                    add_background=False):
    transforms_dict = dict()
    transforms_dict['transform'] = ComposeSegDet([
        RandomMaskCrop(0, 0),
        SquareCrop(),
        Scale(scale_size)
    ])

    transforms_dict['image_transform'] = ToTensorColor()
    transforms_dict['target_transform'] = MaskToTensor(
        add_background=add_background)
    return transforms_dict


def convert_transforms(scale_size=(512, 512)):
    transforms_dict = dict()
    transforms_dict['transform'] = ComposeSegDet([
        SquareCrop(),
        Scale(scale_size)
    ])

    transforms_dict['image_transform'] = ToTensorColor()
    transforms_dict['target_transform'] = None
    return transforms_dict

"""Loss"""

def fb_loss(trues, preds, beta, channel_axis):
    smooth = 1e-4
    beta2 = beta*beta
    batch = preds.shape[0]
    classes = preds.shape[channel_axis]
    preds = tf.reshape(preds, [batch, classes, -1])
    trues = tf.reshape(trues, [batch, classes, -1])
    trues_raw = tf.reduce_sum(trues, axis=-1)
    weights = tf.clip_by_value(trues_raw, 0., 1.)
    TP_raw = preds * trues
    TP = tf.reduce_sum(TP_raw, axis=2)
    FP_raw = preds * (1-trues)
    FP = tf.reduce_sum(FP_raw, axis=2)
    FN_raw = (1-preds) * trues
    FN = tf.reduce_sum(FN_raw, axis=2)
    Fb = ((1+beta2) * TP + smooth)/((1+beta2) * TP + beta2 * FN + FP + smooth)
    Fb = Fb * weights
    score = tf.reduce_sum(Fb) / (tf.reduce_sum(weights) + smooth)
    return tf.clip_by_value(score, 0., 1.)


def make_cross_entropy_target(target):
    # target = target.byte()
    b, c, w, h = target.shape
    ce_target = tf.zeros((b, w, h))
    for channel in range(c):
        ce_target = tf.where(target[:, channel, :, :], channel, ce_target)
    return ce_target


class FBLoss:
    def __init__(self, beta=1, channel_axis=-1):
        self.beta = beta
        self.channel_axis = channel_axis

    def __call__(self, target, output):
        return 1 - fb_loss(target, output, self.beta, self.channel_axis)


class FbCombinedLoss:
    def __init__(self, channel_axis=-1, fb_weight=0.5, fb_beta=1,
                 entropy_weight=0.5, use_bce=True, normalize=False):
        self.fb_weight = fb_weight
        self.entropy_weight = entropy_weight
        self.fb_loss = FBLoss(beta=fb_beta, channel_axis=channel_axis)
        self.use_bce = use_bce
        self.normalize = normalize
        if use_bce:
            self.entropy_loss = BinaryCrossentropy()
        else:
            self.entropy_loss = SparseCategoricalCrossentropy()

    def __call__(self, target, output):
        if self.normalize:
            output = F.softmax(output, dim=1)
        if self.fb_weight > 0:
            fb = self.fb_loss(target, output) * self.fb_weight
        else:
            fb = 0
        if self.entropy_weight > 0:
            if self.use_bce is False:
                target = make_cross_entropy_target(target)
            ce = self.entropy_loss(target, output) * self.entropy_weight
        else:
            ce = 0
        return fb + ce

"""Metrics"""

class FbSegm():
    """Compute F_beta of segmentation."""

    def __init__(self, beta=1, channel=None, channel_axis=-1):
        self.channel = channel
        self.beta = beta
        self.channel_axis = channel_axis
        self.reset()

    def reset(self):
        self.se = 0
        self.count = 0

    def update(self, target_mask, pred_mask):
        if self.channel is not None:
            target_mask = target_mask[:, self.channel:(self.channel+1), :, :]
            pred_mask = pred_mask[:, self.channel:(self.channel+1), :, :]
        segm_fb = fb_loss(target_mask, pred_mask, self.beta, self.channel_axis)
        self.se += segm_fb * target_mask.shape[0]
        self.count += target_mask.shape[0]

    def compute(self):
        if not self.count:
            raise Exception('Must be at least one example for computation')
        else:
            return (self.se/self.count).numpy()

"""train"""

class CustomModel(Model):
    def __init__(self, model_name, n_class, input_shape, old_model_path=None, **kwargs):
        super(CustomModel, self).__init__()  # Call the __init__ method of the base class
        # Set up model
        self.model = get_model(model_name, **kwargs)
        init_input = tf.ones(input_shape)
        _ = self.model.predict(init_input)
        if old_model_path is not None:
            self.load(old_model_path)

    def prepare_train(self, train_loader, n_train, val_loader, n_val, loss_name,
                      optimizer, lr, batch_size, max_epoches, save_directory,
                      reduce_factor=None, epoches_limit=0, metrics=None, early_stoping=None,
                      **kwargs):
        self.loss_function = get_loss(loss_name, **kwargs)
        self.lr = lr
        self.optimizer = get_optimizer(optimizer)(learning_rate=lr)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_examples = n_train
        self.val_examples = n_val
        self.max_epoches = max_epoches
        self.batch_size = batch_size
        self.save_directory = save_directory
        self.reduce_factor = reduce_factor
        self.epoches_limit = epoches_limit
        self.early_stoping = early_stoping
        self.metric_list = None
        if metrics is not None:
            self.metric_list=[]
            for metric in metrics:
                self.metric_list.append(metric)
        make_dir(save_directory)

    def prepare_val(self, val_loader, n_val, loss_name, batch_size,
                      metrics=None, **kwargs):
        self.loss_function = get_loss(loss_name, **kwargs)
        self.val_loader = val_loader
        self.val_examples = n_val
        self.batch_size = batch_size
        self.lr = 0
        self.metric_list = None
        if metrics is not None:
            self.metric_list=[]
            for metric in metrics:
                self.metric_list.append(metric)
    def train_step(self, batch):
        with tf.GradientTape() as tape:
            input, target = batch[0], batch[1]
            prediction = self.model(input, training=True)
            # print(target.shape)
            # print(prediction.shape)
            # print(tf.math.reduce_max(prediction))
            # print(tf.math.reduce_min(prediction))
            loss = self.loss_function(target, prediction)
        # print(self.model.trainable_weights)
        grads = tape.gradient(loss, self.model.trainable_weights)
        # print(grads)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss.numpy(), prediction, target
    def val_step(self, batch):
        input, target = batch
        prediction = self.model.predict(input)
        loss = self.loss_function(target, prediction)
        if self.metric_list is not None:
            for metric in self.metric_list:
                metric.update(target, prediction)
        return loss.numpy(), prediction, target

    def loader_loop(self, loader, number_of_examples,
                    mode='train'):
        running_loss = 0.0
        if self.metric_list is not None:
            for metric in self.metric_list:
                metric.reset()
        for data in loader:
            if mode == 'train':
                loss_value, pred, target = self.train_step(data)
            elif mode == 'val':
                loss_value, pred, target = self.val_step(data)
            running_loss += loss_value
        running_loss /= number_of_examples# / self.batch_size
        return running_loss

    def fit(self, best_on_val=True):
        # with tf.device(self.device):
        best_loss = float('Inf')
        best_val_loss = float('Inf')
        count_without_improvement = 0
        file_path = ''
        # First validation loop
        c_time = time.time()
        running_val_loss = self.loader_loop(self.val_loader, self.val_examples, mode='val')
        self.print_results(0, 0, running_val_loss, c_time)

        # Train and validation loops
        for epoch in range(self.max_epoches):
            c_time = time.time()
            running_loss = self.loader_loop(self.train_loader, self.train_examples, mode='train')
            running_val_loss = self.loader_loop(self.val_loader, self.val_examples, mode='val')
            self.print_results(epoch, running_loss, running_val_loss, c_time)
            # Saving of the best model
            if best_on_val:
                best_val_loss, count_without_improvement, file_path = self.save_best_model(running_val_loss,
                                                                                           best_val_loss,
                                                                                           count_without_improvement,
                                                                                           file_path)
            else:
                best_loss, count_without_improvement, file_path = self.save_best_model(running_loss,
                                                                                       best_loss,
                                                                                       count_without_improvement,
                                                                                       file_path)
            if self.reduce_factor is not None:
                count_without_improvement = self.reduce_on_plateau(self.reduce_factor,
                                                                   self.epoches_limit,
                                                                   count_without_improvement)
            if self.early_stoping is not None:
                if count_without_improvement > self.early_stoping:
                    print("Stopped. Didn't improve for {} epochs.".format(count_without_improvement))
                    break


        print('Training is finished. Train loss: {}. Best val loss: {}', running_loss, best_val_loss)

    def save_best_model(self, save_loss, save_best_loss, count_without_improvement, old_file_path):
        self.save(self.save_directory + '/model_last.h5')

        if save_loss < save_best_loss:
            count_without_improvement = 0
            print('Model_removed:', old_file_path)
            if os.path.exists(old_file_path):
                os.remove(old_file_path)
            model_file_path = self.save_directory + '/model_best_{}.h5'.format(np.round(
                                                                              np.array(save_loss).astype(np.float32),
                                                                              5))

            self.save(model_file_path)
            print('Model_saved:', model_file_path)

            save_best_loss = save_loss
        else:
            count_without_improvement += 1
            model_file_path = old_file_path
        return save_best_loss, count_without_improvement, model_file_path

    def validate(self, val_loader, val_examples):
        c_time = time.time()
        # self.model.eval()
        running_val_loss = self.loader_loop(val_loader, val_examples, mode='val')
        self.print_results(0, 0, running_val_loss, c_time)

    def predict(self, input):
        with tf.device("GPU:0"):
            prediction = self.model.predict(input)
            return prediction

        # pass
    def save(self, path):
        self.model.save_weights(path)
        # pass

    def load(self, path):
        self.model.load_weights(path)

    def print_results(self, epoch, running_loss, running_val_loss, c_time):
        print()
        print(
            'Epoch:', epoch + 1,
            'train_loss:', running_loss,
            'val_loss:', running_val_loss,
            'time:', round(time.time() - c_time, 3),
            's',
            'lr:', self.lr,
        )
        if self.metric_list is not None:
            for metric in self.metric_list:
                value = np.round(float(metric.compute()), 4)
                name = metric.__class__.__name__
                print(name + ': ', value)

    def reduce_on_plateau(self, reduce_factor, epoches_limit, count_without_improvement):
        if count_without_improvement >= epoches_limit:
            count_without_improvement = 0
            self.lr = self.lr*reduce_factor
            self.optimizer.learning_rate = self.lr
        return count_without_improvement

"""models"""

def relu6(x):
    """Relu 6."""
    return tf.nn.relu(x)


def hard_swish(x):
    """Hard swish."""
    return x * tf.nn.relu(x + 3.0) / 6.0


def return_activation(x, nl):
    """Convolution Block
    This function defines a activation choice.
    # Arguments
        x: Tensor, input tensor of conv layer.
        nl: String, nonlinearity activation type.
    # Returns
        Output tensor.
    """
    if nl == 'HS':
        x = Activation(hard_swish)(x)
    if nl == 'RE':
        x = Activation(relu6)(x)
    return x


class ConvBlock(Layer):
    """Convolution Block
    This class defines a 2D convolution operation with BN and activation.
    # Arguments
        # init
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window. Default=(3,3)
            strides: An integer or tuple/list of 2 integers,
                specifying the strides of the convolution along the width and
                height. Can be a single integer to specify the same value for
                all spatial dimensions. Default=1
            nl: String, nonlinearity activation type. Default='RE'
            channel_axis: 1 if channels are first in the image and -1 if the
                last. Default=-1
            padding_scheme: Padding scheme to apply for convolution.
                Default='same'
        # call
            x: Tensor, input tensor of conv layer.
            training: Mode for training-aware layers
    # Returns
        Output tensor.
    """

    def __init__(self, filters, kernel=(3, 3), strides=1, nl='RE',
                 padding='same', channel_axis=-1):
        super(ConvBlock, self).__init__()
        self.channel_axis = channel_axis
        self.nl = nl
        self.conv = Conv2D(filters, kernel, padding=padding, strides=strides)
        self.bn = BatchNormalization(axis=channel_axis)

    def call(self, x, training=True):
        x = self.conv(x)
        # Remove the 'training' argument to convert to TFLite
        x = self.bn(x, training=training)
        return return_activation(x, self.nl)


class Squeeze(Layer):
    """Squeeze and Excitation.
    This function defines a squeeze structure.
    # Arguments
        #call
            inputs: Tensor, input tensor of conv layer
    # Returns
        Output tensor.
    """

    def __init__(self):
        super(Squeeze, self).__init__()

    def build(self, input_shape):
        # print(input_shape)
        self.input_channels = input_shape[-1]
        self.fc1 = Dense(self.input_channels, activation='relu')
        self.fc2 = Dense(self.input_channels, activation='hard_sigmoid')

    def call(self, inputs):
        x = GlobalAveragePooling2D()(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        x = Reshape((1, 1, self.input_channels))(x)
        x = Multiply()([inputs, x])
        return x


class Bottleneck(Layer):
    """Bottleneck
    This class defines a basic bottleneck structure.
    # Arguments
        # init
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            expansion: Integer, expansion factor.
                t is always applied to the input size.
            strides: An integer or tuple/list of 2 integers,specifying the
                strides of the convolution along the width and height.
                Can be a single integer to specify the same value for all
                spatial dimensions.
            squeeze: Boolean, Whether to use the squeeze.
            nl: String, nonlinearity activation type.
            alpha: Multiplier of number of intermediate channels
            channel_axis: 1 if channels are first in the image and -1 if
                the last. Default=-1
        #call
            inputs: Tensor, input tensor of conv layer
            training: Mode for training-aware layers
    # Returns
        Output tensor.
    """

    def __init__(self, filters, kernel, expansion, strides, squeeze, nl,
                 alpha=1.0, channel_axis=-1):
        super(Bottleneck, self).__init__()
        self.strides = strides
        self.filters = filters
        self.nl = nl
        self.squeeze = squeeze
        if self.squeeze:
            self.squeeze_layer = Squeeze()
        tchannel = int(expansion)
        cchannel = int(alpha * filters)

        self.conv_block = ConvBlock(tchannel, kernel=(1, 1),
                                    strides=(1, 1), nl=nl)
        self.dw_conv = DepthwiseConv2D(kernel, strides=(strides, strides),
                                       depth_multiplier=1, padding='same')
        self.bn1 = BatchNormalization(axis=channel_axis)
        self.conv2d = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')
        self.bn2 = BatchNormalization(axis=channel_axis)

    def build(self, input_shape):
        self.r = self.strides == 1 and input_shape[3] == self.filters

    def call(self, inputs, training=True):
        x = self.conv_block(inputs)
        x = self.dw_conv(x)
        # Remove the 'training' argument to convert to TFLite
        x = self.bn1(x, training=training)
        x = return_activation(x, self.nl)
        if self.squeeze:
            x = self.squeeze_layer(x)
        x = self.conv2d(x)
        # Remove the 'training' argument to convert to TFLite
        x = self.bn2(x, training=training)
        if self.r:
            x = Add()([x, inputs])
        return x


class MobileNetV3SmallBackbone(Layer):
    def __init__(self, alpha=1.0, mode='segmentation'):
        """MobileNetV3SmallBackbone.
        # Arguments
            # init
                alpha: Integer, width multiplier.
                mode: String, either "classification" or "segmentation"
            #call
                inputs: Tensor, input tensor of the model
                training: Mode for training-aware layers
        # Returns
            segm_features1: Feature map with h/8 resolution
            segm_features2: Feature map with h/16 resolution
        """
        super(MobileNetV3SmallBackbone, self).__init__()
        self.alpha = alpha
        self.mode = mode
        self.first_conv = ConvBlock(16, (3, 3), strides=2, nl='HS')  # h/2
        self.bottleneck1 = Bottleneck(
            16, (3, 3), expansion=16, strides=2, squeeze=True,
            nl='RE', alpha=alpha)  # h/4
        self.bottleneck2 = Bottleneck(
            24, (3, 3), expansion=72, strides=2, squeeze=False,
            nl='RE', alpha=alpha)  # h/8
        self.bottleneck3 = Bottleneck(
            24, (3, 3), expansion=88, strides=1, squeeze=False,
            nl='RE', alpha=alpha)  # h/8
        self.bottleneck4 = Bottleneck(
            40, (5, 5), expansion=96, strides=2, squeeze=True,
            nl='HS', alpha=alpha)  # h/16
        self.bottleneck5 = Bottleneck(
            40, (5, 5), expansion=240, strides=1, squeeze=True,
            nl='HS', alpha=alpha)  # h/16
        self.bottleneck6 = Bottleneck(
            40, (5, 5), expansion=240, strides=1, squeeze=True,
            nl='HS', alpha=alpha)  # h/16
        self.bottleneck7 = Bottleneck(
            48, (5, 5), expansion=120, strides=1, squeeze=True,
            nl='HS', alpha=alpha)  # h/16
        self.bottleneck8 = Bottleneck(
            48, (5, 5), expansion=144, strides=1, squeeze=True,
            nl='HS', alpha=alpha)  # h/16
        if self.mode == 'classification':
            self.bottleneck9 = Bottleneck(
                96, (5, 5), expansion=288, strides=2, squeeze=True,
                nl='HS', alpha=alpha)  # h/32
            self.bottleneck10 = Bottleneck(
                96, (5, 5), expansion=576, strides=1, squeeze=True,
                nl='HS', alpha=alpha)  # h/32
            self.bottleneck11 = Bottleneck(
                96, (5, 5), expansion=576, strides=1, squeeze=True,
                nl='HS', alpha=alpha)  # h/32
            # Last stage
            self.last_stage_conv1 = ConvBlock(
                576, (1, 1), strides=1, nl='HS')  # h/32
            self.last_stage_conv2 = Conv2D(1280, (1, 1), padding='same')  # h/h

    def call(self, inputs, training=True):
        # print(inputs.shape)
        x = self.first_conv(inputs, training=training)
        x = self.bottleneck1(x, training=training)
        x = self.bottleneck2(x, training=training)
        segm_features1 = self.bottleneck3(x, training=training)
        x = self.bottleneck4(segm_features1, training=training)
        x = self.bottleneck5(x, training=training)
        x = self.bottleneck6(x, training=training)
        x = self.bottleneck7(x, training=training)
        segm_features2 = self.bottleneck8(x, training=training)
        if self.mode == 'segmentation':
            return segm_features1, segm_features2

        elif self.mode == 'classification':
            x = self.bottleneck9(segm_features2, training=training)
            x = self.bottleneck10(x, training=training)
            x = self.bottleneck11(x, training=training)
            # Last stage
            x = self.last_stage_conv1(x, training=training)
            x = GlobalAveragePooling2D()(x)
            x = Reshape((1, 1, 576))(x)
            x = self.last_stage_conv2(x)
            x = return_activation(x, 'HS')
            return x


class LiteRASSP(Layer):
    def __init__(self, shape=(224, 224), n_class=1, avg_pool_kernel=(49, 49),
                 avg_pool_strides=(16, 20),
                 resize_method=ResizeMethod.BILINEAR):
        """LiteRASSP.
        # Arguments
            # init
                input_shape: Tuple/list of 2 integers, spatial shape of input
                    tensor
                n_class: Integer, number of classes.
                avg_pool_kernel: Tuple/integer, size of the kernel for
                    AveragePooling
                avg_pool_strides: Tuple/integer, stride for applying the of
                    AveragePooling operation
            # Call
                inputs: Tensor, input tensor of the model
                training: Mode for training-aware layers
        # Returns
            Output tensor of the original shape
            """
        super(LiteRASSP, self).__init__()
        self.shape = shape
        self.n_class = n_class
        self.avg_pool_kernel = avg_pool_kernel  # 11
        self.avg_pool_strides = avg_pool_strides  # 4
        self.resize_method = resize_method
        # branch1
        self.branch1_convblock = ConvBlock(128, 1, strides=1, nl='RE')
        # branch2
        self.branch2_avgpool = AveragePooling2D(pool_size=self.avg_pool_kernel,
                                                strides=self.avg_pool_strides)
        self.branch2_conv = Conv2D(128, 1, strides=1)
        # bracnh3
        self.branch3_conv = Conv2D(self.n_class, 1, strides=1)
        # merge1_2
        self.merge1_2_conv = Conv2D(self.n_class, 1, strides=1)

    def call(self, inputs, training=True):
        out_feature8, out_feature16 = inputs
        # branch1
        x1 = self.branch1_convblock(out_feature16, training=training)
        # branch2
        s = x1.shape
        x2 = self.branch2_avgpool(out_feature16)
        x2 = self.branch2_conv(x2)
        x2 = Activation('sigmoid')(x2)
        x2 = tf.image.resize(x2,
                             size=(int(s[1]), int(s[2])),
                             method=self.resize_method,
                             preserve_aspect_ratio=False,
                             antialias=False,
                             name=None)
        # branch3
        x3 = self.branch3_conv(out_feature8)
        # merge1_2
        x = Multiply()([x1, x2])
        x = tf.image.resize(x,
                            size=(int(2*s[1]), int(2*s[2])),
                            method=self.resize_method,
                            preserve_aspect_ratio=False,
                            antialias=False,
                            name=None)
        x = self.merge1_2_conv(x)
        # merge3
        x = Add()([x, x3])
        # # out
        x = tf.image.resize(x,
                            size=self.shape,
                            method=self.resize_method,
                            preserve_aspect_ratio=False,
                            antialias=False,
                            name=None)
        x = Activation('sigmoid')(x)
        # x = tf.nn.softmax(x, axis=-1)
        return x


class MobileNetV3SmallSegmentation(Model):
    def __init__(self, alpha=1.0, shape=(224, 224), n_class=1,
                 avg_pool_kernel=(11, 11), avg_pool_strides=(4, 4),
                 resize_method=ResizeMethod.BILINEAR, backbone='small'):
        """MobileNetV3SmallSegmentation.
        # Arguments
            # init
                alpha: Integer, width multiplier.
                input_shape: Tuple/list of 2 integers, spatial shape of input
                    tensor
                n_class: Integer, number of classes.
                avg_pool_kernel: Tuple/integer, size of the kernel for
                    AveragePooling
                avg_pool_strides: Tuple/integer, stride for applying the of
                    AveragePooling operation
                resize_method: Object, One from tensorflow.image.ResizeMethod
                backbone: String, name of backbone to use
            # Call
                inputs: Tensor, input tensor of the model
                training: Mode for training-aware layers
        # Returns
            Result of segmentation
            """
        super(MobileNetV3SmallSegmentation, self).__init__()
        if backbone == 'small':
            self.backbone = MobileNetV3SmallBackbone(
                alpha=alpha, mode='segmentation')
        self.segmentation_head = LiteRASSP(shape=shape,
                                           n_class=n_class,
                                           avg_pool_kernel=avg_pool_kernel,
                                           avg_pool_strides=avg_pool_strides,
                                           resize_method=resize_method)

    def call(self, inputs, training=True):
        segm_inputs = self.backbone(inputs, training)
        output = self.segmentation_head(segm_inputs, training)
        return output

"""utils"""

# Define dictionaries with modules names
loss_dict = {'fb_combined': FbCombinedLoss,
             'bce': BinaryCrossentropy}

model_dict = {'mobilenet_small': MobileNetV3SmallSegmentation}


def get_optimizer(optimizer_name):
    optimizers_dict = {subcl.__name__: subcl
                       for subcl in tf.keras.optimizers.Optimizer.__subclasses__()}
    assert optimizer_name in optimizers_dict.keys(
    ), "Optimizer name is not in TensorFlow optimizers"
    return optimizers_dict[optimizer_name]

def get_model(model_name, **kwargs):
    assert model_name in model_dict.keys(), "Unknown model name"
    model = model_dict[model_name](**kwargs)
    return model


def get_loss(loss_name, **kwargs):
    assert loss_name in loss_dict.keys(), "Unknown loss name"
    loss_function = loss_dict[loss_name](**kwargs)
    return loss_function


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

"""layes"""

def resize_images_bilinear(X, height_factor=1, width_factor=1, target_height=None,
                           target_width=None, data_format='default'):
    if data_format == 'default':
        data_format = K.image_data_format()

    if data_format == 'channels_first':
        original_shape = K.int_shape(X)

        if target_height and target_width:
            new_shape = tf.constant(np.array((target_height, target_width)).astype('int32'))
        else:
            new_shape = tf.shape(X)[2:]
            new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))

        X = K.permute_dimensions(X, [0, 2, 3, 1])
        X = tf.image.resize_bilinear(X, new_shape)
        X = K.permute_dimensions(X, [0, 3, 1, 2])

        if target_height and target_width:
            X.set_shape((None, None, target_height, target_width))
        else:
            X.set_shape((None, None, original_shape[2] * height_factor, original_shape[3] * width_factor))

        return X
    elif data_format == 'channels_last':
        original_shape = K.int_shape(X)

        if target_height and target_width:
            new_shape = tf.constant(np.array((target_height, target_width)).astype('int32'))
        else:
            new_shape = tf.shape(X)[1:3]
            new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))

        X = tf.image.resize_bilinear(X, new_shape)

        if target_height and target_width:
            X.set_shape((None, target_height, target_width, None))
        else:
            X.set_shape((None, original_shape[1] * height_factor, original_shape[2] * width_factor, None))

        return X
    else:
        raise Exception('Invalid data_format: ' + data_format)


class BilinearUpSampling2D(Layer):
    def __init__(self, size=(1, 1), target_size=None, data_format='default', **kwargs):
        if data_format == 'default':
            data_format = K.image_data_format()
        self.size = tuple(size)

        if target_size is not None:
            self.target_size = tuple(target_size)
        else:
            self.target_size = None
        assert data_format in {'channels_last', 'channels_first'}, 'data_format must be in {tf, th}'

        self.data_format = data_format
        self.input_spec = [InputSpec(ndim=4)]

        super(BilinearUpSampling2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            width = int(self.size[0] * input_shape[2] if input_shape[2] is not None else None)
            height = int(self.size[1] * input_shape[3] if input_shape[3] is not None else None)

            if self.target_size is not None:
                width = self.target_size[0]
                height = self.target_size[1]
            return (input_shape[0],
                    input_shape[1],
                    width,
                    height)
        elif self.data_format == 'channels_last':
            width = int(self.size[0] * input_shape[1] if input_shape[1] is not None else None)
            height = int(self.size[1] * input_shape[2] if input_shape[2] is not None else None)

            if self.target_size is not None:
                width = self.target_size[0]
                height = self.target_size[1]
            return (input_shape[0],
                    width,
                    height,
                    input_shape[3])
        else:
            raise Exception('Invalid data_format: ' + self.data_format)

    def call(self, x, mask=None):
        if self.target_size is not None:
            return resize_images_bilinear(x, target_height=self.target_size[0], target_width=self.target_size[1], data_format=self.data_format)
        else:
            return resize_images_bilinear(x, height_factor=self.size[0], width_factor=self.size[1], data_format=self.data_format)

    def get_config(self):
        config = {'size': self.size, 'target_size': self.target_size}
        base_config = super(BilinearUpSampling2D, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

train_batch_size = 8
val_batch_size = 8
INPUT_SIZE = (224, 224)
AUG_PARAMS = [0.75, 1.25, 0.75, 1.25, 0.6, 1.4]
ANG_RANGE = 15

train_img_dir = '/content/drive/MyDrive/MobileNet/DataSet/train/images/'
train_mask_dir = '/content/drive/MyDrive/MobileNet/DataSet/train/masks/'
train_imgs = os.listdir(train_img_dir)# if you have an error take a look here ...
train_masks = os.listdir(train_mask_dir)
train_imgs= sorted([ i for i in train_imgs ])
train_masks= sorted([ i for i in train_masks ])
print("Image Size is : ", len(train_imgs))

val_img_dir = '/content/drive/MyDrive/MobileNet/DataSet/val/images/'
val_mask_dir = '/content/drive/MyDrive/MobileNet/DataSet/val/masks/'
val_imgs = os.listdir(val_img_dir)# if you have an error take a look here ...
val_masks = os.listdir(val_mask_dir)
val_imgs= sorted([ i for i in val_imgs ])
val_masks= sorted([ i for i in val_masks ])


train_trns = train_transforms(dataset='picsart', scale_size=INPUT_SIZE, ang_range=ANG_RANGE,
                                      augment_params=AUG_PARAMS, add_background=False,
                                      crop_scale=0.02)
val_trns = test_transforms(dataset='picsart', scale_size=INPUT_SIZE)
train_dataset = ImageTargetDataset(train_imgs,train_img_dir,train_masks,train_mask_dir,
                                           train_batch_size,
                                           shuffle=True,
                                           device='GPU:0',
                                           **train_trns)
val_dataset_hq = ImageTargetDataset(val_imgs,val_img_dir,val_masks,val_mask_dir,
                                           val_batch_size,
                                           shuffle=False,
                                           device='GPU:0',
                                           **val_trns)



print("Train dataset len:", len(train_dataset))
print("Val dataset len:", len(val_dataset_hq))

# def vis_dataset(dataset, num_samples=train_batch_size):
#     for x in dataset:
#         img, target = x[0], x[1]
#         for i in range(num_samples):
#             plt.imshow(img[i])
#             plt.imshow(np.squeeze(target[i]), alpha=0.4)
#             plt.show()
#         break
# vis_dataset(train_dataset, 8)

# Initialize model params
model_name = 'mobilenet_small'
n_class=1
old_model_path = None  # Or path to the previous saved model
# Train params
n_train = len(train_dataset)
n_val = len(val_dataset_hq)

loss_name = 'fb_combined'
optimizer = 'Adam'
lr = 0.0005
batch_size = train_batch_size
max_epoches = 1000
save_directory = '/content/drive/MyDrive/MobileNet'
reduce_factor = 0.995
epoches_limit = 5
early_stoping = 100
metrics = [FbSegm(channel_axis=-1)]
mobilenet_model = CustomModel(model_name='mobilenet_small',
                              n_class=1,
                              input_shape=(train_batch_size, INPUT_SIZE[0], INPUT_SIZE[1], 3),
                              old_model_path=None)

mobilenet_model.prepare_train(train_loader=train_dataset,
                              val_loader=val_dataset_hq,
                              n_train=n_train,
                              n_val=n_val,
                              loss_name=loss_name,
                              optimizer=optimizer,
                              lr = lr,
                              batch_size = batch_size,
                              max_epoches = max_epoches,
                              save_directory = save_directory,
                              reduce_factor=reduce_factor,
                              epoches_limit=epoches_limit,
                              early_stoping=early_stoping,
                              metrics=metrics)

mobilenet_model.fit()
mobilenet_model.validate(val_dataset_hq, n_val)
