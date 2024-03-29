# Import TF and TF Hub libraries.
import tensorflow as tf
from tensorflow._api.v2 import data
from tensorflow.python.util.dispatch import _signature_from_annotations
import tensorflow_addons as tfa
from filters import gaussian_filter2d
from tensorflow.keras import layers, optimizers, losses, metrics, Model
import tensorflow.keras.applications.mobilenet_v2 as mobilenet_v2

from pycocotools.coco import COCO
import numpy as np
import cv2
from typing import Dict, List
import datetime
import argparse

# Ordering: TL TR BL BR

# 576x384 (wxh)
# Input shape: hxw
def create_model(input_shape=(384,576,3)) -> Model:
    i = layers.Input(shape=input_shape)
    x = tf.cast(i, tf.float32)
    x = mobilenet_v2.preprocess_input(x)

    backbone = mobilenet_v2.MobileNetV2(input_tensor=x, include_top=False, weights='imagenet')
    backbone.trainable = False
    x = backbone(x)

    # Get layers and attach FPN
    backbone_scale_2 = backbone.get_layer('expanded_conv_project_BN').output
    backbone_scale_3 = backbone.get_layer('block_2_add').output
    backbone_scale_4 = backbone.get_layer('block_5_add').output
    backbone_scale_5 = backbone.get_layer('block_12_add').output
    backbone_scale_6 = x
    # backbone_scale_6 = backbone.get_layer('block_16_project_BN').output

    # FPN
    fpn_d = 128
    # m7 = layers.Conv2D(fpn_d, 1, padding='same')(backbone_scale_7)
    # p7 = m7
    # p7_up = layers.UpSampling2D(size=(32,32))(p7)

    m6 = layers.Conv2D(fpn_d, 1, padding='same', name='m6')(backbone_scale_6)
    p6 = layers.Conv2D(fpn_d, 3, padding='same', name='p6')(m6)
    p6 = layers.Conv2D(fpn_d, 3, padding='same')(p6)
    # p6 = layers.Conv2D(fpn_d, 3, padding='same')(p6)
    p6_up = layers.UpSampling2D(size=(16,16))(p6)

    y = layers.Conv2D(fpn_d, 1, padding='same')(backbone_scale_5)
    z = layers.UpSampling2D()(m6)
    m5 = layers.Add(name='m5', )([y,z])
    p5 = layers.Conv2D(fpn_d, 3, padding='same', name='p5')(m5)
    p5 = layers.Conv2D(fpn_d, 3, padding='same')(p5)
    # p5 = layers.Conv2D(fpn_d, 3, padding='same')(p5)
    p5_up = layers.UpSampling2D(size=(8,8))(p5)

    y = layers.Conv2D(fpn_d, 1, padding='same')(backbone_scale_4)
    z = layers.UpSampling2D()(m5)
    m4 = layers.Add(name='m4', )([y,z])
    p4 = layers.Conv2D(fpn_d, 3, padding='same', name='p4')(m4)
    p4 = layers.Conv2D(fpn_d, 3, padding='same')(p4)
    # p4 = layers.Conv2D(fpn_d, 3, padding='same')(p4)
    p4_up = layers.UpSampling2D(size=(4,4))(p4)

    y = layers.Conv2D(fpn_d, 1, padding='same')(backbone_scale_3)
    z = layers.UpSampling2D()(m4)
    m3 = layers.Add(name='m3', )([y,z])
    p3 = layers.Conv2D(fpn_d, 3, padding='same', name='p3')(m3)
    p3 = layers.Conv2D(fpn_d, 3, padding='same')(p3)
    # p3 = layers.Conv2D(fpn_d, 3, padding='same')(p3)
    p3_up = layers.UpSampling2D()(p3)

    y = layers.Conv2D(fpn_d, 1, padding='same')(backbone_scale_2)
    z = layers.UpSampling2D()(m3)
    m2 = layers.Add(name='m2', )([y,z])
    p2 = layers.Conv2D(fpn_d, 3, padding='same', name='p2')(m2)
    p2 = layers.Conv2D(fpn_d, 3, padding='same')(p2)
    # p2 = layers.Conv2D(fpn_d, 3, padding='same')(p2)

    # Concatenate Outputs of FPN
    # x = layers.Concatenate()([p3, p4_up, p5_up, p6_up])
    x = layers.Concatenate()([p2, p3_up, p4_up, p5_up, p6_up])
    x = layers.Dropout(0.1)(x)

    # Boil it down to four heatmaps
    x = layers.Conv2D(fpn_d, 3, padding='same')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv2D(fpn_d, 3, padding='same')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv2D(fpn_d, 3, padding='same')(x)
    # x = layers.Conv2D(fpn_d, 3, padding='same', activation='relu')(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Conv2D(1, 1, padding='same')(x)
    # x = layers.Conv2D(5, 1, padding='same')(x)
    # x = layers.UpSampling2D(interpolation='bilinear', size=(4,4))(x)

    model = Model(inputs=i, outputs=x)
    return model, backbone

def get_corner_points(polygon: np.ndarray) -> List:
    # Ordering: TL TR BL BR
    # x,y
    points = polygon[:,0]

    sorted_x = points[np.argsort(points[:, 0])]
    lefts = sorted_x[0:2]
    rights = sorted_x[2:]

    sorted_y = points[np.argsort(points[:, 1])]
    tops = sorted_y[0:2]
    bottoms = sorted_y[2:]

    # If lefts or rights are equal to tops or bottoms, then we need to assign manually
    if ( np.array_equal(tops, lefts) or np.array_equal(tops, np.flip(lefts, 0))
      or np.array_equal(tops, rights) or np.array_equal(tops, np.flip(rights, 0))
      or np.array_equal(bottoms, lefts) or np.array_equal(bottoms, np.flip(lefts, 0))
      or np.array_equal(bottoms, rights) or np.array_equal(bottoms, np.flip(rights, 0)) ):
        # Sort left and right by y
        lefts_sorted_y = lefts[np.argsort(lefts[:, 1])]
        rights_sorted_y = rights[np.argsort(rights[:, 1])]
        tl = lefts_sorted_y[0,:]
        bl = lefts_sorted_y[1,:]
        tr = rights_sorted_y[0,:]
        br = rights_sorted_y[1,:]

    else:
        tl = tops[np.nonzero(np.all(tops == lefts, 1) | np.all(tops == np.flip(lefts, 0), 1)), :].flatten()
        tr = tops[np.nonzero(np.all(tops == rights, 1) | np.all(tops == np.flip(rights, 0), 1)), :].flatten()
        bl = bottoms[np.nonzero(np.all(bottoms == lefts, 1) | np.all(bottoms == np.flip(lefts, 0), 1)), :].flatten()
        br = bottoms[np.nonzero(np.all(bottoms == rights, 1) | np.all(bottoms == np.flip(rights, 0), 1)), :].flatten()

    corners = np.array([tl, tr, bl, br])

    assert(corners.shape == (4,2))

    return corners.tolist()

def create_coco_dataset(set: str, min_area_percentage: float = 0.05):
    """Returns a tf dataset with (image_file_name, corners)

    Args:
        set (str): 'train' or 'val
    """
    if set != 'train' and set != 'val':
        print("Error: Invalid dataset requested")
        exit

    dataDir = 'd:/datasets/coco'
    dataType = '{}2017'.format(set)
    annDir = '{}/annotations'.format(dataDir)
    annFile = '{}/instances_{}.json'.format(annDir, dataType)
    print ("Will use annotations in " + annFile)

    coco = COCO(annFile)
    category_ids = coco.getCatIds(catNms=['tv'])
    img_ids = coco.getImgIds(catIds=category_ids)

    # Only use those annotations with only four keypoints
    ann_ids = coco.getAnnIds(imgIds=img_ids, catIds=category_ids, iscrowd=None)
    annotations = coco.loadAnns(ann_ids)

    image_files = []
    labels = []

    for ann in annotations:
        # Filter out images containing multiple TVs
        annsForThisImage = coco.loadAnns(coco.getAnnIds(imgIds=[ann['image_id']], catIds=category_ids))
        if len(annsForThisImage) > 1:
            continue

        mask = coco.annToMask(ann)
        # Create a polygon from this binary mask
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]
        # Estimate a polygon for the contour
        epsilon = 0.05*cv2.arcLength(contour, True)
        estimated_polygon = cv2.approxPolyDP(contour, epsilon, True)

        num_points = estimated_polygon.shape[0]

        if num_points == 4 :
            coco_img = coco.loadImgs(ann['image_id'])[0]

            if cv2.contourArea(estimated_polygon) >= min_area_percentage * coco_img['width'] * coco_img['height']:
                image_file = '{}/images/{}/{}'.format(dataDir, dataType, coco_img['file_name'])
                image_files.append(image_file)

                labels.append(get_corner_points(estimated_polygon))

            # color = (0,255,0)

        # # For debugging: load image, draw contours and show it
        # img = coco.loadImgs(ann['image_id'])[0]
        # image = cv2.imread('{}/images/{}/{}'.format(dataDir, dataType, img['file_name']))
        # cv2.drawContours(image, [estimated_polygon], 0, color, 3)
        # cv2.imshow('Contours overlay', image)
        # cv2.waitKey(1)

    files_tensor = tf.constant(image_files)
    labels_tensor = tf.constant(labels)
    dataset = tf.data.Dataset.from_tensor_slices((files_tensor, labels_tensor))

    print("Size of Dataset: {} images".format(len(dataset)))

    for element in dataset.as_numpy_iterator():
        print(element)
        break

    return dataset

def get_padding_params(height, width, target_height, target_width):
    # convert values to float, to ease divisions
    f_height = tf.cast(height, dtype=tf.float32)
    f_width = tf.cast(width, dtype=tf.float32)
    f_target_height = tf.cast(target_height, dtype=tf.float32)
    f_target_width = tf.cast(target_width, dtype=tf.float32)

    # Find the ratio by which the image must be adjusted
    # to fit within the target
    ratio = tf.maximum(f_width / f_target_width, f_height / f_target_height)
    resized_height_float = f_height / ratio
    resized_width_float = f_width / ratio
    resized_height = tf.cast(
        tf.floor(resized_height_float), dtype=tf.int32)
    resized_width = tf.cast(
        tf.floor(resized_width_float), dtype=tf.int32)

    padding_height = (f_target_height - resized_height_float) / 2
    padding_width = (f_target_width - resized_width_float) / 2
    f_padding_height = tf.floor(padding_height)
    f_padding_width = tf.floor(padding_width)
    p_height = tf.maximum(0, tf.cast(f_padding_height, dtype=tf.int32))
    p_width = tf.maximum(0, tf.cast(f_padding_width, dtype=tf.int32))

    return resized_height, resized_width, p_height, p_width

def draw_poly(image, poly):
    img = image.numpy()
    cv2.fillConvexPoly(img, poly.numpy(), (1), cv2.LINE_AA)
    return img

def make_loading_and_heatmap_generation_function(target_width, target_height, sigma=tf.Variable):
    def load_img_and_make_heatmaps(filename, corners):
        image = tf.io.read_file(filename)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.uint8)

        image_shape = tf.shape(image)

        # Get padding offsets
        resized_height, resized_width, padding_y, padding_x = get_padding_params(image_shape[0], image_shape[1], target_height, target_width)

        # Add slice index to corners
        corners = tf.reverse(corners, axis=[1])
        corners = tf.cast(corners, tf.float32)

        # Adjust corner coordinates
        corners_y = tf.cast(corners[:,0] * tf.cast((resized_height / image_shape[0]), tf.float32) + tf.cast(padding_y, tf.float32), tf.int32)
        corners_x = tf.cast(corners[:,1] * tf.cast((resized_width / image_shape[1]), tf.float32) + tf.cast(padding_x, tf.float32), tf.int32)

        corner_slice_indices = tf.constant([0,1,2,3])
        corner_slice_indices = tf.expand_dims(corner_slice_indices, axis=1)
        corners = tf.stack([corners_y, corners_x], axis=1)
        # corners = tf.expand_dims(corners, axis=0)

        def create_heatmap(corner):
            # Make heatmap indices
            heatmap_y = tf.range(0, target_height)
            heatmap_x = tf.range(0, target_width)

            heatmap_indices = tf.meshgrid(heatmap_y, heatmap_x, indexing='ij')
            heatmap_indices = tf.stack(heatmap_indices, axis=2)

            sub = heatmap_indices - corner
            d = tf.norm(tf.cast(sub, tf.float32), axis=2) ** 2
            heatmap = tf.exp(- d / (tf.cast(sigma, tf.float32)**2))

            return heatmap

        heatmaps = tf.map_fn(create_heatmap, corners, fn_output_signature=tf.TensorSpec(shape=(target_height, target_width), dtype=tf.float32))

        heatmaps = tf.stack(tf.unstack(heatmaps, axis=0), axis=2)
        
        # Make a polygon
        corners_reorded = tf.gather(corners, [0,1,3,2])
        corners_reorded = tf.reverse(corners_reorded, axis=[1])
        poly = np.zeros((target_height, target_width), np.float32)
        [poly,] = tf.py_function(draw_poly, [poly, corners_reorded],[tf.float32])

        poly = tf.convert_to_tensor(poly)
        poly = tf.expand_dims(poly, axis=2)
        heatmaps = tf.concat([heatmaps, poly], axis=2)

        heatmaps = heatmaps / tf.reduce_max(heatmaps)
        heatmaps = tf.image.resize(heatmaps, (target_height // 2, target_width // 2))

        heatmaps = tf.gather(heatmaps, [4], axis=2)

        image = tf.image.resize_with_pad(image, target_height, target_width)

        return image, heatmaps

    return load_img_and_make_heatmaps

# # Load the input image.
# image_path = 'PATH_TO_YOUR_IMAGE'
# image = tf.io.read_file(image_path)
# image = tf.compat.v1.image.decode_jpeg(image)
# image = tf.expand_dims(image, axis=0)
# # Resize and pad the image to keep the aspect ratio and fit the expected size.
# image = tf.cast(tf.image.resize_with_pad(image, 192, 192), dtype=tf.int32)

# Download the model from TF Hub.


# # Run model inference.
# outputs = movenet(image)
# # Output is a [1, 1, 17, 3] tensor.
# keypoints = outputs['output_0']

def show(image, heatmaps):
    if isinstance(image, tf.Tensor):
        image = image.numpy()

    image = image / 255.0
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if isinstance(heatmaps, tf.Tensor):
        heatmaps = heatmaps.numpy()

    # heatmaps = [heatmaps[:,:,0], heatmaps[:,:,1], heatmaps[:,:,2], heatmaps[:,:,3], heatmaps[:,:,4]]
    heatmaps = [heatmaps[:,:,0]]

    colors = [(0,0,1), (1,0,0), (0,1,1), (0,1,0), (0,0,1)]
    # TL = Red, TR = Blue, BL = Yellow, BR = Green

    heatmaps_combined = np.zeros(image.shape)
    for i, heatmap in enumerate(heatmaps):
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
        heatmap *= np.array(colors[i])
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmaps_combined += heatmap
    
    image = 1 - (1-image) * (1-heatmaps_combined)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
#   plt.title(heatmaps.numpy().decode('utf-8'))

def parse_args():
    parser = argparse.ArgumentParser("trainer")

    parser.add_argument('--load-model-path', type=str)
    parser.add_argument('--eval-qual', action='store_true')
    parser.add_argument('--train', action='store_true')

    args = parser.parse_args()
    return args

def create_heatmap_dataset(files_dataset, image_width, image_height, sigma):
    load_img_and_make_heatmaps = make_loading_and_heatmap_generation_function(image_width, image_height, sigma)
    files_dataset = files_dataset.shuffle(len(files_dataset))
    dataset = files_dataset.map(load_img_and_make_heatmaps, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

class AdjustSigmaCallback(tf.keras.callbacks.Callback):
    def __init__(self, sigma: tf.Variable):
        super(AdjustSigmaCallback, self).__init__()
        self.sigma = sigma

    def on_epoch_end(self, epoch=None, logs=None):
        if epoch >= 14 and epoch % 4 == 0:
            self.sigma.assign(self.sigma - 2 if self.sigma > 7 else self.sigma)
        
        tf.print(" - Sigma: ", self.sigma)

class MakeBackboneTrainableCallback(tf.keras.callbacks.Callback):
    def __init__(self, backbone: tf.keras.Model):
        super(MakeBackboneTrainableCallback, self).__init__()
        self.backbone = backbone
    
    def on_epoch_end(self, epoch=None, logs=None):
        if epoch == 14:
            self.backbone.trainable = True
            print(" - Making backbone trainable.")


if __name__ == "__main__":
    args = parse_args()

    batch_size = 16

    # image_size = (576,384)
    image_size = (288,192)
    image_width, image_height = image_size

    sigma = tf.Variable(20)

    adjust_sigma_cb = AdjustSigmaCallback(sigma)

    # Create datasets with filenames and corner points
    # Then map these datasets onto an image loader and label generator function
    val_files_dataset = create_coco_dataset('val')

    # --------------------------------- Debugging -------------------------------- #
    # load_img_and_make_heatmaps = make_loading_and_heatmap_generation_function(image_width, image_height, sigma)
    # val_files_dataset = val_files_dataset.shuffle(len(val_files_dataset))

    # # for file, corners in val_files_dataset.take(5):
    # #     image, heatmaps = load_img_and_make_heatmaps(file, corners)
    # #     show(image, heatmaps)

    # val_dataset = val_files_dataset.map(load_img_and_make_heatmaps, num_parallel_calls=tf.data.AUTOTUNE)
    # for image, heatmaps in val_dataset.take(5):
    #     show(image, heatmaps)

    # # sigma.assign(3)
    # # adjust_sigma_cb.on_epoch_end()
    # # adjust_sigma_cb.on_epoch_end()
    # # adjust_sigma_cb.on_epoch_end()
    # # adjust_sigma_cb.on_epoch_end()

    # # for image, heatmaps in val_dataset.take(5):
    # #     show(image, heatmaps)
    # ------------------------------- Debugging End ------------------------------ #

    val_dataset = create_heatmap_dataset(val_files_dataset, image_width, image_height, sigma)

    train_files_dataset = create_coco_dataset('train')
    train_dataset = create_heatmap_dataset(train_files_dataset, image_width, image_height, sigma)

    # TODO: Later on perform curriculum learning by only training on close-up images of the TVs, utilizing the area of the polygon in comparison to the image

    if args.load_model_path and args.load_model_path is not None:
        print("Loading model from: ", args.load_model_path)
        model = tf.keras.models.load_model(args.load_model_path)
    else:
        print("Creating model.")
        model, backbone = create_model(input_shape=(image_height, image_width, 3))
        make_backbone_trainable_cb = MakeBackboneTrainableCallback(backbone)

    model.summary()
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss=losses.MeanSquaredError())

    if args.train:

        # subpath = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        subpath = 'baseline'

        log_dir = "logs/fit/" + subpath
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            update_freq='batch',
            # profile_batch=(10,20)
            )

        checkpoint_path = "checkpoints/"+subpath+"/checkpoint"

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            save_best_only=True)

        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.1, 
            patience=2, 
            verbose=1,
            mode='min',
            min_lr = 0.0000001)

        history = model.fit(
            x=train_dataset,
            epochs=50,
            steps_per_epoch=len(train_files_dataset) // batch_size,
            validation_data=val_dataset,
            validation_steps=len(val_files_dataset) // batch_size,
            callbacks=[tensorboard_callback, lr_callback, make_backbone_trainable_cb, adjust_sigma_cb]
            # callbacks=[tensorboard_callback, model_checkpoint_callback, lr_callback]
            # callbacks=[tensorboard_callback, model_checkpoint_callback, lr_callback, adjust_sigma_cb]
            )
    
        model.save(checkpoint_path)

    if args.eval_qual:
        val_dataset = val_dataset.unbatch()
        for image, heatmaps in val_dataset.take(100):
            exp_image = tf.expand_dims(image, axis=0)
            pred_heatmaps = model(exp_image)
            # pred_heatmaps = model.predict(exp_image)
            pred_heatmaps = tf.squeeze(pred_heatmaps, axis=0)
            pred_heatmaps = pred_heatmaps / tf.reduce_max(pred_heatmaps, axis=[0,1,2])
            show(image, pred_heatmaps)
