from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt
import os

from ObjectDetection.keras_ssd7 import build_model
from ObjectDetection.keras_ssd_loss import SSDLoss
from ObjectDetection.keras_layer_AnchorBoxes import AnchorBoxes
from ObjectDetection.keras_layer_L2Normalization import L2Normalization
from ObjectDetection.ssd_box_encode_decode_utils import SSDBoxEncoder, decode_y, decode_y2
from ObjectDetection.ssd_batch_generator import BatchGenerator
from utils.image_process import dirname,size

flag={"create new model":True,"load_pre_weight":False,"predict":False}

img_height, img_width, img_channels = size[1], size[0], 3  # Height,Width,Number of color channels of the input images
n_classes = 1 # 正样本数
classes = ['backgroud', 'cup']
batch_size=8
epochs=0

# 训练集
# assert os.path.isdir(dirname)
train_images_dir=dirname
train_labels_filename = '../datasets/train.csv'

# 验证集（从label.csv剪切一些）
val_images_dir = '../datasets/img/'
val_labels_filename = '../datasets/val.csv'

def main():
    global img_height,img_width,img_channels,n_classes,batch_size,epochs,classes
    global train_images_dir,train_labels_filename,val_images_dir,val_labels_filename
    #%%
    #1、设置参数
    img_height,img_width,img_channels=img_height,img_width,img_channels # Height,Width,Number of color channels of the input images
    subtract_mean,divide_by_stddev = 127.5,127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval '[0,255]'->`[-127.5,127.5]`->`[-1,1]`.
    n_classes = n_classes # 正样本数
    scales = [0.08, 0.16, 0.32, 0.64, 0.96] # 预设框缩放尺寸，如果不是None，会覆盖min_Scale，max_Scale
    aspect_ratios = [0.5, 1.0, 2.0] # 预设框的宽高比
    two_boxes_for_ar1 = True # 是否要生成两个纵横比为1的锚框
    steps = None # 手动设置锚框网格的步长，不建议
    offsets = None # # 手动设置锚框网格的偏移量，则不建议
    limit_boxes = False # 是否要限制锚盒完全位于图像边界内
    variances = [1.0, 1.0, 1.0, 1.0] # 缩放编码目标坐标的方差列表
    coords = 'centroids' # 使用中心坐标还是最大最小框
    normalize_coords = False # 模型使用相对坐标？[0,1]

    #%%
    #编译模型
    if flag["create new model"]:
        print("建立并编译模型")
        K.clear_session() # 清理内存中的模型
        model = build_model(image_size=(img_height, img_width, img_channels),
                            n_classes=n_classes,
                            l2_regularization=0.0,
                            scales=scales,
                            aspect_ratios_global=aspect_ratios,
                            aspect_ratios_per_layer=None,
                            two_boxes_for_ar1=two_boxes_for_ar1,
                            steps=steps,
                            offsets=offsets,
                            limit_boxes=limit_boxes,
                            variances=variances,
                            coords=coords,
                            normalize_coords=normalize_coords,
                            subtract_mean=subtract_mean,
                            divide_by_stddev=divide_by_stddev,
                            swap_channels=False)

    else:
        # 直接载入编译好的模型
        # TODO: 设置模型路径
        model_path = 'ssd7.h5'
        # We need to create an SSDLoss object in order to pass that to the model loader.
        ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
        K.clear_session()
        model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                       'compute_loss': ssd_loss.compute_loss})
        pass
    if flag["load_pre_weight"]:
        model.load_weights('./ssd7_weights.h5')
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)
    ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

    #%%
    # 准备数据
    # 1、输出格式，两份数据集
    train_dataset = BatchGenerator(box_output_format=['class_id', 'xmin', 'ymin', 'xmax', 'ymax'])
    val_dataset = BatchGenerator(box_output_format=['class_id', 'xmin', 'ymin', 'xmax', 'ymax'])
    # 2、读取数据

    # TODO: 放我的数据

    # 训练集
    train_images_dir,train_labels_filename = train_images_dir,train_labels_filename

    # 验证集（从label.csv剪切一些）
    val_images_dir,val_labels_filename = val_images_dir, val_labels_filename
    # TODO : 这个input_format需要写到csv文件里
    train_dataset.parse_csv(images_dir=train_images_dir,
                            labels_filename=train_labels_filename,
                            input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                            #CSV文件6列的标签，如果用XML，用parse_xml可能更好，看文档。
                            include_classes='all')

    val_dataset.parse_csv(images_dir=val_images_dir,
                          labels_filename=val_labels_filename,
                          input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'],
                          include_classes='all')

    # In[7]:

    # 3: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.
    # 不同特征图创建不同框大小
    # The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
    predictor_sizes = [model.get_layer('classes4').output_shape[1:3],
                       model.get_layer('classes5').output_shape[1:3],
                       model.get_layer('classes6').output_shape[1:3],
                       model.get_layer('classes7').output_shape[1:3]]

    ssd_box_encoder = SSDBoxEncoder(img_height=img_height,
                                    img_width=img_width,
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_global=aspect_ratios,
                                    aspect_ratios_per_layer=None,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    limit_boxes=limit_boxes,
                                    variances=variances,
                                    pos_iou_threshold=0.5,
                                    neg_iou_threshold=0.2,
                                    coords=coords,
                                    normalize_coords=normalize_coords)

    # 4、每批数量

    batch_size = batch_size  # Change the batch size if you like, or if you run into memory issues with your GPU.

    # 5: Set the image processing / data augmentation options and create generator handles.

    # Change the online data augmentation settings as you like
    train_generator = train_dataset.generate(batch_size=batch_size,
                                             shuffle=True,
                                             train=True,
                                             ssd_box_encoder=ssd_box_encoder,
                                             convert_to_3_channels=True,
                                             equalize=False,
                                             brightness=(0.5, 2, 0.5),
                                             # Randomly change brightness between 0.5 and 2 with probability 0.5
                                             flip=0.5,  # Randomly flip horizontally with probability 0.5
                                             translate=((5, 50), (3, 30), 0.5),
                                             # Randomly translate by 5-50 pixels horizontally and 3-30 pixels vertically with probability 0.5
                                             scale=(0.75, 1.3, 0.5),
                                             # Randomly scale between 0.75 and 1.3 with probability 0.5
                                             max_crop_and_resize=False,
                                             random_pad_and_resize=False,
                                             random_crop=False,
                                             crop=False,
                                             resize=False,
                                             gray=False,
                                             limit_boxes=True,
                                             include_thresh=0.4)

    val_generator = val_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         train=True,
                                         ssd_box_encoder=ssd_box_encoder,
                                         convert_to_3_channels=True,
                                         equalize=False,
                                         brightness=False,
                                         flip=False,
                                         translate=False,
                                         scale=False,
                                         max_crop_and_resize=False,
                                         random_pad_and_resize=False,
                                         random_crop=False,
                                         crop=False,
                                         resize=False,
                                         gray=False,
                                         limit_boxes=True,
                                         include_thresh=0.4)

    # 得到样本数量
    n_train_samples = train_dataset.get_n_samples()
    n_val_samples = val_dataset.get_n_samples()

    # 训练
    # TODO: 设置迭代轮数
    epochs = epochs

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=ceil(n_train_samples / batch_size),
                                  epochs=epochs,
                                  callbacks=[ModelCheckpoint('./model/ssd7_weights_epoch-{epoch:02d}_loss-{loss:.4f}.h5',
                                                             monitor='val_loss',
                                                             verbose=1,
                                                             save_best_only=True,
                                                             save_weights_only=True,
                                                             mode='auto',
                                                             period=5),
                                             EarlyStopping(monitor='val_loss',
                                                           min_delta=0.001,
                                                           patience=5),
                                             ReduceLROnPlateau(monitor='val_loss',
                                                               factor=0.5,
                                                               patience=0,
                                                               epsilon=0.001,
                                                               cooldown=0)],
                                  validation_data=val_generator,
                                  validation_steps=ceil(n_val_samples / batch_size))

    # TODO: Set the filename (without the .h5 file extension!) under which to save the model and weights.
    #       Do the same in the `ModelCheckpoint` callback above.
    model_name = 'ssd7'
    # 创建了新模型
    if flag["create new model"]:
        model.save('{}.h5'.format(model_name))
    model.save_weights('{}_weights.h5'.format(model_name))

    print()
    print("Model saved under {}.h5".format(model_name))
    print("Weights also saved separately under {}_weights.h5".format(model_name))
    print()

    if flag["predict"]:
        # 1: Set the generator

        predict_generator = val_dataset.generate(batch_size=1,
                                                 shuffle=True,
                                                 train=False,
                                                 returns={'processed_labels',
                                                          'filenames'},
                                                 convert_to_3_channels=True,
                                                 equalize=False,
                                                 brightness=False,
                                                 flip=False,
                                                 translate=False,
                                                 scale=False,
                                                 max_crop_and_resize=False,
                                                 random_pad_and_resize=False,
                                                 random_crop=False,
                                                 crop=False,
                                                 resize=False,
                                                 gray=False,
                                                 limit_boxes=True,
                                                 include_thresh=0.4)

        # In[ ]:

        # 2: Generate samples

        X, y_true, filenames = next(predict_generator)

        i = 0  # 测试项目

        print("Image:", filenames[i])
        print()
        print("Ground truth boxes:\n")
        print(y_true[i])

        # In[ ]:

        # 3: Make a prediction

        y_pred = model.predict(X)

        # 现在让我们解码原始预测‘y_pred’。
        # 函数`decode_y2()‘将盒坐标从偏移转换回绝对坐标，
        # 只保留正面的预测(即扔掉所有对0级最有信心的框)，
        # 对所有积极的预测应用一个置信阈值，
        # 并按此顺序对其余的预测应用非最大抑制。
        # 如果您想省略NMS步骤，请设置‘iou_阈值=None’。
        #
        # You could also use `decode_y()`, which follows the prodecure outlined in the paper, to decode the raw predictions.
        # The main way in which `decode_y()` and `decode_y2()` differ is that `decode_y2()` performs NMS globally and `decode_y()` performs NMS per class.
        # It is important to understand what difference that makes. One point is that doing NMS per class for 20 classes will take roughly 20-times the time to do NMS just once over all classes,
        # but this slow-down doesn't matter much when decoding a single batch. The more important point is to understand what difference it can make for the resulting final predictions.
        # Performing NMS globally means that the strongest candidate box will eliminate all close boxes around it regardless of their predicted class.
        # This can be good and bad. For example, if one box correctly predicts a sheep and another box incorrectly predicts a cow at similar coordinates,
        # then global NMS would eliminate the incorrect cow box (because it is too close to the correct sheep box), while per-class NMS would not eliminate the incorrect cow box (because boxes are only compared within the same object class).
        # On the other hand, if two objects of different classes are very close together and overlapping and are predicted correctly, then global NMS might eliminate one of the two correct predictions because they are too close together, while per-class NMS will keep both predictions.
        # It's up to you which decoder you use.

        # 4: Decode the raw prediction `y_pred`

        y_pred_decoded = decode_y2(y_pred,
                                   confidence_thresh=0.6,
                                   iou_threshold=0.4,
                                   # top_k='all',
                                   top_k=3,
                                   input_coords='centroids',
                                   normalize_coords=False,
                                   img_height=None,
                                   img_width=None)

        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        print("Decoded predictions (output format is [class_id, confidence, xmin, ymin, xmax, ymax]):\n")
        print(y_pred_decoded[i])
        pass

    #%%
    # 5、画出来

    plt.figure(figsize=(15, 10))
    plt.imshow(X[i])

    current_axis = plt.gca()

    # classes = ['background', 'car', 'truck', 'pedestrian', 'bicyclist', 'light'] # Just so we can print class names onto the image instead of IDs
    classes = classes
    # Draw the ground truth boxes in green (omit the label for more clarity)
    for box in y_true[i]:
        xmin = box[1]
        ymin = box[2]
        xmax = box[3]
        ymax = box[4]
        label = '{}'.format(classes[int(box[0])])
        current_axis.add_patch(
            plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color='green', fill=False, linewidth=2))
        # current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})

    # Draw the predicted boxes in blue
    for box in y_pred_decoded[i]:
        xmin = box[-4]
        ymin = box[-3]
        xmax = box[-2]
        ymax = box[-1]
        # color = colors[int(box[0])]
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        current_axis.add_patch(
            plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color='red', fill=False, linewidth=2))
        current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': 'blue', 'alpha': 1.0})
    plt.show()

if __name__ == '__main__':
    main()