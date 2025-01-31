# Basic library
import math
import os
import numpy as np
import pickle
from io import BytesIO
import gc

# Statistic library
from scipy import stats

# Image preprocessing
import cv2
import PIL
from PIL import Image, ImageOps
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

# Deep learning library for pre&post processing
# import torch.nn.functional as F
# from torch import Tensor
# from torch.autograd import Variable
import onnxruntime

# Local library
from dataset import *

AI_VERSION = "WellXray - Invitrace"
DISCLAIMER = """Disclaimer:
This Artificially Intelligent (AI) system is intended to be used for supporting the chest radiographic interpretation in the case of Lung Opacity. The display results include only the relevant but neither specific nor supporting all significant findings. The final diagnosis must be correlated with the clinical data. This AI cannot substitute the standard radiologic report by the qualified radiologist.
The AI results cannot be constructed as a statement  and cannot be used for any legal purposes."""

# Get the directory of the current script or module
current_dir = os.path.dirname(os.path.abspath(__file__))

# checkpoint = 'save/pylon_densenet169_ImageNet_1024/0/best'
# name = 'pylon_densenet169_ImageNet_1024_selectRad_V2'
name = 'pylon_resnet50_vin1024'
path_checkpoint = f'save/onnx/{name}.onnx'
checkpoint = os.path.join(current_dir, path_checkpoint)
# checkpoint_flush = f'save/onnx/{name}_flush.onnx'

size=1024
interpolation='cubic'

# setting device on GPU if available, else CPU
# https://stackoverflow.com/questions/48152674/how-to-check-if-pytorch-is-using-the-gpu
# dev = 'cuda' if GPUtil.getAvailable() else 'cpu'
dev = 'cpu'
print('Using device:', dev)
print()
#Additional Info when using cuda
# if dev.type == 'cuda':
#     print(torch.cuda.get_device_name(0))
#     print('Memory Usage:')
#     print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
#     print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

print('onnxruntime device:', onnxruntime.get_device())

# eval_transform = make_transform('eval',
#                                 size=size,
#                                 interpolation=interpolation)

# threshold_df = pd.read_json(f'./save/threshold/{name}_combine_threshold.json')
threshold_path = 'save/threshold/vin_cls_v3_val_threshold.json'
threshold_df = pd.read_json(os.path.join(current_dir, threshold_path))
threshold_dict = threshold_df['G-Mean'].to_dict()
# threshold_dict = threshold_df['F1_Score'].to_dict()

# Temporary adjust Pneumothorax threshold
threshold_dict['Pneumothorax'] *= 2

CATEGORIES = list(threshold_dict.keys())
# print(CATEGORIES)
class_dict = {cls:i for i, cls in enumerate(CATEGORIES)}
# df_json = pd.read_json('./save/temperature_parameter.json')
# temperature_dict = df_json['Temperature'].to_dict()

# Val prediction for make percentile
# with open('save/val_predict/out_val_selectRad_data_pylon_densenet169_ImageNet_1024_V2_0.p', 'rb') as fp:
#     val_predict = pickle.load(fp)
# pred_val = val_predict['pred']


class_proba = [finding+'_proba' for finding in CATEGORIES]

print("Current directory:", os.getcwd())
val_path = 'save/val_predict/vin_cls_v3_val.csv'
pred_val = pd.read_csv(os.path.join(current_dir, val_path), usecols=class_proba).values

focusing_finding = [
    # 'Aortic enlargement', 
    # 'Atelectasis', 
    # 'Calcification', 
    'Cardiomegaly', 
    # 'Consolidation', 
    # 'ILD', 
    # 'Infiltration', 
    'Lung Opacity', 
    'Nodule/Mass', 
    # 'Other lesion', 
    'Pleural effusion', 
    # 'Pleural thickening', 
    'Pneumothorax', 
    # 'Pulmonary fibrosis', 
    # 'No finding'
    ]
focusing_finding_dict = {cls:i for i, cls in enumerate(focusing_finding)}

def sigmoid_array(x):
    # if x >= 0:
    #     return 1 / (1 + np.exp(-x)) >> RuntimeWarning: overflow encountered in exp
    # else:
    return np.exp(x)/(1+np.exp(x))  

def read_imagefile(file) -> PIL.Image.Image:
    image = PIL.Image.open(BytesIO(file))
    return image

# Convert DICOM to Numpy Array
def dicom2array(file, voi_lut=True, fix_monochrome=True):
    """Convert DICOM file to numy array
    
    Args:
        file : input object or uploaded file
        path (str): Path to the DICOM file to be converted
        voi_lut (bool): Whether or not VOI LUT is available
        fix_monochrome (bool): Whether or not to apply MONOCHROME fix
        
    Returns:
        Numpy array of the respective DICOM file
    """
    
    # Use the pydicom library to read the DICOM file

    if not isinstance(file, (pydicom.FileDataset, pydicom.dataset.Dataset)):
        try: # If file is uploaded with fastapi.UploadFile
            path = BytesIO(file)
            dicom = pydicom.read_file(path)
        except: # If file is uploaded with streamlit.file_uploader
            dicom = pydicom.read_file(file)
    else: # if file is readed dicom file
        dicom = file
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
        
    # Depending on this value, X-ray may look inverted - fix that
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    # Normalize the image array
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    
    return dicom, data


def preprocess(image):
    _res = eval_transform(image=np.array(image))
    image = np.float32(_res)
    # image = Variable(image, requires_grad=False)
    # image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    image = np.expand_dims(image, axis=(0, 1))
    return image

net_predict = onnxruntime.InferenceSession(checkpoint)

def get_all_pred_df(CATEGORIES, y_calibrated, y_uncalibrated, threshold_dict):
    result_dict = {}
    result_dict['Finding'] = []
    result_dict['Threshold'] = []
    result_dict['Raw_Pred'] = []
    result_dict['Confidence'] = []
    result_dict['isPositive'] = []
    # for pred_cls, prob in zip(all_pred_class, all_prob_class):
    for pred_cls, calibrated_prob, uncalibrated_prob in zip(CATEGORIES, np.array(y_calibrated.ravel()), np.array(y_uncalibrated.ravel())):
        result_dict['Finding'].append(pred_cls)
        result_dict['Threshold'].append(threshold_dict[pred_cls])
        result_dict['Raw_Pred'].append(uncalibrated_prob)
        result_dict['Confidence'].append(calibrated_prob)
        result_dict['isPositive'].append(bool(uncalibrated_prob>=threshold_dict[pred_cls]))

    all_pred_df = pd.DataFrame(result_dict)
    return all_pred_df

# จากปัญหาเรื่อง overconfidence เลยจำเป็นต้อง manual post process แบบนี้ คือ กำหนดให้ค่า pred_prob ที่ threshold เป็น pctile ที่ 50 ไปเลย
def calibrate_prob(pred, pred_val, c_name, kind='rank'):
    thredshold = threshold_dict[c_name]
    i_class = class_dict[c_name]

    pred_val_c_name = pred_val[:,i_class]
    y_pred_val = pred_val.copy()

    is_over_threshold = pred[:,i_class] >= thredshold
    if is_over_threshold:
        y_pred_val[:,i_class][y_pred_val[:,i_class] >= thredshold] = 1
        y_pred_val[:,i_class][y_pred_val[:,i_class] < thredshold] = 0
        y_pred_bool = y_pred_val[:,i_class].astype(bool)
        pred_array_sel_threshold = pred_val_c_name[y_pred_bool]
        result = stats.percentileofscore(pred_array_sel_threshold, pred[:,i_class], kind=kind)
        result = 50 + result/2
    else:
        y_pred_val[:,i_class][y_pred_val[:,i_class] >= thredshold] = 0
        y_pred_val[:,i_class][y_pred_val[:,i_class] < thredshold] = 1
        y_pred_bool = y_pred_val[:,i_class].astype(bool)
        pred_array_sel_threshold = pred_val_c_name[y_pred_bool]
        result = stats.percentileofscore(pred_array_sel_threshold, pred[:,i_class], kind=kind)
        result = result/2
    # print('Min value of this class:',min(pred_array_over_threshold), 
    #       '\nMax value of this class:', max(pred_array_over_threshold))
    # print(pred[:,i_class])
    
    return result

def predict(image, net_predict, threshold_dict, class_dict):
    
    ort_session = net_predict
    ort_inputs = {ort_session.get_inputs()[0].name: image}
    out = ort_session.run(None, ort_inputs)
    pred ,seg = out
    # pred = torch.from_numpy(pred)
    # seg= torch.from_numpy(seg)
    pred, seg = sigmoid_array(pred), sigmoid_array(seg)
    print("Predict shape:")
    print("pred.shape:", pred.shape)
    print("pred_seg.shape:", seg.shape)

    # Interpolation
    # seg = cv2.resize(seg, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    width = 1024
    height = 1024

    img_stack = seg[0]
    img_stack_sm = np.zeros((len(img_stack), width, height))

    for idx in range(len(img_stack)):
        img = img_stack[idx, :, :]
        img_sm = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        img_stack_sm[idx, :, :] = img_sm
        
    seg = np.expand_dims(img_stack_sm, axis=0)

    del ort_session
    del net_predict
    del out
    gc.collect()
    
    y_pred = pred.copy()
    y_calibrated = pred.copy()
    y_uncalibrated = pred.copy()

    for i, (c_name, thredshold) in enumerate(threshold_dict.items()):
        i_class = class_dict[c_name]
        y_calibrated[:,i_class] = calibrate_prob(y_calibrated, pred_val, c_name)/100
        # y_calibrated[:,i_class] = np.clip(y_calibrated[:,i_class]/temperature_dict[c_name], 0.00, 1.00) #  calibrating prob with weight from temperature scaling technique
        y_pred[:,i_class][y_pred[:,i_class] >= thredshold] = 1
        y_pred[:,i_class][y_pred[:,i_class] < thredshold] = 0
        
        
    all_pred_class = np.array(CATEGORIES)[y_pred[0] == 1]

    df_prob_class = pd.DataFrame(y_uncalibrated, columns=CATEGORIES) # To use risk score from raw value of model

    risk_dict = {'risk_score': 1- df_prob_class['No finding'].values[0]}
    
    all_pred_df = get_all_pred_df(CATEGORIES, y_calibrated, y_uncalibrated, threshold_dict)
    
    return pred, seg, all_pred_class, all_pred_df, risk_dict

def overlay_cam(img, cam, weight=0.5, img_max=255.):
    """
    Red is the most important region
    Args:
        img: numpy array (h, w) or (h, w, 3)
    """
    if len(img.shape) == 2:
        h, w = img.shape
        img = img.reshape(h, w, 1)
        img = np.repeat(img, 3, axis=2)

    # print('img:',img.shape)
    # print('seg:',cam.shape)
    h, w, c = img.shape

    img_max = img.max()
    # normalize the cam
    x = cam
    # x = x - x.min()
    # x = x / x.max()
    # resize the cam
    x = cv2.resize(x, (w, h))
    x = x - x.min()
    x = x / x.max()
    # Clip value to [0 1]
    x = np.clip(x, 0, 1)

    # coloring the cam
    x = cv2.applyColorMap(np.uint8(255 * (1 - x)), cv2.COLORMAP_JET)
    x = np.float32(x) / 255.

    # overlay
    x = img / img_max + weight * x
    x = x / x.max()
    return x

def fill_report_to_img(cam, Confidence, finding):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib import cm
    import matplotlib.pyplot as plt
    import textwrap
    Disclaimer = DISCLAIMER
    version_text = f"Developed by\n{AI_VERSION}"
    fig = plt.figure()
    ax1 = plt.subplot(111)

    text_wrapped = textwrap.fill(Disclaimer, 128)
    # cam = overlay_cam(img, seg[0, i_class])
    im = plt.imshow(cam, cmap='jet') ;plt.title(f'Probability of {finding}: {Confidence}', fontsize =8)
    cbaxes = inset_axes(ax1, width="40%", height="3%", loc=3) 
    cbar = plt.colorbar(cax=cbaxes, ticks=[0,1], orientation='horizontal')
    cbar.ax.set_xticklabels(['Low','High'], fontsize = 6)
    plt.text(1.2, -5.0, text_wrapped, ha="center", 
                fontsize=3, style='italic',
                bbox={"facecolor":"white", "alpha":0.5, "pad":3}, 
#                 wrap=True
                )
    plt.text(2.4, -1, version_text, ha='right', va='top',
                fontsize=4, 
#              style='oblique', 
                wrap=True)
        
    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)

    # plt.tight_layout()
    return fig

def get_multiclass_heatmap(img, all_pred_df, seg, class_dict, findings):
    import textwrap
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    if isinstance(findings, list):
        n_findings = len(findings)
        max_col_subplot = 3
        max_row_subplot = np.ceil(n_findings/max_col_subplot).astype(int)
        fig, axes = plt.subplots(max_row_subplot,max_col_subplot, 
                                 figsize=((max_col_subplot*2)+1, max_row_subplot*3),
#                                 dpi = 256
                                )
        fig.subplots_adjust(wspace=0, hspace=0)

    elif isinstance(findings, str):
        n_findings = 1
        fig, ax1 = plt.subplots(1,1)
        finding = findings
    else:
        raise TypeError('findings support only list or str')
    
    seg_calibrated = seg.copy()
    fontfactor = 2/max_col_subplot

    for i in range(max_row_subplot*max_col_subplot):
        row = np.floor(i/max_col_subplot).astype(int)
        col = int(i%max_col_subplot)
        ax = axes[row,col]
        if i < len(findings):
            finding = findings[i]
            i_class = class_dict[finding]
            Prob = all_pred_df[all_pred_df.Finding == finding]['Confidence'].values[0]
            threshold = all_pred_df[all_pred_df.Finding == finding]['Threshold'].values[0]
            print(f"{finding}: Threshold:{threshold:.3f}, Prob: {Prob:.3f}")
            # if Prob >= threshold:
            if Prob >= 0.5:
                Confidence = f'{Prob:.0%}'
            else:
                Confidence = 'Low'

            # Normalized heatmap with calibrated probability
            seg_calibrated_class = seg_calibrated[0, i_class]
            seg_calibrated_class = seg_calibrated_class - seg_calibrated_class.min()
            seg_calibrated_class = seg_calibrated_class / seg_calibrated_class.max()
            seg_calibrated_class = seg_calibrated_class * Prob

            cam = overlay_cam(img, seg_calibrated_class)
            im_cam = ax.imshow(cam, cmap='jet') ;

            # add the text to the top center of the image
            text = f'{finding}: {Confidence}'
            x_pos = img.shape[1] / 2
            y_pos = img.shape[0] * 0.05 #image.shape[0] / 2

            # automatically adjust the fontsize of text
            fontsize = int(fig.get_figwidth() * 2 * fontfactor)
            ax.text(x_pos, y_pos, text, ha='center', va='center', color='white', fontweight='bold', fontsize = fontsize)
            
            # Remove tick label
            im_cam.axes.get_xaxis().set_visible(False)
            im_cam.axes.get_yaxis().set_visible(False)

        else:
            # create an array of the same shape as the input image, filled with white pixels
            img = np.full_like(cam, 1)
            im = ax.imshow(img) ;

            # Remove tick label
            im.axes.get_xaxis().set_visible(False)
            im.axes.get_yaxis().set_visible(False)
        
            # Remove Frame
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)

            
    x_pos = img.shape[1]
    y_pos = img.shape[0]

    ax = axes[-1, 1]
    cbaxes = inset_axes(ax, width="40%", height="3%", loc='lower left') 
    # create the colorbar
    cbar = fig.colorbar(im_cam, cax=cbaxes, ticks=[0,1], orientation='horizontal')
    # set the tick labels and font size
    cbar.ax.set_xticklabels(['Low','High'], fontsize = fig.get_figwidth())

    # Add Version
    version_text = f"Developed by\n{AI_VERSION}"
    ax = axes[-1, -1]
    # ใต้ Subplot
    ax.text(x_pos*0.98, y_pos*0.98, version_text, ha = 'right', va = 'bottom',
                fontsize = fig.get_figwidth(), wrap = True)

    # Add Disclaimer
    Disclaimer = DISCLAIMER

    ax = axes[-1, 1]
    fontsize = fig.get_figwidth()
    text_wrapped = textwrap.fill(Disclaimer, 96)
    ax.text(x_pos / 2, y_pos*1.25, text_wrapped, ha='center', va='center', color='black', 
            fontweight='bold', fontsize = fontsize,
           style='italic', wrap=True,
            bbox={"facecolor":"white", "alpha":0.5, "pad":3}
           )
    
    return fig, axes

def hsv2rgb(h, s, v):
    import colorsys
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))

def plot_bbox_from_df(df_bbox, root_path = None, img_col_path = 'image_id', img_col_class_id = 'class_id', 
                      rad_id = None, class_name = None, mapping = None, 
                      n_show = 1, 
                      save_fig_name = None):
    
    import cv2
    import os
    import pydicom, numpy as np
    # import matplotlib.pyplot as plt
    

    mapping = mapping
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7
    thickness = 2
    

    
    df_path = df_bbox.reset_index(drop=True)
    # Select specific case
    if (rad_id != None) | (class_name != None):
        rad_id = '' if rad_id == None else rad_id
        class_name = '' if class_name == None else class_name
        df_path = df_bbox[(df_bbox.rad_id.str.contains(rad_id)) & 
                            (df_bbox.class_name.str.contains(class_name)) ]
        
    # Random select
    n_show = min(len(df_path), n_show)
    
    row = int(np.ceil(n_show/2))
    # plt.figure(figsize=(25,row*15))
    
#     try:
#         df_image_name = df_path.drop_duplicates([img_col_path]).sample(n_show).reset_index(drop=True)
#     except: # if sample less than n_show, bring all
#         df_image_name = df_path.drop_duplicates([img_col_path]).reset_index(drop=True)
    df_image_name = df_path.drop_duplicates([img_col_path]).sort_values(by='image_id').reset_index(drop=True)
    
    for idx in range(n_show):
        try:
            sample_img_path = df_image_name[img_col_path][idx]
            # print("sample_img_path:",idx, n_show, sample_img_path)
        except:
            break

        if root_path != None:
            # print('root_path:',root_path)
            try: 
                dicom = pydicom.dcmread(os.path.join(root_path,sample_img_path))
                inputImage = dicom.pixel_array
                print('read dcm')

                # depending on this value, X-ray may look inverted - fix that:
                if dicom.PhotometricInterpretation == "MONOCHROME1":
                    inputImage = np.amax(inputImage) - inputImage
            except:
                inputImage = PIL.Image.open(os.path.join(root_path,sample_img_path.replace('.dcm','.png')))
                print('read png')

            inputImage = np.stack([inputImage, inputImage, inputImage])
            inputImage = inputImage.astype('float32')
            inputImage = inputImage - inputImage.min()
            inputImage = inputImage / inputImage.max()
            inputImage = inputImage.transpose(1, 2, 0)
            inputImage = (inputImage*255).astype(int)
            # https://github.com/opencv/opencv/issues/14866
            inputImage = cv2.UMat(inputImage).get()
        else: 
            inputImage = np.zeros([3000,3000,3])
            inputImage = inputImage.astype(int)
            inputImage = cv2.UMat(inputImage).get()
            
        df_image = df_path[df_path.image_id == sample_img_path].reset_index(drop=True)
        
        all_class = []
#         print(df_image.index)
        for sub_idx in df_image.index:
            box = df_image.loc[sub_idx, 'x_min':'y_max'].values.astype(int)
            cls_id = df_image[img_col_class_id][sub_idx]
            rad_id = df_image.rad_id[sub_idx]

            cv2.rectangle(inputImage,(int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                          hsv2rgb(int(cls_id) / 14, 1, 1),thickness)
            inputImage = cv2.putText(inputImage,  mapping[int(cls_id)]+'_'+rad_id, (box[0], box[1]), font, fontScale,
                                       hsv2rgb(int(cls_id) / 14, 0.7, 1), thickness, cv2.LINE_AA)
            
            all_class.append(mapping[int(cls_id)])
        
        # print(row,2,idx+1)
        # plt.subplot(row,2,idx+1)
        # plt.imshow(inputImage, cmap='gray')
        # plt.title(f'{" | ".join(set(all_class))}', fontsize=14, fontweight='bold');
        # plt.text(15, -0.01, sample_img_path);
        # plt.title(sample_img_path, y=-0.01); # print image_id 
        # plt.tight_layout()

        title_class = f'{" | ".join(set(all_class))}'

    return inputImage, title_class

    # if save_fig_name != None:
    #     plt.savefig(save_fig_name+'.png')
        