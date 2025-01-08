# Basic library
import time
from glob import glob

# Application library
import streamlit as st

# Local library
from FastAPI_Predict_CXR_ONNX_V2_1 import *
import SessionState

import os 
# Get the directory of the current script or module
current_dir = os.path.dirname(os.path.abspath(__file__))

# https://blog.streamlit.io/introducing-new-layout-options-for-streamlit/
# https://docs.streamlit.io/en/stable/api.html?highlight=beta_set_page_config#streamlit.set_page_config
# Use the full page instead of a narrow central column

st.set_page_config(
    page_title="AI-Assisted Rapid Chest Radiographs Reading",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    )

state = SessionState.get()

STYLE = """
<style>
img {
    max-width: 100%;
}
</style>
"""

# Annotation from radiologist
# df_annot_bbox = pd.read_csv('save/csv/Annotated_bbox_chula_cxr_Sam_all.csv')
df_annot_bbox = pd.read_csv(os.path.join(current_dir, 'save/csv/Annotated_bbox_chula_cxr_Sam_all_new_ratio.csv'))
df_annot_bbox.loc[:,'Image Index'] = df_annot_bbox['image_id'].apply(lambda x: x.replace('.dcm', '.png')) 
df_group = df_annot_bbox.groupby(['image_id','rad_id']).agg({'class_name':lambda x: (x+', ').sum().strip(', ')})
df_group_res = df_group.reset_index()
df_group_res = df_group_res.groupby(['image_id']).agg({'class_name':lambda x: (x+', ').sum().strip(', ')})
df_group_res = df_group_res.reset_index().sort_values('image_id')
df_group_res.class_name = df_group_res.class_name.apply(lambda x: set(x.split(', ')))

df_annot = df_group_res.copy()
df_annot.loc[:,'class_name'] = df_annot['class_name'].apply(list)

# Finding from labeller
df_test_2_merge = pd.read_csv(os.path.join(current_dir, 'save/csv/Chula_CXR_test_2_for_labelme_merge.csv'))
df_test_2_merge.loc[:,'image_id'] = df_test_2_merge['Image Index'].apply(lambda x: x.replace('.png', '.dcm')) 
df_labeller = df_test_2_merge[df_test_2_merge['image_id'].isin(df_annot['image_id'])]
df_labeller = df_labeller.reset_index(drop=True).sort_values('image_id')

# Root path of Uploaded DICOM file
# root_path = 'save/img_temp'
# Delete all file in folder
# for f in os.listdir(root_path):
#     if not f.endswith(".dcm"):
#         continue
#     os.remove(os.path.join(root_path, f))
#     print("Remove file:", f)
# root_dicom = 'save/Sample_Dicom_Present_png'
# root_dicom = 'sample_cxr_with_finding/Sample Data/Osteoporisis'
root_dicom = os.path.join(current_dir, 'sampled_images')

def welcome():
    # https://www.pluralsight.com/guides/deploying-image-classification-on-the-web-with-streamlit-and-heroku
    # https://laptrinhx.com/image-processing-using-streamlit-4003254469/
    # col_img_1, col_img_2 = st.columns([1,2])
    # col_img_1.image('mmt_logo.jpeg',use_column_width='auto')
    # col_img_2.image('Invitrace LOGO-02.png',use_column_width='auto')
    st.image(os.path.join(current_dir, 'Invitrace_LOGO_02.png'),use_column_width='auto')
    st.title('AI-Assisted Rapid Chest Radiographs Reading')
    # st.header("Brain Tumor MRI Classification Example")
    st.subheader('An AI analysis application for abnormality detection on chest radiographs')
    # st.text("Upload a Chest Radiographs Image for Abnormality Detection")

# https://docs.streamlit.io/en/stable/api.html

class FileUpload(object):
 # https://www.youtube.com/watch?v=Uh_2F6ENjHs
    def __init__(self):
        self.fileTypes = ['jpg','png','jpeg','dcm','dicom']
 
    def run(self):
        """
        Upload File on Streamlit Code
        :return:
        """
        # st.info(__doc__)
        global state
        st.markdown(STYLE, unsafe_allow_html=True)

        upload_type = st.sidebar.selectbox(
            'Upload type',
            ('Select from local', 'Select from database')
            )
        if upload_type == 'Select from local':
            file = st.sidebar.file_uploader("Choose an image file", type=self.fileTypes)
            show_file = st.sidebar.empty()
            if not file:
                show_file.info("Please upload a file of type: " + ", ".join(self.fileTypes))
                return
            is_dcmformat = file.name.split(".")[-1].lower() in ("dcm", "dicom"); # print('is_dcmformat:', is_dcmformat)
            is_imgformat = file.name.split(".")[-1].lower() in ("jpg", "jpeg", "png"); # print('is_imgformat:', is_imgformat)
            self.image_name = file.name.split('/')[-1]
        elif upload_type == 'Select from database':
            result_image_path_list = sorted([dcm for dcm in os.listdir(root_dicom) if dcm.endswith('.png') and 'report' not in dcm])
            dcm_name = st.sidebar.selectbox("Choose an image file", result_image_path_list)
            state.dcm_name = dcm_name
            show_file = st.sidebar.empty()
            # print('Show path:', os.path.join(root_dicom,state.dcm_name))
            file = open(os.path.join(root_dicom,state.dcm_name), 'rb')
            is_dcmformat = False
            is_imgformat = True
            self.image_name = file.name.split('/')[-1].replace('.png','.dcm')
            st.write(state.dcm_name)



        # content = file.getvalue() #; print(type(content))
        # Check file format
        # is_dcmformat = file.name.split(".")[-1].lower() in ("dcm", "dicom"); # print('is_dcmformat:', is_dcmformat)
        # is_imgformat = file.name.split(".")[-1].lower() in ("jpg", "jpeg", "png"); # print('is_imgformat:', is_imgformat)
        # self.image_name = file.name.split('/')[-1]

        # Pre-processing
        if is_dcmformat:
            dicom, image = dicom2array(file)
            # path = BytesIO(file)
            # image = PIL.Image.open(file)
            # raw_image = np.array(image.copy())
            # dicom.save_as(f'save/img_temp/{self.image_name}')
        elif is_imgformat:
            # image = read_imagefile(content.read())
            image = PIL.Image.open(file)
            # applying greyscale method
            image = PIL.ImageOps.grayscale(image)
            
        else:
            return "Image must be dicom or jpg or png format!"
        
        raw_image = np.array(image.copy())
        self.image = raw_image
        

        # If upload new image, Reset all state parameter
        try:
            # print('Get into pre reset state')
            # print('state.file_2:', state.file_2)
            # print('file.name:', file.name)
            # print('state.file_2 != file.name:', state.file_2 != file.name)
            if state.file_2.replace('.png','.dcm') != file.name.split('/')[-1].replace('.png','.dcm'): # When upload new image, reset all state
                # print('Get into reset state')
                try: state = reset_button(state)
                except Exception as e: print(e)
                # print('State has been reset')
                state.file_2 = file.name.split('/')[-1].replace('.png','.dcm')
                print('Retrieve new image, reset all state parameter')
        except:
            pass
        # print('Image shape:', raw_image.shape)
        # Show Image
        if is_dcmformat:   
            show_file.image(raw_image, caption='Chest Radiographs Image')
        elif is_imgformat:
            try:
                show_file.image(file, caption='Chest Radiographs Image')
            except:
                show_file.image(raw_image, caption='Chest Radiographs Image')
        file.close()  
        return raw_image, self.image_name

    def predict_process(self):
        # Prediction Process
        image = preprocess(self.image)
        # print('In predict_process:', image.shape)
        pred, seg, all_pred_class, all_pred_df, risk_dict = predict(image, net_predict, threshold_dict, class_dict)
        # pred, seg = pred.numpy(), seg.numpy()
        
        
        # Show predict result
        # text_result = 'Predicted Finding: \n' + str('\n'.join(all_pred_prob_class))
        # st.text(all_pred_class)
        # st.text(text_result) # Same as print()
        # st.write('Predicted Finding: \n' + str('\n'.join(all_pred_prob_class))) # Same as display()
        return all_pred_class, all_pred_df, seg, risk_dict['risk_score']


def reset_button(state):
    state.description_text = st.empty()
    state.text_result = ''
    state.finding_contour = ''
    state.all_pred_class =[]
    state.finding_bbox = []
    state.t = 0
    state.text_report = False
    state.annot_pos_finding = None
    state.labbeler_pos_finding = None
    return state


welcome()
# islogin = login_page()
# if islogin:
#     if st.sidebar.button('Logout'):
#         state.key += 1
#         islogin = False


helper = FileUpload()
output = helper.run()


try:
    raw_image, file_name = output
except:
    raw_image, file_name = None, None

print('file_name:', file_name)

# try:
#     text_report = df_labeller[df_labeller.image_id == file_name].Finalized_report.values[0]
#     annot_pos_finding = pd.DataFrame(sorted(list(df_group_res[df_group_res.image_id == file_name].class_name.values[0])), columns=['Finding'])
#     labbeler_pos_finding = pd.DataFrame(sorted(np.array(CATEGORIES)[df_labeller[df_labeller.image_id == file_name][CATEGORIES].values[0] == 1]) , columns=['Finding'])
#     state.text_report = text_report
#     state.annot_pos_finding = annot_pos_finding
#     state.labbeler_pos_finding = labbeler_pos_finding
# except:
#     pass


# col1_1, col1_2 = st.columns(2)

# try:
#     if state.text_report:
#         isGroundtruth = col1_1.button("Show Ground Truth")
#         isReset = col1_2.button("Reset")
#     else:
#         isReset = col1_1.button("Reset")
# except:
#     isReset = col1_1.button("Reset")

# print('isGroundtruth:', isGroundtruth)
# print('isReset:', isReset)
# if isGroundtruth:
#     st.markdown('**_Finalized Report:_**')
#     st.write(state.text_report)
#     text_col1, text_col2 = st.beta_columns(2)
#     text_col1.markdown('**_Positive Finding (Annotated by Radiologist):_**')
#     text_col1.write(state.annot_pos_finding)
#     text_col2.markdown('**_Positive Finding (Extract from report):_**')
#     text_col2.write(state.labbeler_pos_finding)

# if isReset:
#     state = reset_button(state)


# col2_1, col2_2 = st.beta_columns(2)
# isPredict = col2_1.button("Predict")
# isContour = col2_2.button('Show Contour')

isPredict = st.button("Predict")

if isPredict:
    start = time.time()
    state.description_text = st.empty()
    state.description_text.text('Image Analyzing ...')
    state.all_pred_class, state.all_pred_df, state.pred_seg, state.risk_score = helper.predict_process()
    state.all_pred_df_sel_col = state.all_pred_df[['Finding','Confidence','isPositive']].sort_values('Confidence', ascending = False).reset_index(drop=True)
    state.all_pred_df_sel_col.index += 1 # start index at 1 for Pandas DataFrame
    state.focus_pred_class = state.all_pred_df[state.all_pred_df.Finding.isin(focusing_finding)].reset_index(drop=True)[['Finding','Confidence', 'isPositive']]
    state.focus_pred_class_pos = state.focus_pred_class[state.focus_pred_class.isPositive == 1][['Finding','Confidence']].sort_values('Confidence', ascending = False).reset_index(drop=True)
    state.focus_pred_class_pos.index += 1 # start index at 1 for Pandas DataFrame
    state.list_of_finding = state.all_pred_class
    state.t = time.time() - start
    state.file_2 = file_name
    # To clear memory in gpu
    del helper
    # torch.cuda.empty_cache()
    # description_text.text(f'Execution time = {state.t:.3f} seconds')
    # https://discuss.streamlit.io/t/checkbox-to-download-some-data-and-trigger-button/4160/2

# try:
#     if (isGroundtruth or state.t != 0 or state.gt) and state.text_report:
#         st.markdown('**_Finalized Report:_**')
#         st.write(state.text_report)
#         text_col1, text_col2 = st.columns(2)
#         text_col1.markdown('**_Positive Finding (Annotated by Radiologist):_**')
#         text_col1.write(state.annot_pos_finding)
#         text_col2.markdown('**_Positive Finding (Extract from report):_**')
#         text_col2.write(state.labbeler_pos_finding)
#         state.gt = isGroundtruth

# except:
#     pass




try:
    if state.t != 0:
        state.description_text.text('')
        # st.markdown('Streamlit is **_really_ cool**.')
        st.markdown('**_Prediction (AI Model):_**')
        st.text(f'Execution time = {state.t:.3f} seconds')
        st.text(f'Abnormality Score: {state.risk_score:>3.0%}')
        
        chk_all = st.checkbox("Show all list of finding")
        col4_1, col4_2 = st.columns(2)
        if chk_all: 
            state.list_of_finding = sorted(CATEGORIES)
            state.list_of_finding = state.all_pred_df_sel_col.Finding.values
            style_df = state.all_pred_df_sel_col.style.bar(subset=['Confidence'], align='mid', color=['#d65f5f', '#5fba7d'])\
                            .format({'Confidence': '{:.2%} '})
        else: 
            state.list_of_finding = state.focus_pred_class_pos.Finding.values
            style_df = state.focus_pred_class_pos.style.bar(subset=['Confidence'], align='mid', color=['#d65f5f', '#5fba7d'])\
                            .format({'Confidence': '{:.2%} '})
        col4_1.table(style_df)
        # col4_1.write(style_df)
except:
    pass


try:  

    # print('list_of_finding:', state.list_of_finding)
    state.finding_contour = col4_2.radio("Select finding to see contour", state.list_of_finding)
    selected_finding_text = st.empty()            
    selected_finding_text.text('Selected Finding: '+ state.finding_contour)
    seg = state.pred_seg
    start = time.time()
    finding = state.finding_contour
    i_class = class_dict[finding]
    image_plot = raw_image
    cam = overlay_cam(image_plot, seg[0, i_class])
    state.cam = cam
    t = time.time() - start
    state.t_contour = t
    
    #### Create report
    Prob = state.all_pred_df[state.all_pred_df.Finding == finding]['Confidence'].values[0]
    threshold = state.all_pred_df[state.all_pred_df.Finding == finding]['Threshold'].values[0]
    if Prob >= threshold:
        Confidence = f'{Prob:.0%}'
    else:
        Confidence = 'Low'
        
    fig_report = fill_report_to_img(cam, Confidence, finding)
    state.fig_report = fig_report
    
except:
    state.Bbox = raw_image
    state.title_class = 'No bounding box in database'

try:
    if state.text_report:
        chk_bbox = st.checkbox("Show list of bbox")
        if chk_bbox:
            state.finding_bbox = st.radio("Select finding to see bbox", sorted(np.append(np.array(['All']), state.annot_pos_finding.values.ravel())))
        else:
            state.finding_bbox = 'All'
    if  state.finding_bbox == 'All':
        df_temp_bbox = df_annot_bbox[(df_annot_bbox.image_id == file_name)]
    else:
        df_temp_bbox = df_annot_bbox[(df_annot_bbox.image_id == file_name) & (df_annot_bbox.class_name.isin([state.finding_bbox]))]

    # st.dataframe(df_temp_bbox)
    # st.write(file_name)
    # print('df_temp_bbox:', df_temp_bbox)
    # print('Pre title_class:', state.title_class)
    map_list = df_temp_bbox.class_name.unique()
    mapping = {i: map_list[i] for i in range(0, len(map_list))}
    cls_to_idx = {v: k for k, v in mapping.items()}
    df_temp_bbox.loc[:,'class_id'] = df_temp_bbox['class_name'].apply(lambda x: cls_to_idx[x])
    inputImage, title_class = plot_bbox_from_df(df_temp_bbox, root_path=root_dicom, mapping=mapping)
    state.Bbox = inputImage
    state.title_class = title_class
except:
    pass


col3_1, col3_2, col3_3 = st.columns(3)

try: 
    if state.t or state.text_report:
        pass
except:
    state.t = None
    state.text_report = None

if isPredict or state.t:
    isContour = col3_1.button("Show Contour")
    if state.text_report:
        isContour_Bbox = col3_3.button('Show Contour & Bounding Box')
else:
    isContour = False
    
    
try:
    if (state.text_report) and file_name:
        isBbox = col3_2.button('Show Bounding Box')
except:
    pass

try: 
    if isContour:
        col4_1, col4_2 = st.columns(2)
        # show_file_contour = st.empty()
        # cam = state.cam
        # finding = state.finding_contour
        # t = state.t_contour
        # selected_finding_text.text('Selected Finding: '+ state.finding_contour + f'\nExecution time = {t:.3f} seconds')
        # show_file_contour.image(cam, caption=f'Chest Radiographs Image with "{finding}" Contour', width=600)
        
        # Show image with report
        # show_fig_report = st.empty()
        fig_report = state.fig_report
        # show_fig_report.image(fig_report, caption=f'Chest Radiographs Image Report with "{finding}" Contour', width=600)
        col4_1.pyplot(fig_report)
        

    if isBbox:
        show_file_bbox = st.empty()
        inputImage, title_class = state.Bbox, state.title_class
        # print('title_class:', state.title_class, title_class)
        show_file_bbox.image(inputImage, caption=f'Bounding box: "{title_class}"', width=600)

    if isContour_Bbox:
        show_file = st.empty()
        finding = state.finding_contour
        seg = state.pred_seg
        i_class = class_dict[finding]
        inputImage, title_class = state.Bbox, state.title_class
        cam = overlay_cam(inputImage, seg[0, i_class])
        show_file.image(cam, caption=f'Chest Radiographs Image with "{finding}" Contour and Bounding box of {title_class}', width=600)

except:
    pass


# For show Full Report
col5_1, col5_2, col5_3 = st.columns(3)

if isPredict or state.t:
    isFullReport = col5_1.button("Show Full Report")  
else:
    isFullReport = False
    
try: 
    if isFullReport:
        col6_1, col6_2 = st.columns(2)
        
        image_plot = raw_image
        all_pred_df = state.all_pred_df
        seg = state.pred_seg
        findings = focusing_finding
        
        fig_full_report, axes = get_multiclass_heatmap(image_plot, all_pred_df, seg, class_dict, findings)
        
        col6_1.pyplot(fig_full_report)
except:
    pass
