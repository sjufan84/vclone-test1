""" Streamlit Vocal Cloning UI """
# Here is what we need this app to do:

# 1) Take in audio from the user via recording or uploading
# -- What format does the algorithm expect for the audio?
# -- Do we put restrictions / recommendations on what types of audio
# the user can upload / record?

# 2 -- Pass that audio to the algorithm
# -- Figure out what features we need to extract for 
# similarity score

# 3 -- Allow the user to play around with different features
# i.e. changing pitch, etc.

# 4 -- Allow the user to select either Joel or Jenny to clone
# their voice with.

# 5 -- Return the similarity score, numpy arrays, and cloned audio
    # async function to return the similarity scores, numpys (for charting)
    # and cloned audio   

import traceback
import requests
import tempfile
import numpy as np
import librosa
import io
from audio_recorder_streamlit import audio_recorder
import streamlit as st
import torch, os, sys, warnings, shutil
import datetime
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
now_dir = os.getcwd()
sys.path.append(now_dir)
tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/infer_pack" % (now_dir), ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "weights"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
import soundfile as sf
from fairseq import checkpoint_utils
import gradio as gr
import logging
from vc_infer_pipeline import VC
from config import Config
from i18n import I18nAuto
from utils import load_audio, CSVutil

i18n = I18nAuto()


# Create session state
# Define a function to reset the other pages to their default state
#def reset_pages():
#    st.session_state.cocktail_page = 'get_cocktail_info'
#    st.session_state.inventory_page = 'upload_inventory'
#    st.session_state.menu_page = 'upload_menus'
#reset_pages()

def init_chat_session_variables():
    """ Initialize the session state """
    session_vars = [
        'model', 'index_path', 'model_path', 'sid0', 'name_choices', "spk_item", "audio_files", 
        'pitch_adjustments', 'output_audio', 'input_audio'
    ]
    default_values = [
        None, None, None, '', ["Joel", "Jenny"], 0, [], 0, None, None
    ]

    for var, default_value in zip(session_vars, default_values):
        if var not in st.session_state:
            st.session_state[var] = default_value

init_chat_session_variables()

# Establish the titlea
st.title("Vocal Clone Demo v1")

global DoFormant, Quefrency, Timbre

try:
    DoFormant, Quefrency, Timbre = CSVutil('csvdb/formanting.csv', 'r', 'formanting')
    DoFormant = (
        lambda DoFormant: True if DoFormant.lower() == 'true' else (False if DoFormant.lower() == 'false' else DoFormant)
    )(DoFormant)
except (ValueError, TypeError, IndexError):
    DoFormant, Quefrency, Timbre = False, 1.0, 1.0
    CSVutil('csvdb/formanting.csv', 'w+', 'formanting', DoFormant, Quefrency, Timbre)

def download_models():
    # Download hubert base model if not present
    if not os.path.isfile('./hubert_base.pt'):
        response = requests.get('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt')

        if response.status_code == 200:
            with open('./hubert_base.pt', 'wb') as f:
                f.write(response.content)
            print("Downloaded hubert base model file successfully. File saved to ./hubert_base.pt.")
        else:
            raise Exception("Failed to download hubert base model file. Status code: " + str(response.status_code) + ".")
        
    # Download rmvpe model if not present
    if not os.path.isfile('./rmvpe.pt'):
        response = requests.get('https://drive.usercontent.google.com/download?id=1Hkn4kNuVFRCNQwyxQFRtmzmMBGpQxptI&export=download&authuser=0&confirm=t&uuid=0b3a40de-465b-4c65-8c41-135b0b45c3f7&at=APZUnTV3lA3LnyTbeuduura6Dmi2:1693724254058')

        if response.status_code == 200:
            with open('./rmvpe.pt', 'wb') as f:
                f.write(response.content)
            print("Downloaded rmvpe model file successfully. File saved to ./rmvpe.pt.")
        else:
            raise Exception("Failed to download rmvpe model file. Status code: " + str(response.status_code) + ".")

download_models()


def formant_apply(qfrency, tmbre):
    Quefrency = qfrency
    Timbre = tmbre
    DoFormant = True
    CSVutil('csvdb/formanting.csv', 'w+', 'formanting', DoFormant, qfrency, tmbre)
    
    return ({"value": Quefrency, "__type__": "update"}, {"value": Timbre, "__type__": "update"})

def get_fshift_presets():
    fshift_presets_list = []
    for dirpath, _, filenames in os.walk("./formantshiftcfg/"):
        for filename in filenames:
            if filename.endswith(".txt"):
                fshift_presets_list.append(os.path.join(dirpath,filename).replace('\\','/'))
                
    if len(fshift_presets_list) > 0:
        return fshift_presets_list
    else:
        return ''

def formant_enabled(cbox, qfrency, tmbre, frmntapply, formantpreset, formant_refresh_button):
    
    if (cbox):

        DoFormant = True
        CSVutil('csvdb/formanting.csv', 'w+', 'formanting', DoFormant, qfrency, tmbre)
        #print(f"is checked? - {cbox}\ngot {DoFormant}")
        
        return (
            {"value": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
        )
        
    else:
        
        DoFormant = False
        CSVutil('csvdb/formanting.csv', 'w+', 'formanting', DoFormant, qfrency, tmbre)
        
        #print(f"is checked? - {cbox}\ngot {DoFormant}")
        return (
            {"value": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
        )
        


def preset_apply(preset, qfer, tmbr):
    if str(preset) != '':
        with open(str(preset), 'r') as p:
            content = p.readlines()
            qfer, tmbr = content[0].split('\n')[0], content[1]
            
            formant_apply(qfer, tmbr)
    else:
        pass
    return ({"value": qfer, "__type__": "update"}, {"value": tmbr, "__type__": "update"})

def update_fshift_presets(preset, qfrency, tmbre):
    
    qfrency, tmbre = preset_apply(preset, qfrency, tmbre)
    
    if (str(preset) != ''):
        with open(str(preset), 'r') as p:
            content = p.readlines()
            qfrency, tmbre = content[0].split('\n')[0], content[1]
            
            formant_apply(qfrency, tmbre)
    else:
        pass
    return (
        {"choices": get_fshift_presets(), "__type__": "update"},
        {"value": qfrency, "__type__": "update"},
        {"value": tmbre, "__type__": "update"},
    )

def get_index_path(sid):
    if sid == "Jenny":
        index_path = './logs/added_IVF533_Flat_nprobe_1.index'
    elif sid == "Joel":
        index_path = "./logs/trained_IVF479_Flat_nprobe_1.index"
    return index_path

def get_model_path(sid):
    """ Retrieve the model path based on the sid0 selection """
    if sid == "Joel":
        model_path = "./weights/joel11320.pth"
    elif sid == "Jenny":
        model_path = "./weights/jenny3450.pth"
    else:
        model_path = ""
    st.session_state.model_path = model_path
    return model_path

def save_to_wav(record_button):
    """ """
    if record_button is None:
        pass
    else:
        path_to_file=record_button
        new_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+'.wav'
        new_path='./audios/'+new_name
        shutil.move(path_to_file,new_path)
        return new_path
    
def save_to_wav2(dropbox):
    file_path=dropbox.name
    shutil.move(file_path,'./audios')
    return os.path.join('./audios',os.path.basename(file_path))

def get_vc(sid):
    global n_spk, tgt_sr, net_g, vc, cpt, version
    if sid == "" or sid == []:
        global hubert_model
        if hubert_model is not None:  # 考虑到轮询, 需要加个判断看是否 sid 是由有模型切换到无模型的
            print("clean_empty_cache")
            del net_g, n_spk, vc, hubert_model, tgt_sr  # ,cpt
            hubert_model = net_g = n_spk = vc = hubert_model = tgt_sr = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            ###楼下不这么折腾清理不干净
            if_f0 = cpt.get("f0", 1)
            version = cpt.get("version", "v1")
            if version == "v1":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs256NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
            elif version == "v2":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs768NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
            del net_g, cpt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cpt = None
        return {"visible": False, "__type__": "update"}
    person = get_model_path(sid)
    print("loading %s" % person)
    cpt = torch.load(person, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]
    return {"visible": False, "max_value": n_spk, "__type__": "update"}

def vc_single(sid,
    input_audio_path:str=None,
    f0_up_key:int=0,
    f0_file=None,
    f0_method:str="harvest",
    file_index:str=None,
    #file_index2,
    # file_big_npy,
    index_rate:float=0.75,
    filter_radius:int=3,
    resample_sr:int=0,
    rms_mix_rate:float=0.21,
    protect:float=0.33,
    crepe_hop_length:int=120,
    ):  # spk_item, input_audio0, vc_transform0,f0_file,f0method0
    global tgt_sr, net_g, vc, hubert_model, version
    if input_audio_path is None:
        return "You need to upload an audio", None
    f0_up_key = int(f0_up_key)
    try:
        audio = load_audio(input_audio_path, 16000, DoFormant, Quefrency, Timbre)
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max
        times = [0, 0, 0]
        if hubert_model is None:
            load_hubert()
        if_f0 = cpt.get("f0", 1)
        file_index = (
            (
                file_index.strip(" ")
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip(" ")
                .replace("trained", "added")
            )
        ) 
        # file_big_npy = (
        #     file_big_npy.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        # )
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            sid,
            audio,
            input_audio_path,
            times,
            f0_up_key,
            f0_method,
            file_index,
            # file_big_npy,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            crepe_hop_length,
            f0_file=f0_file,
        )
        if resample_sr >= 16000 and tgt_sr != resample_sr:
            tgt_sr = resample_sr
        index_info = (
            "Using index:%s." % st.session_state.index_path
            if st.session_state.index_path is not None
            else "Index not used."
        )
        return "Success.\n %s\nTime:\n npy:%ss, f0:%ss, infer:%ss" % (
            index_info,
            times[0],
            times[1],
            times[2],
        ), (tgt_sr, audio_opt)
    except:
        info = traceback.format_exc()
        print(info)
        return info, (None, None)

config = Config()
# from trainset_preprocess_pipeline import PreProcess
logging.getLogger("numba").setLevel(logging.WARNING)

hubert_model = None

def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["./hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()


weight_root = "weights"
index_root = "logs"
name_choices = ["Joel", "Jenny"]

def main_ui():
    """ The main UI page structure """
    st.markdown("""
    Below we feature the ability to generate a clone of your voice based\
    on the models we trained using Jenny and Joel's voices.  There are several\
    different options to play around with, some of which are experimental,\
    but first start with adjusting pitch if necessary and then running the model\
    to get a baseline result.  Remember, we used *less than 10 minutes* of vocal\
    stems from Joel and Jenny to train their models.  One can imagine the quality\
    you can generate with longer training runs.
    """)
    st.markdown("""**You can either upload a clip of your\
    vocals or record a clip to be converted.  Currently we only have it set up to\
    copy one clip, but it is possible to convert by the batch, thus creating the\
    ability to combine shorter clips into one long one which could even be an entire\
    song.**
    """)
    st.text("")
    sid0 = st.selectbox(label="Choose which artist you would like to clone.",
    options=st.session_state.name_choices)
    st.session_state.index_path = get_index_path(sid0)
    st.session_state.model_path = get_model_path(sid0)
    get_vc(sid0)
              
    audio_container = st.container()
    audio = None
    with audio_container:
        #col1, col2 = st.columns([2,1], gap="medium")
        #with col1:
        audio_upload = st.file_uploader(label="Upload an audio clip here.  Make sure the clip is\
        only vocals and as clean as possible.", type=["wav", "mp3"])
        if audio_upload:
            # Create a temporary file to save the uploaded file
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(audio_upload.read())
            
            # Display the temporary file path (you can remove this later)
            st.write(f"Temporary file path: {tfile.name}")

            # Pass the temporary file path to your function
            st.session_state.input_audio = tfile.name   
            audio, sr = librosa.load(tfile.name)
            st.session_state.audio_files.append({tfile.name: audio})
        #with col2:
        #    st.text("")
            #st.markdown("""
            #<p style="text-align:center; margin-right=20px;"><b>Record Audio</b></p>
            #""", unsafe_allow_html=True)
            #col4, col5 = st.columns(2)
            #with col4:
            #    st.markdown(":green[Stopped] / \
            #                :red[Recording]")
            #with col5:
                #recorded_audio_bytes = audio_recorder(text="", icon_name="record-vinyl",
                #sample_rate=16000, neutral_color = "green", icon_size="3x")
            #st.text("")
            #st.text("")
            #st.text("")
            #if recorded_audio_bytes:
            #    st.markdown("""
            #                <p style="color:#EDC480; font-size: 23px; text-align: center;">
            #                Recorded audio clip:
            #                </p>
            #                """, unsafe_allow_html=True)
            #    st.audio(recorded_audio_bytes)
            #    recorded_audio = librosa.load(io.BytesIO(recorded_audio_bytes))
                # Using soundfile to read the audio file into a NumPy array
            #    st.session_state.audio_bytes_list.append(recorded_audio)
            # Put the audio recorder here    
            #record_button=gr.Audio(source="microphone", label="OR Record audio.", type="filepath")
            #record_audio = None
            #dropbox.upload(fn=save_to_wav2, inputs=[dropbox], outputs=[input_audio0])
            #dropbox.upload(fn=change_choices2, inputs=[], outputs=[input_audio0])
            #refresh_button2 = gr.Button("Refresh", type="primary", size='sm')
            #record_button.change(fn=save_to_wav, inputs=[record_button], outputs=[input_audio0])
            #record_button.change(fn=change_choices2, inputs=[], outputs=[input_audio0])
        
            #if st.session_state.audio_files != []:
            #    selected_input_audio = st.selectbox("Select audio to convert",
            #    options=[file[0] for file in st.session_state.audio_files])
            #    with open(selected_input_audio, 'rb') as f:
            #        audio_input = f.read().decode("utf-8")
        st.markdown("**Below you can play around with diferent settings.\
            It is a good idea to change the pitch and clone first to establish\
            a baseline and then adjust the others.  Note that some of the features\
            are experimental, so may not work as expected in certain cases.**")
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            # Index should be hard coded based on the selected model
            #st.markdown(":blue[Index file associated with the model:]")
            #st.markdown(f"**{st.session_state.index_path}**")
            #vc_output2 = st.audio(
            #    label="Output Audio (Click on the Three Dots in the Right Corner to Download)",
            #    type='filepath',
            #)
            # For now, the pitch extraction will be hardcoded to be "harvest"
            #f0method=st.radio(label="Optional: Change the Pitch Extraction Algorithm.\
            #        Extraction methods are sorted from 'worst quality' to\
            #        'best quality'. mangio-crepe may or may not be better\
            #        than rmvpe in cases where 'smoothness' is more important,\
            #        but rmvpe is the best overall.",
            #options=["pm", "dio", "crepe-tiny", "mangio-crepe-tiny", "crepe", "harvest", "mangio-crepe", "rmvpe"], # Fork Feature. Add Crepe-Tiny
            #index=5
        #)
            # Adjust the pitch if you want
            pitch_adjustments = st.slider("Pitch Adjustment -- If going from male to female\
                                        go lower, if female to male higher.", min_value=-12,
                                        max_value=12, step=1, value=0)
            st.session_state.pitch_adjustments = pitch_adjustments
            
            # Allow the user to select the resample rate
            resample_sr0 = st.slider(
                min_value=0,
                max_value=48000,
                label=i18n("后处理重采样至最终采样率，0为不进行重采样"),
                value=0,
                step=1000,
                )
            protect0 = st.slider(
                min_value=0.00,
                max_value=0.50,
                label=i18n("保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果"),
                value=0.33,
                step=0.01,
                )
            #crepe_hop_length = st.slider(
            #    min_value=1,
            #    max_value=512,
            #    step=1,
            #    label="Mangio-Crepe Hop Length. Higher numbers will\
            #        reduce the chance of extreme\
            #        pitch changes but lower numbers will increase accuracy. 64-192 is a good range to experiment with.",
            #    value=120,
            #    )
            #filter_radius0 = st.slider(
            #min_value=0,
            #max_value=7,
            #label=i18n(">=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音"),
            #value=3,
            #step=1,
            #)
        with col2:
            #rms_mix_rate0 = st.slider(
            #    min_value=0.00,
            #    max_value=1.00,
            #    label=i18n("输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络"),
            #    value=0.21,
            #    )
            # Allow the user to select the index_rate
            index_rate = st.slider("Index Rate -- Affects how closely\
            the model matches certain vocal characteristics", min_value=0.00, max_value=1.00, value=0.67, step=0.01)
            if st.session_state.input_audio is not None:
                st.markdown("**Original Audio Clip:**")
                st.audio(audio, sample_rate=sr)
            
            #formanting = st.checkbox(
            #    value=bool(DoFormant),
            #    label="[EXPERIMENTAL] Formant shift inference audio"            )
            #if formanting:
            #    qfrency = st.slider(
            #        value=8.0,
            #        label="Quefrency for formant shifting",
            #        min_value=0.0,
            #        max_value=16.0,
            #        step=0.1,
            #        )
            #    tmbre = st.slider(
            #        value=8.0,
            #        label="Timbre for formant shifting",
            #        min_value=0.0,
            #        max_value=16.0,
            #        step=0.1,
            #    )
            #formant_preset.change(fn=preset_apply, inputs=[formant_preset, qfrency, tmbre], outputs=[qfrency, tmbre])
            #frmntbut = st.button("Apply", type="primary", visible=bool(DoFormant))
            #if frmntbut:
            #    formanting.change(fn=formant_enabled,inputs=[formanting,qfrency,tmbre,frmntbut,formant_preset,formant_refresh_button],outputs=[formanting,qfrency,tmbre,frmntbut,formant_preset,formant_refresh_button])
            #frmntbut.onclick(fn=formant_apply,inputs=[qfrency, tmbre], outputs=[qfrency, tmbre])
            #formant_refresh_button.click(fn=update_fshift_presets,inputs=[formant_preset, qfrency, tmbre],outputs=[formant_preset, qfrency, tmbre])
            
            #vc_output1 = gr.Textbox("")
            #f0_file = gr.File(label=i18n("F0曲线文件, 可选, 一行一个音高, 代替默认F0及升降调"), visible=False)
            
            convert_button = st.button("Convert Audio", type='primary', use_container_width=True)
            if convert_button:
                if st.session_state.input_audio is None:
                    st.warning("You need to upload an audio file first.")
                with st.spinner("Converting audio..."):
                    st.session_state.output_audio = vc_single(
                        0,
                        f0_method="harvest",
                        input_audio_path=st.session_state.input_audio,
                        file_index=st.session_state.index_path,
                        # file_index2,
                        # file_big_npy1,
                        f0_up_key=pitch_adjustments,
                        index_rate=index_rate,
                        resample_sr=resample_sr0,
                        protect=protect0,
                    )
            if st.session_state.output_audio is not None:
                st.markdown("**Converted Audio Clip:**")
                st.audio(st.session_state.output_audio[1][1], sample_rate=st.session_state.output_audio[1][0])
main_ui()
