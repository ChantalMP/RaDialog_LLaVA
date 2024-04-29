import argparse
import base64
import json
import os
import pathlib
import random
from copy import deepcopy
from enum import auto, Enum
from io import BytesIO

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from PIL import Image
from huggingface_hub import snapshot_download, hf_hub_download
from skimage import io
from torch.backends import cudnn
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop
from transformers import pipeline, AutoModel, AutoTokenizer

from LLAVA.llava.constants import IMAGE_TOKEN_INDEX
from LLAVA.llava.conversation import conv_vicuna_v1
from LLAVA.llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, process_image_biovil
from LLAVA.llava.model.builder import load_pretrained_model
from findings_classifier.chexpert_train import LitIGClassifier, ExpandChannels
from local_config import JAVA_HOME, JAVA_PATH

# Activate for deterministic demo, else comment
SEED = 16
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
cudnn.benchmark = False
cudnn.deterministic = True

# set java path
os.environ["JAVA_HOME"] = JAVA_HOME
os.environ["PATH"] = JAVA_PATH + os.environ["PATH"]
os.environ['GRADIO_TEMP_DIR'] = os.path.join(os.getcwd(), "gradio_tmp")

'''
The patient has only vascular congestion, no edema, can you correct it?
Reformulate the radiology report in easy language for a patient to understand.
Can you give a short explanation of what a CABG surgery is?

Write an impression section for this report.
From which view was the image taken?
Does the patient have any heart-related issues?
'''


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )

    args = parser.parse_args()

    return args


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()


gen_report = True
pred_chexpert_labels = json.load(open('findings_classifier/predictions/structured_preds_chexpert_log_weighting_test_macro.json', 'r'))

# Initialize the ASR model
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", device=0, torch_dtype=torch.float16)

global_stream = None


def transcribe_audio(new_chunk):
    global global_stream
    if new_chunk is None:
        return gr.update(value="")

    sr, y = new_chunk
    y = y.astype(np.float32)
    if y.ndim > 1 and y.shape[1] == 2:
        y = y.mean(axis=1)
    y /= np.max(np.abs(y))

    # Concatenate the new chunk to the global stream
    if global_stream is not None:
        global_stream = np.concatenate([global_stream, y])
    else:
        global_stream = y

    # Placeholder for your transcription logic
    transcribed_text = transcriber({"sampling_rate": sr, "raw": global_stream})["text"]

    return gr.update(value=transcribed_text, interactive=True)


def reset_global_stream():
    global global_stream
    global_stream = None
    print("Global stream reset")


# Initialize the text-to-speech pipeline with the Microsoft model
vctk_model = AutoModel.from_pretrained("kakao-enterprise/vits-vctk").to(
    "cuda")  # requires espeak. or espeak-ng with alias to espeak install with sudo dnf install espeak-ng then in bash alias espeak=espeak-ng
vctk_tokenizer = AutoTokenizer.from_pretrained("kakao-enterprise/vits-vctk")
text_to_speech_non_english_pipe = pipeline("text-to-speech", model="facebook/mms-tts-deu", device=0)
current_non_english_lang = 'de'
language_classification_pipe = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
language_detection_to_mms_mapping = {'de': 'deu', 'ar': 'ara', 'bg': 'bul', 'el': 'ell', 'en': 'eng', 'es': 'spa', 'fr': 'fra', 'hi': 'hin',
                                     'it': 'ita', 'ja': 'jpn', 'nl': 'nld', 'pl': 'pol',
                                     'pt': 'por', 'ru': 'rus', 'tr': 'tur', 'zh': 'cmn'}


def text_to_speech(text):
    global current_non_english_lang, text_to_speech_non_english_pipe
    # Generate speech using the specified text and speaker embedding
    text = text.replace('*', ' .')  # remove bullet points
    lang = language_classification_pipe(text)[0]['label']
    if lang != 'en' and lang in language_detection_to_mms_mapping:
        if lang != current_non_english_lang:
            text_to_speech_non_english_pipe = pipeline("text-to-speech", model=f"facebook/mms-tts-{language_detection_to_mms_mapping[lang]}")
            current_non_english_lang = lang
        speech = text_to_speech_non_english_pipe(text)
    else:
        inputs = vctk_tokenizer(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            vctk_model.speaking_rate = 0.9
            outputs = vctk_model(**inputs, speaker_id=6)[0].cpu().numpy()  # speaker 6, 28, 10 or 77 are good. For woman, 48, or 92 are also good.
        speech = {'audio': outputs, 'sampling_rate': 22050}
    # Convert the numpy array (audio data) to a byte stream
    audio_bytes = BytesIO()
    sf.write(audio_bytes, np.ravel(speech['audio']), samplerate=speech["sampling_rate"], format='wav')
    audio_bytes.seek(0)
    # Encode the audio to base64 for HTML audio element
    audio_base64 = base64.b64encode(audio_bytes.read()).decode("utf-8")
    audio_html = f'<audio src="data:audio/wav;base64,{audio_base64}" controls autoplay></audio>'
    return audio_html


def init_chexpert_predictor():
    # download from huggingface hub
    ckpt_path = hf_hub_download(repo_id="ChantalPellegrini/RaDialog-interactive-radiology-report-generation", revision="main", filename="findings_classifier/checkpoints/chexpert_train/ChexpertClassifier.ckpt")
    #ckpt_path = f"findings_classifier/checkpoints/chexpert_train/ChexpertClassifier-epoch=06-val_f1=0.36.ckpt"
    chexpert_cols = ["No Finding", "Enlarged Cardiomediastinum",
                     "Cardiomegaly", "Lung Opacity",
                     "Lung Lesion", "Edema",
                     "Consolidation", "Pneumonia",
                     "Atelectasis", "Pneumothorax",
                     "Pleural Effusion", "Pleural Other",
                     "Fracture", "Support Devices"]
    model = LitIGClassifier.load_from_checkpoint(ckpt_path, num_classes=14, class_names=chexpert_cols, strict=False)
    model.eval()
    model.cuda()
    model.half()
    cp_transforms = Compose([Resize(512), CenterCrop(488), ToTensor(), ExpandChannels()])

    return model, np.asarray(model.class_names), cp_transforms


def remap_to_uint8(array: np.ndarray, percentiles=None) -> np.ndarray:
    """Remap values in input so the output range is :math:`[0, 255]`.

    Percentiles can be used to specify the range of values to remap.
    This is useful to discard outliers in the input data.

    :param array: Input array.
    :param percentiles: Percentiles of the input values that will be mapped to ``0`` and ``255``.
        Passing ``None`` is equivalent to using percentiles ``(0, 100)`` (but faster).
    :returns: Array with ``0`` and ``255`` as minimum and maximum values.
    """
    array = array.astype(float)
    if percentiles is not None:
        len_percentiles = len(percentiles)
        if len_percentiles != 2:
            message = (
                'The value for percentiles should be a sequence of length 2,'
                f' but has length {len_percentiles}'
            )
            raise ValueError(message)
        a, b = percentiles
        if a >= b:
            raise ValueError(f'Percentiles must be in ascending order, but a sequence "{percentiles}" was passed')
        if a < 0 or b > 100:
            raise ValueError(f'Percentiles must be in the range [0, 100], but a sequence "{percentiles}" was passed')
        cutoff: np.ndarray = np.percentile(array, percentiles)
        array = np.clip(array, *cutoff)
    array -= array.min()
    array /= array.max()
    array *= 255
    return array.astype(np.uint8)


def load_image(path) -> Image.Image:
    """Load an image from disk.

    The image values are remapped to :math:`[0, 255]` and cast to 8-bit unsigned integers.

    :param path: Path to image.
    :returns: Image as ``Pillow`` ``Image``.
    """
    # Although ITK supports JPEG and PNG, we use Pillow for consistency with older trained models
    image = io.imread(path)

    image = remap_to_uint8(image)
    return Image.fromarray(image).convert("L")


# load LLaVA model
model_path = snapshot_download(repo_id="ChantalPellegrini/RaDialog-interactive-radiology-report-generation", revision="main")
model_path = pathlib.Path(model_path)
model_name = "llava-v1.5-7b-task-lora_radialog_instruct_llava_biovil_unfrozen_2e-5_5epochs_v5_checkpoint-21000"
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base='liuhaotian/llava-v1.5-7b',
                                                                       model_name=model_name,
                                                                       load_8bit=False, load_4bit=False)

model.config.tokenizer_model_max_length = None

cp_model, cp_class_names, cp_transforms = init_chexpert_predictor()

'''Conversation template for prompt'''
conv = conv_vicuna_v1.copy()

# Global variable to store the DICOM string
dicom = None
image_tensor = None


def remap_to_uint8(array: np.ndarray, percentiles=None) -> np.ndarray:
    """Remap values in input so the output range is :math:`[0, 255]`.

    Percentiles can be used to specify the range of values to remap.
    This is useful to discard outliers in the input data.

    :param array: Input array.
    :param percentiles: Percentiles of the input values that will be mapped to ``0`` and ``255``.
        Passing ``None`` is equivalent to using percentiles ``(0, 100)`` (but faster).
    :returns: Array with ``0`` and ``255`` as minimum and maximum values.
    """
    array = array.astype(float)
    if percentiles is not None:
        len_percentiles = len(percentiles)
        if len_percentiles != 2:
            message = (
                'The value for percentiles should be a sequence of length 2,'
                f' but has length {len_percentiles}'
            )
            raise ValueError(message)
        a, b = percentiles
        if a >= b:
            raise ValueError(f'Percentiles must be in ascending order, but a sequence "{percentiles}" was passed')
        if a < 0 or b > 100:
            raise ValueError(f'Percentiles must be in the range [0, 100], but a sequence "{percentiles}" was passed')
        cutoff: np.ndarray = np.percentile(array, percentiles)
        array = np.clip(array, *cutoff)
    array -= array.min()
    array /= array.max()
    array *= 255
    return array.astype(np.uint8)


def create_chest_xray_transform_for_inference(resize: int, center_crop_size: int) -> Compose:
    """
    Defines the image transformation pipeline for Chest-Xray datasets.

    :param resize: The size to resize the image to. Linear resampling is used.
                   Resizing is applied on the axis with smaller shape.
    :param center_crop_size: The size to center crop the image to. Square crop is applied.
    """

    transforms = [Resize(resize), CenterCrop(center_crop_size), ToTensor(), ExpandChannels()]
    return Compose(transforms)


vis_transforms_biovil = create_chest_xray_transform_for_inference(512, center_crop_size=448)


def load_image_biovil(path) -> Image.Image:
    """Load an image from disk.

    The image values are remapped to :math:`[0, 255]` and cast to 8-bit unsigned integers.

    :param path: Path to image.
    :returns: Image as ``Pillow`` ``Image``.
    """
    # Although ITK supports JPEG and PNG, we use Pillow for consistency with older trained models
    if path.suffix in [".jpg", ".jpeg", ".png"]:
        image = io.imread(path)
    else:
        raise ValueError(f"Image type not supported, filename was: {path}")

    image = remap_to_uint8(image)
    return Image.fromarray(image).convert("L")


def get_response(input_text, dicom):
    global image_tensor
    if input_text.endswith(".png") or input_text.endswith(".jpg") or input_text.endswith(".jpeg"):
        cp_image = load_image(input_text)
        cp_image = cp_transforms(cp_image)

        # if self.vision_tower == 'biovil':
        image = load_image_biovil(pathlib.Path(input_text))
        image_tensor = process_image_biovil([image], vis_transforms_biovil)
        # else:
        #     vic_image = Image.open(input_text[-1]).convert('RGB')
        #     image_tensor = process_images([vic_image], image_processor, model.config)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.bfloat16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.bfloat16)

        dicom = input_text.split('/')[-1].split('.')[0]
        if dicom in pred_chexpert_labels:
            findings = ', '.join(pred_chexpert_labels[dicom]).lower().strip()
        else:
            logits = cp_model(cp_image[None].half().cuda())
            preds_probs = torch.sigmoid(logits)
            preds = preds_probs > 0.5
            pred = preds[0].cpu().numpy()
            findings = cp_class_names[pred].tolist()
            findings = ', '.join(findings).lower().strip()

        if gen_report:
            input_text = (
                f"<image>. Predicted Findings: {findings}. You are to act as a radiologist and write the finding section of a chest x-ray radiology report for this X-ray image and the given predicted findings. "
                "Write in the style of a radiologist, write one fluent text without enumeration, be concise and don't provide explanations or reasons.")

        # save image embedding with torch
        else:
            print(findings)
            input_text = (
                f"<image>. Does the patient have cardiomegaly?")

    else:  # free chat
        input_text = input_text
        findings = None

    '''Generate prompt given input prompt'''
    conv.append_message(conv.roles[0], input_text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    '''Call vicuna model to generate response'''
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            use_cache=True,
            max_new_tokens=600,
            stopping_criteria=[stopping_criteria]
        )

    new_pred = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip().replace("</s>", "")

    # remove last message in conv
    conv.messages.pop()
    conv.append_message(conv.roles[1], new_pred)
    return new_pred, findings


# Function to update the global DICOM string
def set_dicom(value):
    global dicom
    dicom = value


def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.update(value="", interactive=False)


def add_file(filepath):
    global file_path
    print("Adding image file to chat history")
    conv.messages = []
    file_path = filepath
    print("Chat history cleared")
    return []


# Function to clear the chat history
def clear_history():
    global global_stream
    conv.messages = []
    global_stream = None
    update_msg1, update_msg2, update_msg3 = update_proposed_texts("", saved_findings, False)
    print("Chat history and image cleared")
    return [], None, update_msg1, update_msg2, update_msg3  # Return empty history to the Chatbot and None to ImageDisplay


def bot(history):
    global file_path, saved_findings
    # You can now access the global `dicom` variable here if needed
    if file_path is not None:
        response, findings = get_response(deepcopy(file_path), None)
        saved_findings = findings
        file_path = None
    else:
        try:
            response, findings = get_response(history[-1][0], None)
        except Exception as e:
            raise Exception(f"Make sure to first select an image before clicking the upload button. {e}")
    print(response)
    complete_audio_html = text_to_speech(response)

    # show report generation prompt if first message after image
    if len(history) == 0:
        input_text = f"You are to act as a radiologist and write the finding section of a chest x-ray radiology report for this X-ray image and the given predicted findings. Write in the style of a radiologist, write one fluent text without enumeration, be concise and don't provide explanations or reasons."
        if findings is not None:
            input_text = f"(img_tokens) Predicted Findings: {findings}. {input_text}"
        history.append([input_text, None])

    # test if history + response is more than 4096 tokens
    # if so, remove last message
    conv_text = ' '.join([conv.system] + [msg[0] for msg in history] + [msg[1] for msg in history if msg[1] is not None] + [response])
    if len(tokenizer.encode(conv_text)) > 4000:  # 4096 tokens is the max input length for LLava
        print("History too long")
        # display message that history is too long
        response = "The conversation history is too long. Please clear the history and start a new conversation."
        set_visible_to = False
    else:
        set_visible_to = True

    history[-1][1] = response
    assert response is not None
    update_msg1, update_msg2, update_msg3 = update_proposed_texts(history, saved_findings, set_visible_to)
    return history, update_msg1, update_msg2, update_msg3, complete_audio_html


custom_css = """
#custom_button {
    padding: 5px 10px; /* Adjust padding as needed to reduce the height */
    margin: 0; /* Remove any default margins */
    line-height: initial; /* Reset line height */
}
"""


# Function to fill the textbox with a predefined/proposed text
def fill_textbox_with_proposed(proposed_text):
    # update button text
    return proposed_text  # Correctly return just the string to update the textbox


# Function to update the proposed text buttons
possible_questions_after_report = [
    "The patient does not have <INPUT FINDING HERE>, please correct the radiology report.",
    "Can you write an impression section for this report?"]

all_findings = ["cardiomegaly", "a lung opacity", "a lung lesion", "an edema", "pneumonia", "atelectasis", "a pneumothorax", "pleural Effusion"]


def update_proposed_texts(history, saved_findings, set_visible_to=True):
    possible_questions_always = [
        "List all findings in the image.",
        f"Does the patient have {random.choice(all_findings)}?",
        f"Does the patient have any {random.choice(['heart', 'lung'])}-related issues?",
        "Can you translate the radiology report to German?",
        "Provide a summary of the radiology report in bullet points.",
        "Reformulate the radiology report in easy language for a patient to understand.",
        "From which view was this image taken?"
    ]

    predicted_findings = saved_findings.split(',') if saved_findings is not None else []
    # remove "no finding, "no common finding"
    predicted_findings = [f for f in predicted_findings if "no" not in f and "support" not in f]
    if len(predicted_findings) > 0 and '' not in predicted_findings:
        possible_questions_always.append(f"Can you explain what {random.choice(predicted_findings)} is?")

    if len(history) == 1:  # only report generation done
        question_proposals = random.sample(possible_questions_after_report + possible_questions_always, k=3)
    else:
        question_proposals = random.sample(possible_questions_always, k=3)
    return gr.Button(value=question_proposals[0], visible=set_visible_to), gr.Button(value=question_proposals[1], visible=set_visible_to), gr.Button(
        value=question_proposals[2],
        visible=set_visible_to)


if __name__ == '__main__':
    with gr.Blocks(theme=gr.themes.Base(primary_hue="emerald")) as demo:
        gr.Markdown("# RaDialog")  # Heading as Markdown
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    image_display = gr.Image(type='filepath', height=500, sources=['upload'])
                with gr.Row():
                    # create upload button
                    upload_button = gr.Button("Upload image", variant="primary", size="lg")
                with gr.Row():
                    # TODO add example images from MIMIC-CXR in demo_examples folder or add own examples and adapt paths
                    example_images = ["demo_examples/9b3b2ac9-c7621799-9c520077-028dc771-d93cf2d7.jpg",
                                      "demo_examples/f59791dd-2e8e1e7a-607b2f6e-18b713c7-aed09023.jpg",
                                      "demo_examples/88dd4b9d-f5dc2b18-5e9e6141-943b90b2-39b71300.jpg",
                                      "demo_examples/bd3dc01c-c67b8f05-580c3880-de7352aa-4118828e.jpg",
                                      "demo_examples/afa46108-e06269ce-05deb812-e12dad4d-ef863113.jpg"]
                    # The Examples component
                    examples = gr.Examples(examples=example_images, inputs=image_display)

            with gr.Column():
                chatbot = gr.Chatbot(
                    [],
                    elem_id="chatbot",
                    height=500
                )
                html_audio_player = gr.HTML(visible=False)  # Initially hidden
                with gr.Row():
                    txt = gr.Textbox(
                        show_label=False,
                        placeholder="Type or record text and press enter.",
                        container=False,
                        scale=3
                    )
                    audio_input = gr.Audio(sources=["microphone"], streaming=True, show_label=False, scale=1,
                                           waveform_options={"show_recording_waveform": False, "show_controls": False})

                # Proposed text buttons
                proposed_text_button1 = gr.Button("Proposed Question", elem_id="proposed_text_button1", size="sm", variant="secondary", visible=False)
                proposed_text_button2 = gr.Button("Proposed Question", elem_id="proposed_text_button2", size="sm", variant="secondary", visible=False)
                proposed_text_button3 = gr.Button("Proposed Question", elem_id="proposed_text_button3", size="sm", variant="secondary", visible=False)

                clear_btn = gr.Button("Clear History", variant="primary", size="lg")

            clear_btn.click(clear_history, None, [chatbot, image_display, proposed_text_button1, proposed_text_button2, proposed_text_button3])

            txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
                bot, chatbot, [chatbot, proposed_text_button1, proposed_text_button2, proposed_text_button3, html_audio_player]
            )
            txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)
            img_msg = image_display.upload(add_file, image_display, [chatbot], queue=False)
            up_msg = upload_button.click(add_file, image_display, [chatbot], queue=False).then(
                bot, chatbot, [chatbot, proposed_text_button1, proposed_text_button2, proposed_text_button3, html_audio_player]
            )

            # Link the buttons to the function to update the textbox
            proposed_text_button1.click(fill_textbox_with_proposed, [proposed_text_button1], [txt])
            proposed_text_button2.click(fill_textbox_with_proposed, [proposed_text_button2], [txt])
            proposed_text_button3.click(fill_textbox_with_proposed, [proposed_text_button3], [txt])

            audio_msg = audio_input.change(transcribe_audio, [audio_input], [txt], queue=False)
            audio_input.start_recording(reset_global_stream, queue=False)

    demo.dependencies[11]["show_progress"] = "hidden"  # transcribe_audio

    demo.queue()
    demo.launch()
