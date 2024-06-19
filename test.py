import argparse
import json
import os
import pathlib

from huggingface_hub import snapshot_download

from LLAVA.llava.constants import IMAGE_TOKEN_INDEX
from local_config import PATH_TO_MIMIC_CXR, JAVA_HOME, JAVA_PATH, VIS_ROOT

# set java path
os.environ["JAVA_HOME"] = JAVA_HOME
os.environ["PATH"] = JAVA_PATH + os.environ["PATH"]
os.environ["WANDB_PROJECT"] = 'radiolog_llava'

from LLAVA.llava.conversation import SeparatorStyle, conv_vicuna_v1, conv_llava_med
from LLAVA.llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria, process_image_biovil

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch.backends import cudnn
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop
from skimage import io

from downstream_tasks.automated_correction import get_correction_prompts
from downstream_tasks.chexpert_classification_downstream import get_chexpert_prompts_all, get_chexpert_prompts_bin

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm

from data.create_data import MyReportProcessor, MIMICEvalCap

from chexbert.run_chexbert import run_chexbert_labeler
from LLAVA.llava.model.builder import load_pretrained_model

torch.multiprocessing.set_sharing_strategy('file_system')


class ExpandChannels:
    """
    Transforms an image with one channel to an image with three channels by copying
    pixel intensities of the image along the 1st dimension.
    """

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        :param data: Tensor of shape [1, H, W].
        :return: Tensor with channel copied three times, shape [3, H, W].
        """
        if data.shape[0] != 1:
            raise ValueError(f"Expected input of shape [1, H, W], found {data.shape}")
        return torch.repeat_interleave(data, 3, dim=0)


class MIMIC_Dataset(Dataset):
    def __init__(self, split, truncate=None, prompt_type="basic", vision_tower='openai/clip-vit-large-patch14-336',
                 base_model='liuhaotian/llava-v1.5-7b', view_class=False, do_impression=False):
        super().__init__()
        # load csv file
        self.split = pd.read_csv(f'{PATH_TO_MIMIC_CXR}/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv')
        self.reports = pd.read_csv('mimic-cxr/reports_processed/mimic_cxr_sectioned.csv')
        self.metadata = pd.read_csv(f'{PATH_TO_MIMIC_CXR}/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv')
        self.view_class = view_class
        self.do_impression = do_impression

        # drop reports where findings are nan
        self.reports = self.reports.dropna(subset=['findings'])
        if self.do_impression:
            self.reports = self.reports.dropna(subset=['impression'])
        self.chexpert_cols = ["No Finding", "Enlarged Cardiomediastinum",
                              "Cardiomegaly", "Lung Opacity",
                              "Lung Lesion", "Edema",
                              "Consolidation", "Pneumonia",
                              "Atelectasis", "Pneumothorax",
                              "Pleural Effusion", "Pleural Other",
                              "Fracture", "Support Devices"]

        self.img_ids = {img_id: i for i, img_id in enumerate(self.reports['dicom_id'])}
        self.chexpert = pd.read_csv(f'data/data_files/finding_chexbert_labels.csv')

        if split == 'validate':
            self.pred_chexpert_labels = json.load(open('findings_classifier/predictions/structured_preds_chexpert_log_weighting_val_macro.json', 'r'))
        elif split == 'test':
            self.pred_chexpert_labels = json.load(
                open('findings_classifier/predictions/structured_preds_chexpert_log_weighting_test_macro.json', 'r'))

        self.vis_root = VIS_ROOT

        self.prompt_type = prompt_type

        self.split_ids = set(self.split.loc[self.split['split'] == split]['dicom_id'])
        self.train_ids = set(self.split.loc[self.split['split'] == 'train']['dicom_id'])

        # get all dicom_ids where "split" is split
        self.annotation = self.reports.loc[self.reports['dicom_id'].isin(self.split_ids)]
        if truncate is not None:
            self.annotation = self.annotation[:truncate]

        self.annotation['findings'] = self.annotation['findings'].apply(lambda x: x.replace('\n', ''))

        # Extract patient_id from Img_Folder (3rd part) and study_id is the name of the notefile without the pre-pending 's'
        self.annotation['subject_id'] = self.annotation['Img_Folder'].apply(lambda x: int(x.split('/')[2].lstrip('p')))
        self.annotation['study_id'] = self.annotation['Note_file'].apply(lambda x: int(x.lstrip('s').rstrip('.txt')))

        # Merge chexpert labels with annotation dataframe
        self.annotation = pd.merge(self.annotation, self.chexpert, how='left', left_on=['dicom_id'],
                                   right_on=['dicom_id'])

        if self.view_class:
            # merge with metadata
            self.annotation = pd.merge(self.annotation, self.metadata, how='left', left_on=['dicom_id'], right_on=['dicom_id'])
            self.annotation = self.annotation.dropna(subset=['ViewPosition'])

        # read prompt from json
        prompts = json.loads(Path("vicuna_prompts.json").read_text(encoding="UTF-8"))
        self.text_processor = MyReportProcessor(
            prompt=prompts[prompt_type], max_words=1000,
            prompt_neg=prompts[prompt_type.replace("matching_examples", "neg_matching_examples")])

        self.vis_transforms_biovil = self.create_chest_xray_transform_for_inference(512, center_crop_size=448)
        self.vision_tower = vision_tower
        self.base_model = base_model

    def create_chest_xray_transform_for_inference(self, resize: int, center_crop_size: int) -> Compose:
        """
        Defines the image transformation pipeline for Chest-Xray datasets.

        :param resize: The size to resize the image to. Linear resampling is used.
                       Resizing is applied on the axis with smaller shape.
        :param center_crop_size: The size to center crop the image to. Square crop is applied.
        """

        transforms = [Resize(resize), CenterCrop(center_crop_size), ToTensor(), ExpandChannels()]
        return Compose(transforms)

    def create_structured_chexpert_findings(self, ann):
        pred_chexpert_labels = self.pred_chexpert_labels[str(ann['dicom_id'])]
        no_labels = len(pred_chexpert_labels) == 0
        counter = 0
        no_findings = "No Finding" in pred_chexpert_labels
        if no_findings:
            counter += 1
        supp_devices = "Support Devices" in pred_chexpert_labels
        if supp_devices:
            counter += 1
        # We check if there are any findings except no findings and support devices
        if len(pred_chexpert_labels) > counter and no_findings:
            pred_chexpert_labels.remove("No Finding")
            no_findings = False
        finding_string = ', '.join(pred_chexpert_labels).lower().strip()
        return no_labels, finding_string

    def load_image(self, image_file):
        image = Image.open(VIS_ROOT + "/" + image_file).convert('RGB')
        return image

    def remap_to_uint8(self, array: np.ndarray, percentiles=None) -> np.ndarray:
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

    def load_image_biovil(self, path) -> Image.Image:
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

        image = self.remap_to_uint8(image)
        return Image.fromarray(image).convert("L")

    def __getitem__(self, index):
        ann = self.annotation.iloc[index]
        # ann = train_dataset.annotation[train_dataset.annotation["dicom_id"] == '0f915690-0dc82379-a7ab8ade-d5aa9707-7eeb43f6'].iloc[0]

        caption = ann['findings'].strip()
        dicom_id = ann["dicom_id"]
        impression = ann['impression'].strip() if self.do_impression else ""

        no_labels, finding_string = self.create_structured_chexpert_findings(ann)
        # no_labels = False
        # finding_string = 'atelectasis, pleural effusion'

        if self.view_class:
            view_target = ann['ViewPosition'].lower().strip()
        else:
            view_target = ""

        if self.view_class:
            # change user prompt to "what is the view?"
            input_text = "<image> From what view was this image taken?"
        else:
            input_text = self.text_processor(finding_string, no_labels=no_labels)

        if 'LLaVAMed' in self.base_model:
            conv = conv_llava_med.copy()
        else:
            conv = conv_vicuna_v1.copy()

        conv.append_message(conv.roles[0], input_text)
        if self.do_impression:
            conv.append_message(conv.roles[1], caption)
            conv.append_message(conv.roles[0], "Write the impression section corresponding to this radiology report.")
            conv.append_message(conv.roles[1], None)
        else:
            conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if self.vision_tower == 'biovil':
            image = self.load_image_biovil(pathlib.Path(VIS_ROOT, os.path.join(ann["Img_Folder"], ann["Img_Filename"])))
            image_tensor = process_image_biovil([image], self.vis_transforms_biovil)
        else:
            image = self.load_image(ann["Img_Folder"] + "/" + ann["Img_Filename"])
            image_tensor = process_images([image], image_processor, model.config)

        return {
            "text_input": prompt,
            "text_target": caption,
            "chexpert_labels": ann[self.chexpert_cols].astype(float).values,
            "dicom": dicom_id,
            "img_path": ann["Img_Folder"] + "/" + ann["Img_Filename"],
            "image_tensor": image_tensor,
            "view_target": view_target,
            "impression": impression
        }

    def __len__(self):
        return len(self.annotation)


def compute_metrics(all_preds, evaluator):
    scores, _ = evaluator.evaluate(all_preds)
    b1, b2, b3, b4, meteor, rouge = scores["Bleu_1"], scores["Bleu_2"], scores["Bleu_3"], scores["Bleu_4"], scores["METEOR"], scores["ROUGE_L"]
    return b1, b2, b3, b4, meteor, rouge


def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def extract_report(pred):
    pred = pred.split("ASSISTANT:")[1]
    if 'report:' in pred:
        return pred.split("report:")[1]
    elif 'Report:' in pred:
        return pred.split("Report:")[1]
    elif 'REPORT:' in pred:
        return pred.split("REPORT:")[1]
    else:
        return pred


class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def stratified_sample(df, simulated_epochs=1):
    # We want to reduce the number of examples with no finding to 1/14th of the dataset. We achieve this easily by first seperating the dataset into 2 groups: no finding and finding.
    # either no finding, or nothing is considered a no finding
    no_findings_indices = df.annotation[((df.annotation['No Finding'] == 1) | ((df.annotation[df.chexpert_cols] == 1).sum(1) == 0) == 1)].index
    finding_indices = df.annotation.index.difference(no_findings_indices)
    no_findings_indices = no_findings_indices.tolist()
    finding_indices = finding_indices.tolist()

    # we are striving to lose as little no_finding data as possible. So instead of just reducing the number of no_finding examples, we will increase the number of finding examples. Just clone and extend dataset
    finding_indices = finding_indices * simulated_epochs
    # subsample the no finding examples to be 1/14th of the new dataset
    new_dataset_size = len(finding_indices) * 14 / 13
    new_no_finding_count = int(new_dataset_size / 14)
    # merge considering the new dataset size
    all_indices = finding_indices + no_findings_indices[:new_no_finding_count]
    return all_indices


def load_checkpoint_data(file_path):
    if Path(file_path).exists():
        with open(file_path, 'r') as file:
            return json.load(file)
    return {}


def update_checkpoint_data(file_path, model_name, checkpoint_id, wandb_run_id=None):
    data = load_checkpoint_data(file_path)
    if model_name not in data:
        data[model_name] = {"checkpoints": [], "wandb_run_id": wandb_run_id}
    if checkpoint_id not in data[model_name]["checkpoints"]:
        data[model_name]["checkpoints"].append(checkpoint_id)
    if wandb_run_id:
        data[model_name]["wandb_run_id"] = wandb_run_id
    with open(file_path, 'w') as file:
        json.dump(data, file)


if __name__ == '__main__':
    # read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="img_matching_examples_ig2_noexamples_IMG_findings",
                        help="prompt type")  # options=["basic", "advanced", "gen_examples", "matching_examples"]
    parser.add_argument("--model_base", type=str, default='liuhaotian/llava-v1.5-7b', help="base model name")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers for dataloader")
    parser.add_argument("--do_sample", action="store_true", help="", default=False)
    parser.add_argument("--temperature", type=float, default=1.0, help="")
    parser.add_argument("--num_beams", type=int, default=1, help="beam size for generation")
    parser.add_argument("--do_corr", action="store_true", help="", default=False)
    parser.add_argument("--do_cp_bin_qa", action="store_true", help="", default=False)
    parser.add_argument("--do_cp_all_qa", action="store_true", help="", default=False)
    parser.add_argument("--strat_eval", action="store_true", help="", default=False)
    parser.add_argument("--split", type=str, help="", default='validate')
    parser.add_argument("--vision_tower", type=str, default='openai/clip-vit-large-patch14-336')
    parser.add_argument("--do_view_class", action="store_true", help="", default=False)
    parser.add_argument("--do_impression", action="store_true", help="", default=False)

    args = parser.parse_args()
    prompt_type = args.prompt

    mode = "evaluate"

    if args.do_view_class:
        # classify in four options: (AP, PA, Lateral, and LL).
        setup_seeds(42)
        val_dataset = MIMIC_Dataset(split=args.split, truncate=None, prompt_type=prompt_type, vision_tower=args.vision_tower,
                                    base_model=args.model_base, view_class=True)

        batchsize = 12  # 12
        if args.strat_eval: #only for evaluation not for testing
            stratified_indices = stratified_sample(val_dataset, simulated_epochs=1)
            sampler = SubsetSampler(stratified_indices)
            data_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=args.num_workers, sampler=sampler)
        else:
            data_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=args.num_workers)

        # load LLaVA model
        model_path = snapshot_download(repo_id="ChantalPellegrini/RaDialog-interactive-radiology-report-generation", revision="main")
        model_path = pathlib.Path(model_path)
        model_name = "llava-v1.5-7b-task-lora_radialog_instruct_llava_biovil_unfrozen_2e-5_5epochs_v5_checkpoint-21000"
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base='liuhaotian/llava-v1.5-7b',
                                                                               model_name=model_name,
                                                                               load_8bit=False, load_4bit=False)
        model.config.tokenizer_padding_side = "left"

        conv = conv_vicuna_v1.copy()

        '''Report Generation'''
        exp_name = f"{model_name}_view_class"

        view_targets = []
        view_preds = []

        print("Dataloader len: ", len(data_loader))
        for _, batch in tqdm(enumerate(data_loader)):
            text_input = batch["text_input"]

            text_target = batch["text_target"]
            chexpert_labels = batch["chexpert_labels"]
            dicom_id = batch["dicom"]
            image_tensor = batch["image_tensor"]
            view_target = batch["view_target"]

            if type(image_tensor) is list:
                image_tensor = [image.to(model.device, dtype=torch.bfloat16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.bfloat16)

            image_tensor = image_tensor.squeeze(1)
            input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in text_input]

            # merge with left padding
            inverted_input_ids = [torch.flip(input_id, dims=[0]) for input_id in input_ids]
            input_ids = torch.nn.utils.rnn.pad_sequence(inverted_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
            # invert back
            input_ids = torch.flip(input_ids, dims=[1]).to(model.device)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,
                    use_cache=True,
                    max_new_tokens=300,
                    stopping_criteria=[stopping_criteria],
                    pad_token_id=tokenizer.pad_token_id
                )

            pred = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:].tolist(), skip_special_tokens=True)
            for pred in pred:
                if ' ap ' in pred.lower():
                    view_preds.append('ap')
                elif ' pa ' in pred.lower():
                    view_preds.append('pa')
                elif ' lateral ' in pred.lower():
                    view_preds.append('lateral')
                elif ' ll ' in pred.lower():
                    view_preds.append('ll')
                else:
                    view_preds.append('unknown')

            view_targets.extend(view_target)

        # calculate metrics (macro F1 and accuracy)
        accuracy = accuracy_score(view_targets, view_preds)
        f1 = f1_score(view_targets, view_preds, average='macro')
        print(f"Accuracy: {accuracy}, F1: {f1}")
        # confusion matrix
        print(pd.crosstab(pd.Series(view_targets), pd.Series(view_preds), rownames=['True'], colnames=['Predicted'], margins=True))
        # save results
        with open(f"vicuna_results/view_class_results_{exp_name}.txt", "w") as f:
            f.write(f"Accuracy: {accuracy}, F1: {f1}")
            f.write("\n")
            f.write(f"Confusion Matrix")
            f.write("\n")
            f.write(str(pd.crosstab(pd.Series(view_targets), pd.Series(view_preds), rownames=['True'], colnames=['Predicted'], margins=True)))

    elif args.do_impression:
        # Given the findings section of a report, generate its impression section (in our case given image and findings section)
        setup_seeds(42)
        val_dataset = MIMIC_Dataset(split=args.split, truncate=None, prompt_type=prompt_type, vision_tower=args.vision_tower,
                                    base_model=args.model_base, view_class=False, do_impression=True)
        evaluator = MIMICEvalCap(val_dataset.annotation, val_dataset.img_ids, do_impression=True)

        batchsize = 12  # 12
        if args.strat_eval:
            stratified_indices = stratified_sample(val_dataset, simulated_epochs=1)
            sampler = SubsetSampler(stratified_indices)
            data_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=args.num_workers, sampler=sampler)
        else:
            data_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=args.num_workers)

        # load LLaVA model
        model_path = snapshot_download(repo_id="ChantalPellegrini/RaDialog-interactive-radiology-report-generation", revision="main")
        model_path = pathlib.Path(model_path)
        model_name = "llava-v1.5-7b-task-lora_radialog_instruct_llava_biovil_unfrozen_2e-5_5epochs_v5_checkpoint-21000"
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base='liuhaotian/llava-v1.5-7b',
                                                                               model_name=model_name,
                                                                               load_8bit=False, load_4bit=False)
        model.config.tokenizer_padding_side = "left"

        conv = conv_vicuna_v1.copy()

        '''Report Generation'''
        exp_name = f"{model_name}_impression"

        text_targets = []
        text_inputs = []
        dicom_ids = []
        all_preds = []

        print("Dataloader len: ", len(data_loader))
        for _, batch in tqdm(enumerate(data_loader)):
            text_input = batch["text_input"]

            gt_report = batch["text_target"]
            chexpert_labels = batch["chexpert_labels"]
            dicom_id = batch["dicom"]
            image_tensor = batch["image_tensor"]
            view_target = batch["view_target"]
            impression_target = batch["impression"]

            if type(image_tensor) is list:
                image_tensor = [image.to(model.device, dtype=torch.bfloat16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.bfloat16)

            image_tensor = image_tensor.squeeze(1)
            input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in text_input]

            # merge with left padding
            inverted_input_ids = [torch.flip(input_id, dims=[0]) for input_id in input_ids]
            input_ids = torch.nn.utils.rnn.pad_sequence(inverted_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
            # invert back
            input_ids = torch.flip(input_ids, dims=[1]).to(model.device)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,
                    use_cache=True,
                    max_new_tokens=300,
                    stopping_criteria=[stopping_criteria],
                    pad_token_id=tokenizer.pad_token_id
                )

            pred = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:].tolist(), skip_special_tokens=True)

            text_targets.extend(impression_target)
            text_inputs.extend(text_input)
            dicom_ids.extend(dicom_id)

            all_preds.extend(pred)

        eval_preds = [{"image": None, "caption": pred, "image_id": val_dataset.img_ids[dicom]} for pred, dicom in zip(all_preds, dicom_ids)]
        bleu1_score, bleu2_score, bleu3_score, bleu4_score, meteor_score, rouge_score = compute_metrics(eval_preds, evaluator)
        # save results to file
        with open(f'vicuna_results/results_{exp_name}_impression.txt', 'w') as f:
            f.write(f"Prompt: {text_input}\n")
            f.write(f"Avg Bleu 1: {bleu1_score}\n")
            f.write(f"Avg Bleu 2: {bleu2_score}\n")
            f.write(f"Avg Bleu 3: {bleu3_score}\n")
            f.write(f"Avg Bleu 4: {bleu4_score}\n")
            f.write(f"Avg Meteor: {meteor_score}\n")
            f.write(f"Avg Rouge: {rouge_score}\n")


    else:
        if mode == "evaluate":
            # set all seeds to make code deterministic
            setup_seeds(42)
            val_dataset = MIMIC_Dataset(split=args.split, truncate=None, prompt_type=prompt_type, vision_tower=args.vision_tower,
                                        base_model=args.model_base)

            batchsize = 12  # 12
            if args.strat_eval:
                stratified_indices = stratified_sample(val_dataset, simulated_epochs=1)
                sampler = SubsetSampler(stratified_indices)
                data_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=args.num_workers, sampler=sampler)
            else:
                data_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=args.num_workers)

            # load LLaVA model
            model_path = snapshot_download(repo_id="ChantalPellegrini/RaDialog-interactive-radiology-report-generation", revision="main")
            model_path = pathlib.Path(model_path)
            model_name = "llava-v1.5-7b-task-lora_radialog_instruct_llava_biovil_unfrozen_2e-5_5epochs_v5_checkpoint-21000"
            tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base='liuhaotian/llava-v1.5-7b',
                                                                                   model_name=model_name,
                                                                                   load_8bit=False, load_4bit=False)
            model.config.tokenizer_padding_side = "left"

            evaluator = MIMICEvalCap(val_dataset.annotation, val_dataset.img_ids)
            conv = conv_vicuna_v1.copy()

            '''Report Generation'''
            exp_name = f"{model_name}"
            # exp_name = f"debug"
            if args.do_corr:
                exp_name += "_before_corr"
            if args.do_cp_bin_qa:
                exp_name += "_before_cp_bin_qa"
            if args.do_cp_all_qa:
                exp_name += "_before_cp_all_qa"

            text_targets = []
            text_inputs = []
            all_preds = []
            all_chexpert_labels = []
            dicom_ids = []
            eval_preds = []
            preds_history = []
            finding_strings = []
            all_study_ids = []

            print("Dataloader len: ", len(data_loader))
            for _, batch in tqdm(enumerate(data_loader)):
                text_input = batch["text_input"]

                text_target = batch["text_target"]
                chexpert_labels = batch["chexpert_labels"]
                dicom_id = batch["dicom"]
                image_tensor = batch["image_tensor"]

                if batchsize == 1:
                    text_input = batch["text_input"][0]
                    text_target = batch["text_target"][0]
                    chexpert_labels = batch["chexpert_labels"][0]
                    dicom_id = batch["dicom"][0]
                    image_tensor = batch["image_tensor"][0]

                if type(image_tensor) is list:
                    image_tensor = [image.to(model.device, dtype=torch.bfloat16) for image in image_tensor]
                else:
                    image_tensor = image_tensor.to(model.device, dtype=torch.bfloat16)

                if batchsize == 1:
                    input_ids = tokenizer_image_token(text_input, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
                    all_chexpert_labels.append(chexpert_labels.numpy())
                else:
                    image_tensor = image_tensor.squeeze(1)
                    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in text_input]

                    # merge with left padding
                    inverted_input_ids = [torch.flip(input_id, dims=[0]) for input_id in input_ids]
                    input_ids = torch.nn.utils.rnn.pad_sequence(inverted_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
                    # invert back
                    input_ids = torch.flip(input_ids, dims=[1]).to(model.device)
                    all_chexpert_labels.extend(chexpert_labels.numpy())

                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor,
                        do_sample=False,
                        use_cache=True,
                        max_new_tokens=300,
                        stopping_criteria=[stopping_criteria],
                        pad_token_id=tokenizer.pad_token_id
                    )
                if batchsize == 1:
                    pred = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip().replace("</s>", "")
                else:
                    pred = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:].tolist(), skip_special_tokens=True)

                if batchsize == 1:
                    text_targets.append(text_target)
                    text_inputs.append(text_input)
                    dicom_ids.append(dicom_id)

                    all_preds.append(pred)
                    preds_history.append(text_input + pred)
                else:
                    text_targets.extend(text_target)
                    text_inputs.extend(text_input)
                    dicom_ids.extend(dicom_id)

                    all_preds.extend(pred)
                    preds_history.extend([text_input[i] + pred[i] for i in range(len(pred))])

            pred_dir = Path("chexbert").absolute() / "outputs" / "predictions"

            # save predictions
            with open(pred_dir / "predictions_{}.csv".format(exp_name), "w") as f:
                for i in range(len(all_preds)):
                    f.write('"' + all_preds[i].replace('"', '') + '"\n')

            eval_preds = [{"image": None, "caption": pred, "image_id": val_dataset.img_ids[dicom]} for pred, dicom in zip(all_preds, dicom_ids)]
            bleu1_score, bleu2_score, bleu3_score, bleu4_score, meteor_score, rouge_score = compute_metrics(eval_preds, evaluator)

            # chexpert score
            # save results to txt file
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)

            # run chexpert labeler
            torch.cuda.empty_cache()

            run_chexbert_labeler(reports_path=pred_dir / "predictions_{}.csv".format(exp_name),
                                 output_path=pred_dir / "labels_{}.csv".format(exp_name))

            # read chexpert labels from file
            cp_pred = pd.read_csv(pred_dir / "labels_{}.csv".format(exp_name))
            pred_labels = np.array(cp_pred[val_dataset.chexpert_cols].values)
            all_chexpert_labels = np.array(all_chexpert_labels)

            # Map present (1) cases to 1 and absent (0, was NaN) and uncertain (-1) cases to 0
            all_chexpert_labels = np.nan_to_num(all_chexpert_labels, nan=0)
            pred_labels = np.nan_to_num(pred_labels, nan=0)
            all_chexpert_labels[all_chexpert_labels == -1] = 0
            pred_labels[pred_labels == -1] = 0

            # Calculate F1 score
            mean_f1 = f1_score(all_chexpert_labels, pred_labels, average="macro")
            mean_prec = precision_score(all_chexpert_labels, pred_labels, average="macro")
            mean_rec = recall_score(all_chexpert_labels, pred_labels, average="macro")
            sample_f1 = f1_score(all_chexpert_labels, pred_labels, average="samples")

            print("Macro F1 Score:", mean_f1)
            print("Sample F1 Score:", sample_f1)

            # Calculate Accuracy
            acc_scores = []
            for i in range(all_chexpert_labels.shape[1]):
                acc = accuracy_score(all_chexpert_labels[:, i], pred_labels[:, i])
                acc_scores.append(acc)

            mean_acc = np.mean(acc_scores)

            # save results to file
            with open(f'vicuna_results/results_{exp_name}.txt', 'w') as f:
                f.write(f"Prompt: {text_input}\n")
                f.write(f"Avg Bleu 1: {bleu1_score}\n")
                f.write(f"Avg Bleu 2: {bleu2_score}\n")
                f.write(f"Avg Bleu 3: {bleu3_score}\n")
                f.write(f"Avg Bleu 4: {bleu4_score}\n")
                f.write(f"Avg Meteor: {meteor_score}\n")
                f.write(f"Avg Rouge: {rouge_score}\n")
                f.write(f"Mean Chexpert F1: {mean_f1}\n")
                f.write(f"Mean Chexpert Precision: {mean_prec}\n")
                f.write(f"Mean Chexpert Recall: {mean_rec}\n")
                f.write(f"Sample Chexpert F1: {sample_f1}\n")
                f.write(f"Mean Chexpert Accuracy: {mean_acc}\n")

            '''
            Automatic Prompt Correction
            '''
            if args.do_corr:
                print("start correction")
                batchsize = 1
                data_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=args.num_workers)
                correction_prompts = get_correction_prompts(preds_history, val_dataset.chexpert_cols, pred_labels, all_chexpert_labels)
                # rerun vicuna with correction prompts
                text_targets_corr = []
                text_inputs_corr = []
                all_preds_corr = []
                all_chexpert_labels_corr = []
                dicom_ids_corr = []
                eval_preds_corr = []
                for idx, batch in tqdm(enumerate(data_loader)):
                    # use the corrected prompts
                    text_input = [correction_prompts[i] for i in range(batchsize * idx, min(batchsize * (idx + 1), len(correction_prompts)))][0]
                    text_target = batch["text_target"][0]
                    chexpert_labels = batch["chexpert_labels"][0]
                    all_chexpert_labels_corr.append(chexpert_labels.numpy())
                    dicom_id = batch["dicom"][0]
                    image_tensor = batch["image_tensor"][0]

                    if type(image_tensor) is list:
                        # image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
                        image_tensor = [image.to(model.device, dtype=torch.bfloat16) for image in image_tensor]
                    else:
                        # image_tensor = image_tensor.to(model.device, dtype=torch.float16)
                        image_tensor = image_tensor.to(model.device, dtype=torch.bfloat16)

                    input_ids = tokenizer_image_token(text_input, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
                    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids,
                            images=image_tensor,
                            do_sample=False,
                            use_cache=True,
                            max_new_tokens=300,
                            stopping_criteria=[stopping_criteria]
                        )

                    pred = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip().replace("</s>", "")

                    text_targets_corr.append(text_target)
                    text_inputs_corr.append(text_input)
                    dicom_ids_corr.append(dicom_id)
                    if "KEEP_OLD" not in text_input:
                        all_preds_corr.append(pred)
                    else:
                        all_preds_corr.append(text_input.split("</s>USER: KEEP_OLD")[0].split("ASSISTANT:")[-1].strip())

                # save predictions
                pred_dir = Path("chexbert").absolute() / "outputs" / "predictions"
                with open(pred_dir / "predictions_{}_after_corrections.csv".format(exp_name), "w") as f:
                    for i in range(len(all_preds_corr)):
                        f.write('"' + all_preds_corr[i].replace('"', '') + '"\n')

                eval_preds_corr = [{"image": None, "caption": pred, "image_id": val_dataset.img_ids[dicom]} for pred, dicom in
                                   zip(all_preds_corr, dicom_ids_corr)]
                bleu1_score, bleu2_score, bleu3_score, bleu4_score, meteor_score, rouge_score = compute_metrics(eval_preds_corr, evaluator)

                print("Bleu 1 Score:", bleu1_score)
                print("Bleu 2 Score:", bleu2_score)
                print("Bleu 3 Score:", bleu3_score)
                print("Bleu 4 Score:", bleu4_score)
                print("Meteor Score:", meteor_score)
                print("Rouge Score:", rouge_score)

                # chexpert score
                # save results to txt file
                if not os.path.exists(pred_dir):
                    os.makedirs(pred_dir)

                # run chexpert labeler
                # del lang_model
                torch.cuda.empty_cache()

                run_chexbert_labeler(reports_path=pred_dir / "predictions_{}_after_corrections.csv".format(exp_name),
                                     output_path=pred_dir / "labels_{}_after_corrections.csv".format(exp_name))

                # read chexpert labels from file
                cp_pred = pd.read_csv(pred_dir / "labels_{}_after_corrections.csv".format(exp_name))
                pred_labels = np.array(cp_pred[val_dataset.chexpert_cols].values)
                all_chexpert_labels = np.array(all_chexpert_labels_corr)

                # Map present (1) cases to 1 and absent (0, was NaN) and uncertain (-1) cases to 0
                all_chexpert_labels = np.nan_to_num(all_chexpert_labels, nan=0)
                pred_labels = np.nan_to_num(pred_labels, nan=0)
                all_chexpert_labels[all_chexpert_labels == -1] = 0
                pred_labels[pred_labels == -1] = 0

                # Calculate F1 score
                mean_f1 = f1_score(all_chexpert_labels, pred_labels, average="macro")
                mean_prec = precision_score(all_chexpert_labels, pred_labels, average="macro")
                mean_rec = recall_score(all_chexpert_labels, pred_labels, average="macro")
                sample_f1 = f1_score(all_chexpert_labels, pred_labels, average="samples")

                print("Macro F1 Score:", mean_f1)
                print("Sample F1 Score:", sample_f1)

                # Calculate Accuracy
                acc_scores = []
                for i in range(all_chexpert_labels.shape[1]):
                    acc = accuracy_score(all_chexpert_labels[:, i], pred_labels[:, i])
                    acc_scores.append(acc)

                mean_acc = np.mean(acc_scores)
                # print(acc_scores)
                print("Mean Accuracy:", mean_acc)

                # save results to file
                with open(f'vicuna_results/results_{exp_name}_after_corrections.txt', 'w') as f:
                    f.write(f"Prompt: {text_input}\n")
                    f.write(f"Avg Bleu 1: {bleu1_score}\n")
                    f.write(f"Avg Bleu 2: {bleu2_score}\n")
                    f.write(f"Avg Bleu 3: {bleu3_score}\n")
                    f.write(f"Avg Bleu 4: {bleu4_score}\n")
                    f.write(f"Avg Meteor: {meteor_score}\n")
                    f.write(f"Avg Rouge: {rouge_score}\n")
                    f.write(f"Mean Chexpert F1: {mean_f1}\n")
                    f.write(f"Mean Chexpert Precision: {mean_prec}\n")
                    f.write(f"Mean Chexpert Recall: {mean_rec}\n")
                    f.write(f"Sample Chexpert F1: {sample_f1}\n")
                    f.write(f"Mean Chexpert Accuracy: {mean_acc}\n")

            '''
            CheXpert Label Prediction
            '''

            if args.do_cp_all_qa:
                chexpert_prompts = get_chexpert_prompts_all(preds_history, val_dataset.chexpert_cols)
                batchsize = 1
                data_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=args.num_workers)

                chexpert_preds = []
                for idx, batch in tqdm(enumerate(data_loader)):
                    text_input = [chexpert_prompts[i] for i in range(batchsize * idx, min(batchsize * (idx + 1), len(chexpert_prompts)))][0]
                    text_target = batch["text_target"][0]
                    chexpert_labels = batch["chexpert_labels"][0]
                    dicom_id = batch["dicom"][0]
                    input_ids = tokenizer_image_token(text_input, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
                    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
                    image_tensor = batch["image_tensor"][0]

                    if type(image_tensor) is list:
                        # image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
                        image_tensor = [image.to(model.device, dtype=torch.bfloat16) for image in image_tensor]
                    else:
                        # image_tensor = image_tensor.to(model.device, dtype=torch.float16)
                        image_tensor = image_tensor.to(model.device, dtype=torch.bfloat16)

                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids,
                            images=image_tensor,
                            do_sample=False,
                            use_cache=True,
                            max_new_tokens=300,
                            stopping_criteria=[stopping_criteria]
                        )

                    pred = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip().replace("</s>", "")
                    preds = [pred]

                    # iterate through all chexpert labels and check if they are in finding preds
                    finding_preds_cleaned = []
                    for finding_pred in preds:
                        finding_pred_cleaned = []
                        for label in val_dataset.chexpert_cols:
                            if label.lower() in finding_pred:
                                finding_pred_cleaned.append(label.lower())
                        # convert to one-hot
                        finding_pred_cleaned = [1 if c.lower() in finding_pred_cleaned else 0 for c in val_dataset.chexpert_cols]
                        finding_preds_cleaned.append(finding_pred_cleaned)
                    chexpert_preds.extend(finding_preds_cleaned)

                # compare to ground truth
                chexpert_preds = np.array(chexpert_preds)
                chexpert_preds = np.nan_to_num(chexpert_preds, nan=0)
                all_chexpert_labels[all_chexpert_labels == -1] = 0

                # Calculate F1 score
                mean_f1 = f1_score(all_chexpert_labels, chexpert_preds, average="macro")
                mean_prec = precision_score(all_chexpert_labels, chexpert_preds, average="macro")
                mean_rec = recall_score(all_chexpert_labels, chexpert_preds, average="macro")
                try:
                    auc = roc_auc_score(all_chexpert_labels, chexpert_preds, average="macro")
                except ValueError:
                    auc = -1
                acc = accuracy_score(all_chexpert_labels.flatten(), chexpert_preds.flatten())

                print("Macro F1 Score:", mean_f1)
                print("Macro AUC Score:", auc)
                print("Macro Precision Score:", mean_prec)
                print("Macro Recall Score:", mean_rec)
                print("Accuracy Score:", acc)

                with open(f'vicuna_results/results_{exp_name}_after_cp_all_qa.txt', 'w') as f:
                    f.write(f"Prompt: {text_input}\n")
                    f.write(f"Mean Chexpert F1: {mean_f1}\n")
                    f.write(f"Mean Chexpert Precision: {mean_prec}\n")
                    f.write(f"Mean Chexpert Recall: {mean_rec}\n")
                    f.write(f"Mean Chexpert Accuracy: {acc}\n")
                    f.write(f"Mean Chexpert AUC: {auc}\n")

            if args.do_cp_bin_qa:
                chexpert_prompts = get_chexpert_prompts_bin(preds_history, val_dataset.chexpert_cols)
                batchsize = 1
                data_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=args.num_workers)

                chexpert_preds = []
                for idx, batch in tqdm(enumerate(data_loader)):
                    text_input = chexpert_prompts[idx]
                    chexpert_labels = batch["chexpert_labels"]
                    dicom_id = batch["dicom"]

                    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in text_input]

                    # merge with left padding
                    inverted_input_ids = [torch.flip(input_id, dims=[0]) for input_id in input_ids]
                    input_ids = torch.nn.utils.rnn.pad_sequence(inverted_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
                    # invert back
                    input_ids = torch.flip(input_ids, dims=[1]).to(model.device)

                    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
                    image_tensor = batch["image_tensor"][0]

                    if type(image_tensor) is list:
                        # image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
                        image_tensor = [image.to(model.device, dtype=torch.bfloat16) for image in image_tensor]
                    else:
                        # image_tensor = image_tensor.to(model.device, dtype=torch.float16)
                        image_tensor = image_tensor.to(model.device, dtype=torch.bfloat16)

                    image_tensor = torch.stack([image_tensor] * len(input_ids))

                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids,
                            images=image_tensor,
                            do_sample=False,
                            use_cache=True,
                            max_new_tokens=300,
                            stopping_criteria=[stopping_criteria]
                        )

                    preds = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:].tolist(), skip_special_tokens=True)

                    chexpert_preds.append([1 if "yes" in p.split("ASSISTANT:")[-1].lower() else 0 for idx, p in enumerate(preds)])

                relevant_cols = [c for c in val_dataset.chexpert_cols if c not in ["No Finding"]]
                relevant_cols_idx = [val_dataset.chexpert_cols.index(c) for c in relevant_cols]
                no_findings_idx = val_dataset.chexpert_cols.index("No Finding")
                any_findings = np.array(chexpert_preds)[:, relevant_cols_idx].sum(axis=1)
                any_findings[any_findings > 0] = 1
                # invert
                no_findings = 1 - any_findings
                # compare to ground truth
                chexpert_preds = np.array(chexpert_preds)
                chexpert_preds[:, no_findings_idx] = no_findings
                chexpert_preds = np.nan_to_num(chexpert_preds, nan=0)
                all_chexpert_labels[all_chexpert_labels == -1] = 0

                # Calculate F1 score
                mean_f1 = f1_score(all_chexpert_labels, chexpert_preds, average="macro")
                mean_prec = precision_score(all_chexpert_labels, chexpert_preds, average="macro")
                mean_rec = recall_score(all_chexpert_labels, chexpert_preds, average="macro")
                try:
                    auc = roc_auc_score(all_chexpert_labels, chexpert_preds, average="macro")
                except ValueError:
                    auc = -1
                acc = accuracy_score(all_chexpert_labels.flatten(), chexpert_preds.flatten())

                print("Macro F1 Score:", mean_f1)
                print("Macro AUC Score:", auc)
                print("Macro Precision Score:", mean_prec)
                print("Macro Recall Score:", mean_rec)
                print("Accuracy Score:", acc)

                # save results to file
                with open(f'vicuna_results/results_{exp_name}_after_cp_bin_qa.txt', 'w') as f:
                    f.write(f"Prompt: {text_input[0]}\n")
                    f.write(f"Mean Chexpert F1: {mean_f1}\n")
                    f.write(f"Mean Chexpert Precision: {mean_prec}\n")
                    f.write(f"Mean Chexpert Recall: {mean_rec}\n")
                    f.write(f"Mean Chexpert Accuracy: {acc}\n")
                    f.write(f"Mean Chexpert AUC: {auc}\n")
