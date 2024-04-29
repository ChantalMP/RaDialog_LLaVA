
from enum import auto, Enum
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from nltk import word_tokenize
import numpy as np

class MyReportProcessor():
    def __init__(self, prompt="", max_words=50, prompt_neg=""):
        self.prompt = prompt
        self.max_words = max_words
        self.prompt_neg = prompt_neg

    def __call__(self, findings, no_labels=False):
        prompt = self.prompt

        if no_labels:
            findings = "no common findings"  # cannot write which findings as we don't no them
        prompt = prompt.format(findings=findings)

        return prompt



class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()

class MIMICEvalCap:
    def __init__(self, gts, img_id_map, do_impression=False):

        self.gts = gts
        if do_impression:
            self.gt_key = "impression"
        else:
            self.gt_key = "findings"

        # invert img_id_map
        self.dicom_to_id = img_id_map
        self.id_to_dicom = {v: k for k, v in img_id_map.items()}

        print('setting up scorers...')
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L")
        ]


    def preprocess(self, s):
        s = s.replace('\n', '')
        s = s.replace('<s>', '')
        s = s.replace('</s>', '')
        return s

    def evaluate(self, res):

        res = {self.id_to_dicom[elem["image_id"]]: elem["caption"] for elem in res}
        res_keys_set = set(res.keys())
        gts = {}
        gts_img_id = {}
        for _, elem in self.gts.iterrows():
            dicom_id = elem["dicom_id"]
            if dicom_id in res_keys_set:
                gts[dicom_id] = [elem[self.gt_key]]
                gts_img_id[self.dicom_to_id[dicom_id]] = [elem[self.gt_key]]

        # gts = {elem["dicom_id"]: [elem["findings"]] for _, elem in self.gts.iterrows() if elem["dicom_id"] in res.keys()}
        # gts_img_id = {self.dicom_to_id[elem["findings"]]: [elem["Note"]] for _, elem in self.gts.iterrows() if elem["dicom_id"] in res.keys()}
        assert res.keys() == gts.keys()
        # =================================================
        # Pre-process sentences
        # =================================================
        print('tokenization...')
        for dicom in res.keys():
            pred_text = ' '.join(word_tokenize(self.preprocess(res[dicom]))).lower()
            true_text = ' '.join(word_tokenize(self.preprocess(gts[dicom][0]))).lower()

            res[dicom] = [pred_text]
            gts[dicom] = [true_text]

        # =================================================
        # Compute scores
        # =================================================
        final_scores = {}
        for scorer, method in self.scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    final_scores[m] = sc
                    #final_scores["elem_wise_" + str(m)] = scs
                    print("%s: %0.3f" % (m, sc))
            else:
                print("%s: %0.3f" % (method, score))
                #final_scores["elem_wise_" + str(method)] = scores
                final_scores[method] = score

        final_scores['agg_metrics'] = np.mean(list({k: v for k, v in final_scores.items() if "elem_wise" not in k}.values()))

        return final_scores, gts_img_id