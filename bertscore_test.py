import re

from bert_score import BERTScorer

if __name__ == '__main__':
    PRED_CSV_PATH = ""
    with open(PRED_CSV_PATH) as f:
        predictions = f.read().split('"\n"')
        predictions = [p.replace('"', '') for p in predictions]

    TARGETS_CSV_PATH = ""
    with open(TARGETS_CSV_PATH) as f:
        targets = f.read().split('"\n"')
        targets = [p.replace('"', '') for p in targets]

    assert len(predictions) == len(targets)

    test_reports = [re.sub(r' +', ' ', test) for test in targets]
    method_reports = [re.sub(r' +', ' ', report) for report in predictions]

    scorer = BERTScorer(
        model_type="distilroberta-base",
        batch_size=256,
        lang="en",
        rescale_with_baseline=True,
        idf=False,
        idf_sents=test_reports)
    _, _, f1 = scorer.score(method_reports, test_reports)

    print(f1.mean())
