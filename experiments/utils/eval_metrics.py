# loading the metrics
import evaluate
exact_match_metric = evaluate.load("exact_match")
sacrebleu = evaluate.load("sacrebleu")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")


def bleu_compute(pred, suffix):
    return sacrebleu.compute(predictions = pred, references = suffix)["score"]

def em_compute(pred, suffix):
    return exact_match_metric.compute(references=pred, predictions=suffix)['exact_match']

def meteor_compute(pred, suffix):
    return meteor.compute(predictions = pred, references = suffix)["meteor"]

def rouge_compute(pred, suffix):
    return rouge.compute(predictions = pred, references = suffix)["rougeL"]