# eval_utils.py
import torch
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU


def calculate_bleu(predictions, references):
    bleu = BLEU()
    return bleu.corpus_score(predictions, [references]).score


def calculate_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        scores.append(scorer.score(ref, pred))
    return {
        "rouge1": sum([s["rouge1"].fmeasure for s in scores]) / len(scores),
        "rouge2": sum([s["rouge2"].fmeasure for s in scores]) / len(scores),
        "rougeL": sum([s["rougeL"].fmeasure for s in scores]) / len(scores),
    }


def generate_translations(model, dataloader, tokenizer, device, max_length=50):
    model.eval()
    predictions = []
    references = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"]

            generated_ids = model.generate(
                input_ids, max_length=max_length, num_beams=4, early_stopping=True
            )

            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

            predictions.extend(preds)
            references.extend(refs)

    return predictions, references
