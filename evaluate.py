
import os
import argparse
import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration

from main import BreakDataset
from break_evaluator.scripts.evaluate_predictions import evaluate, format_qdmr


def process_args(args):
    # load data
    try:
        metadata = pd.read_csv(args.dataset_file) # question_id, decomposition, question_text
        ids = metadata["question_id"].to_list()
        questions = metadata["question_text"].to_list()
        golds = [format_qdmr(decomp) for decomp in metadata["decomposition"].to_list()]
    except Exception as ex:
        raise ValueError(f"Could not load dataset file {args.dataset_file}", ex)

    # load predictions
    try:
        preds_file = pd.read_csv(args.preds_file)
        predictions = [format_qdmr(pred) for pred in preds_file['decomposition'].to_list()]
    except Exception as ex:
        raise ValueError(f"Could not load predictions file {args.preds_file}", ex)

    assert len(golds) == len(predictions), "mismatch number of gold questions and predictions"

    # if args.random_n and len(golds) > args.random_n:
    #     indices = random.sample(range(len(ids)), args.random_n)
    #     ids = [ids[i] for i in indices]
    #     questions = [questions[i] for i in indices]
    #     golds = [golds[i] for i in indices]
    #     predictions = [predictions[i] for i in indices]

    if not args.no_cache:
        norm_rules.load_cache(args.dataset_file.replace(".csv", "__cache"))

    res = evaluate(ids=ids,
                   questions=questions,
                   golds=golds,
                   decompositions=predictions,
                   metadata=metadata,
                   output_path_base=args.output_file_base,
                   metrics=args.metrics)

    if not args.no_cache:
        norm_rules.save_cache(args.dataset_file.replace(".csv", "__cache"))

    return res


def build_dataset_file(dataset, save_dir, data_name): # question_id, decomposition, question_text
    csv_path = os.path.join(save_dir, data_name + ".csv")
    dataset.data.to_csv(csv_path, index=False) # if using this then need also to preprocessed, I skiped this as we can do this one time so I did it manually
    return csv_path


def build_predictions_file(model, tokenizer, dataset, save_dir, data_name): # decomposition
    csv_path = os.path.join(save_dir, data_name + ".csv")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    def postprocess(qdmr_decomposition):
        qdmr_decomposition = qdmr_decomposition.replace('@@', ';')
        qdmr_decomposition = qdmr_decomposition.replace('##', '#')
        return qdmr_decomposition 

    def predict_qdmr(examples):
        inputs = examples['question_text']

        model_inputs = tokenizer(inputs, return_tensors="pt")
        model_inputs = model_inputs.to(device)
        qdmr_encoding = model.generate(**model_inputs, max_length=256)
        qdmr_decomposition = tokenizer.decode(qdmr_encoding[0], skip_special_tokens=True)
        qdmr_decomposition = postprocess(qdmr_decomposition)

        new_data = {}
        new_data["decomposition"] = qdmr_decomposition

        return new_data   

    dataset.map(predict_qdmr)
    dataset.data.to_csv(csv_path, index=False)

    return csv_path


def eval_metrics(args):

    args.no_cache = True
    args.metrics = ['exact_match', 'sari', 'ged', 'normalized_exact_match']
    val_dataset = BreakDataset('validation')

    if 'joberant' in os.path.abspath('./'):
        model = BartForConditionalGeneration.from_pretrained(args.ckpt, cache_dir=cache_dir)
        tokenizer = BartTokenizer.from_pretrained(args.ckpt, cache_dir=cache_dir)
    else:
        model = BartForConditionalGeneration.from_pretrained(args.ckpt)
        tokenizer = BartTokenizer.from_pretrained(args.ckpt)

    ### declare dataset ###
    # args.dataset_file = build_dataset_file(val_dataset, args.save_dir, "val_dataset") # path to a csv dataset file, with "question_id", "decomposition" and "question_text" columns
    args.dataset_file = args.val_dataset_path # path to a csv dataset file, with "question_id", "decomposition" and "question_text" columns
    args.preds_file = build_predictions_file(model, tokenizer, val_dataset, args.save_dir, "predictions") # path to a csv predictions file, with "decomposition" column
    # args.preds_file = '/Users/meitarshechter/Git/break-evaluator/predictions.csv' # path to a csv predictions file, with "decomposition" column
    args.output_file_base = args.save_dir #'/Users/meitarshechter/Git/QDMR-for-question-generation/evaluations_log/' 

    res = process_args(args)

    # rename for AllenAI leader board
    map = {'exact_match': 'EM', 'normalized_exact_match': 'norm_EM', 'sari': 'SARI', 'ged': 'GED'}
    print(res)
    res = {map.get(k, k): v for k,v in res.items()}

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate QDMR predictions")
    parser.add_argument("--val_dataset_path", type=str, help="path to the validation dataset (a csv file)")
    parser.add_argument("--ckpt", type=str, help="path to the checkpoint to evaluate")
    parser.add_argument("--save_dir ", type=str, help="path to the directory we'd like to save the results to")
    args = parser.parse_args()

    # args.ckpt = '/Users/meitarshechter/Desktop/Studies/CS - MSc/NLP/Final-Project/checkpoint-85000'
    # args.save_dir = '/Users/meitarshechter/Git/break-evaluator'
    # args.val_dataset_path = '/Users/meitarshechter/Git/break-evaluator/val_dataset.csv'
    os.makedirs(args.save_dir, exist_ok=True)

    eval_metrics(args)

