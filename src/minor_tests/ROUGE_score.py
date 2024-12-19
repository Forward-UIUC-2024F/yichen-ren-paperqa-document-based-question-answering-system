import os
from rouge_score import rouge_scorer

dir_true = "pubmedqa_answer"
dir_pred = "pubmedqa_answer_Llama3_1000_dense_path_retrieval"

output_file = "rouge_scores_1000_dense_path_retrieval.csv"

def calculate_rouge(true_file, pred_file):
    with open(true_file, 'r') as f:
        reference_text = f.read().strip()
    with open(pred_file, 'r') as f:
        candidate_text = f.read().strip()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_text, candidate_text)
    return scores

with open(output_file, 'w') as out_file:
    for filename in os.listdir(dir_pred):
        true_file = os.path.join(dir_true, filename)
        pred_file = os.path.join(dir_pred, filename)

        if os.path.exists(true_file):
            rouge_scores = calculate_rouge(true_file, pred_file)
            # out_file.write(f"ROUGE Scores for {filename}:\n")
            out_file.write(f"{filename.split('.')[0]},")
            
            for key, score in rouge_scores.items():
                # out_file.write(f"  {key}: Precision={score.precision:.2f}, Recall={score.recall:.2f}, F1={score.fmeasure:.2f}\n")
                out_file.write(f"{score.precision:.2f}, {score.recall:.2f}, {score.fmeasure:.2f},")
            out_file.write("\n")
        else:
            out_file.write(f"Missing true file for {filename}\n\n")

print(f"ROUGE scores have been saved to {output_file}.")