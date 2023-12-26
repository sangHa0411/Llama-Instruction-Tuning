
import numpy as np
from typing import Dict, Any

class InstructionMetrics :

    def __init__(self, ) :
        pass
    
    # Find minimum perplexity candidate in data and check if minimum candidate and label are same.
    def get_multiple_choice_acc(self, results: Dict[str, Any]) :
        total_acc, total_acc_norm = 0, 0

        for data_id in results :
            result = results[data_id]
            log_probs = result["log_prob"]
            normalized_log_probs = result["normalized_log_prob"]
            label = result["label"]

            min_log_prob = min(log_probs)
            min_normalized_log_prob = min(normalized_log_probs)

            min_log_prob_idx = log_probs.index(min_log_prob)
            min_normalized_log_prob_idx = normalized_log_probs.index(min_normalized_log_prob)

            if min_log_prob_idx == label :
                total_acc += 1

            if min_normalized_log_prob_idx == label :
                total_acc_norm += 1

        total_acc = total_acc / len(results)
        total_acc_norm = total_acc_norm / len(results)

        return {"acc" : total_acc, "acc_norm" : total_acc_norm}

    # Just score accuracy using generation after ####.
    def get_gsm8k_acc(self, results: Dict[str, Any]) :
        total_acc = 0
        for data_id in results :
            result = results[data_id]

            generation = result["generation"]
            label = result["label"] 

            generation_tgt = generation.split("####")[-1].strip()
            label_tgt = label.split("####")[-1].strip()

            if generation_tgt == label_tgt :
                total_acc += 1

        total_acc = total_acc / len(results)
        return {"acc" : total_acc}

    # This code and algorithm are based on https://github.com/voidism/DoLa/blob/main/tfqa_mc_eval.py.
    def get_truthful_qa_mc2(self, results: Dict[str, Any]) :
        mc2 = 0
        for data_id in results :
            result = results[data_id]
            log_probs = result["log_prob"]
            labels = result["label"]

            scores_true, scores_false = [], []
            for i in range(len(labels)) :
                label = labels[i]

                if label == 1 :
                    scores_true.append(log_probs[i])
                else :
                    scores_false.append(log_probs[i])

            probs_true = np.exp(scores_true)
            probs_false = np.exp(scores_false)

            probs_true = probs_true / (sum(probs_true) + sum(probs_false))
            mc2 += sum(probs_true)

        mc2 /= len(results)
        total_mc2 = {"mc2" : mc2}    
        return total_mc2

