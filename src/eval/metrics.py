
import evaluate
from typing import List

class InstructionMetrics :

    def __init__(self, ) :
        self.bleu_scorer = evaluate.load("bleu")

    def get_multiple_exact_match(self, predictions: List[str], labels: List[str]) :
        assert len(predictions) == len(labels)

        total_acc = 0
        for p, l in zip(predictions, labels) :
            if p.strip() == l.strip() :
                total_acc += 1

        total_acc = total_acc / len(predictions)
        return {"exact_match" : total_acc}

    def get_gsm8k_accuracy(self, predictions: List[str], labels: List[str]) :
        assert len(predictions) == len(labels)

        total_acc = 0
        for p, l in zip(predictions, labels) :
            p_tgt = p.split("####")[-1].strip()
            l_tgt = l.split("####")[-1].strip()

            if p_tgt == l_tgt :
                total_acc += 1

        total_acc = total_acc / len(predictions)
        return {"acc" : total_acc}

    def get_truthful_qa_blue(self, predictions: List[str], labels: List[List[str]]) :
        assert len(predictions) == len(labels)

        score = self.bleu_scorer.compute(predictions=predictions, rferences=labels)
        total_bleu = score["bleu"]
        return {"blue" : total_bleu}
