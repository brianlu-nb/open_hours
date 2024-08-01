from typing import Optional
from hours_prompts import Prompt, PromptType as ptype

report_to_bool = (
'''{title}

.                  |   Expected Positives  |   Expected Negatives  |       Total       
-------------------+-----------------------+-----------------------+--------------------
Reported Positives | {tp:^21} | {fp:^21} | {rp:^17} 
-------------------+-----------------------+-----------------------+--------------------
Reported Negatives | {fn:^21} | {tn:^21} | {rn:^17} 
-------------------+-----------------------+-----------------------+--------------------
. Error in Report  | {nfp:^21} | {nfn:^21} | {rnf:^17} 
-------------------+-----------------------+-----------------------+--------------------
.     Total        | {ep:^21} | {en:^21} | {total:^17}

Accuracy: {acc:8.4f}%

# Trials: {num_trials}
# Correct: {correct}
# Error: {err}

% Correct: {correctp:8.4f}%
% Error: {errp:8.4f}%

.            |  Precision  |    Recall   |   F1 Score
-------------+-------------+-------------+-------------
. Positives  |  {pprec:8.4f}%  |  {precl:8.4f}%  |  {pf1:8.4f}%
-------------+-------------+-------------+-------------
. Negatives  |  {nprec:8.4f}%  |  {nrecl:8.4f}%  |  {nf1:8.4f}%

''')

report_to_hours = (
'''{title}

.                  |       Result       
-------------------+--------------------
.     Correct      | {rp:^17} 
-------------------+--------------------
.    Incorrect     | {rn:^17} 
-------------------+--------------------
. Error in Report  | {rnf:^17} 
-------------------+--------------------
.     Total        | {total:^17}

Accuracy: {acc:8.4f}%

# Trials: {num_trials}
# Correct: {correct}
# Error: {err}

% Correct: {correctp:8.4f}%
% Error: {errp:8.4f}%

''')

report_to_list = (
'''{title}

.                  |   Expected Positives  |   Expected Negatives  |    New Invented Key   |       Total       
-------------------+-----------------------+-----------------------+-----------------------+--------------------
Reported Positives | {tp:^21} | {fp:^21} | {nkp:^21} | {rp:^17} 
-------------------+-----------------------+-----------------------+-----------------------+--------------------
Reported Negatives | {fn:^21} | {tn:^21} | {nkn:^21} | {rn:^17} 
-------------------+-----------------------+-----------------------+-----------------------+--------------------
.  Key Not Found   | {nfp:^21} | {nfn:^21} |          ---          | {rnf:^17} 
-------------------+-----------------------+-----------------------+-----------------------+--------------------
.     Total        | {ep:^21} | {en:^21} | {tnk:^21} | {total:^17}

Accuracy: {acc:8.4f}%

# Trials: {num_trials}
# Correct: {correct}
# Error: {err}

% Correct: {correctp:8.4f}%
% Error: {errp:8.4f}%

.            |  Precision  |    Recall   |   F1 Score
-------------+-------------+-------------+-------------
. Positives  |  {pprec:8.4f}%  |  {precl:8.4f}%  |  {pf1:8.4f}%
-------------+-------------+-------------+-------------
. Negatives  |  {nprec:8.4f}%  |  {nrecl:8.4f}%  |  {nf1:8.4f}%

''')

report_templates = {
    ptype.TO_BOOL: report_to_bool,
    ptype.TO_HOURS: report_to_hours,
    ptype.TO_LIST: report_to_list,
}

class Report:
    def __init__(self, prompt_type: Optional[ptype]=None):
        self.prompt_type = prompt_type
        self._count = 0
        self._tp = 0
        self._tn = 0
        self._fp = 0
        self._fn = 0
        self._ep = 0
        self._en = 0
        self._rp = 0
        self._rn = 0
        self._prompt_errors = 0
        self._individual_errors = 0
        self._correct = 0
    
    @property
    def true_positives(self):
        return self._tp
    @property
    def true_negatives(self):
        return self._tn
    @property
    def false_positives(self):
        return self._fp
    @property
    def false_negatives(self):
        return self._fn
    @property
    def expected_positives(self):
        return self._ep
    @property
    def expected_negatives(self):
        return self._en
    @property
    def reported_positives(self):
        return self._rp
    @property
    def reported_negatives(self):
        return self._rn
    @property
    def count(self):
        return self._count
    @property
    def correct(self):
        return self._correct
    @property
    def prompt_errors(self):
        return self._prompt_errors
    @property
    def individual_errors(self):
        return self._individual_errors
    @property
    def total(self):
        return self._ep + self._en
    @property
    def accuracy(self):
        return (self._tp + self._tn) / self.total if self.total > 0 else 0
    @property
    def positive_precision(self):
        return self._tp / self._rp if self._rp > 0 else 0
    @property
    def positive_recall(self):
        return self._tp / self._ep if self._ep > 0 else 0
    @property
    def positive_F1(self):
        precision = self.positive_precision
        recall = self.positive_recall
        return 2 * (precision * recall) / (precision + recall) if precision > 0 and recall > 0 else 0
    @property
    def negative_precision(self):
        return self._tn / self._rn if self._rn > 0 else 0
    @property
    def negative_recall(self):
        return self._tn / self._en if self._en > 0 else 0
    @property
    def negative_F1(self):
        precision = self.negative_precision
        recall = self.negative_recall
        return 2 * (precision * recall) / (precision + recall) if precision > 0 and recall > 0 else 0
    @property
    def positives_not_found(self):
        return self._ep - self._tp - self._fn
    @property
    def negatives_not_found(self):
        return self._en - self._tn - self._fp
    @property
    def total_not_found(self):
        return self.positives_not_found + self.negatives_not_found
    @property
    def positive_new_keys(self):
        return self._rp - self._tp - self._fp
    @property
    def negative_new_keys(self):
        return self._rn - self._tn - self._fn
    @property
    def total_new_keys(self):
        return self.positive_new_keys + self.negative_new_keys

    def update(self, prompt: Prompt, response=None):
        self._count += 1
        self._ep += prompt.expected_positives()
        self._en += prompt.expected_negatives()
        if response:
            self._tp += prompt.true_positives(response)
            self._tn += prompt.true_negatives(response)
            self._fp += prompt.false_positives(response)
            self._fn += prompt.false_negatives(response)
            self._rp += prompt.reported_positives(response)
            self._rn += prompt.reported_negatives(response)
            if prompt.correct(response):
                self._correct += 1
        else:
            self._prompt_errors += 1
            self._individual_errors += len(prompt.hours)

    def report_str(self, title: Optional[str] = 'output'):
        output_type = self.prompt_type if self.prompt_type else ptype.TO_LIST
        return report_templates[output_type].format(
            title=title,
            tp=self.true_positives, tn=self.true_negatives, fp=self.false_positives, fn=self.false_negatives,
            ep=self.expected_positives, en=self.expected_negatives, rp=self.reported_positives, rn=self.reported_negatives,
            total=self.total,
            nfp=self.positives_not_found, nfn=self.negatives_not_found, rnf=self.total_not_found,
            nkp=self.positive_new_keys, nkn=self.negative_new_keys, tnk=self.total_new_keys,
            pprec=self.positive_precision * 100, precl=self.positive_recall * 100, pf1=self.positive_F1 * 100,
            nprec=self.negative_precision * 100, nrecl=self.negative_recall * 100, nf1=self.negative_F1 * 100,
            acc=self.accuracy * 100,
            num_trials=self.count,
            err=self.prompt_errors, errp=self.prompt_errors / self.count * 100 if self.count > 0 else 0,
            correct=self.correct, correctp=self.correct / self.count * 100 if self.count > 0 else 0,
        )

    def postfix(self):
        return {
            "Correct": self.correct,
            "Error": self.prompt_errors,
            "Accuracy": self.accuracy,
            "Precision": self.positive_precision,
            "Recall": self.positive_recall
        }