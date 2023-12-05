# Sarcasm Detection in Text
## Introduction
This project is focused on building and training a Natural Language Processing model to detect sarcasm in textual data. Utilizing advanced machine learning techniques and NLP libraries, the goal is to accurately discern sarcastic statements in a given dataset.

## Usage (How to run)
There are four programs: subtask_A.py, subtask_B.py, and data_visualization_A.py. subtask_A.py, subtask_C.py, and subtask_B.py are the programs used to train and
test the classifiers. To run these programs from the command terminal, you type 
'python "program name" "classifier method" "training data filename" "test data filename"'. The options for classifier methods are 'nb' for naive bayes,
'lr' for logistic regression, and 'baseline' for the most frequent baseline. Alternatively you can output to a file by typing
'python "program name" "classifier method" "training data filename" "test data filename" > "output filename"'

data_visualization_A.py is used to plot the different f1 scores for the different sized NB ngrams for subtask A. Type "python data_visualization_A.py" in the 
terminal to run. You must make sure that you use standard output to save the output
from subtask_A.py to the filenames given inside the program. You also want to make sure in subtask_A.py that you are using the appropriate ngram_range values on 
line 85. In the github repo, there are already the calculated output files, so running the program itself should work. 
These files are 'nb1_Ar.txt' to 'nb4_Ar.txt' and 'nb1_En.txt' to 'nb4_En.txt'

For subtask A the training data is 'train.Ar.taskA.csv', 'train.En.taskA.csv' and the test data is 'task_A_Ar_test.csv' and 'task_A_En_test.csv'
For subtask B the training data is 'train.En.taskB.csv' and the test data is 'task_B_En_test.csv'
For subtask C the training data is 'train.Ar.taskC.csv' and the test data is 'task_C_Ar_test.csv'

## Citation for Datasets
This repository contains the datasets used for iSarcasmEval shared-task (Task 6 at SemEval 2022). The full details are available in the overview paper SemEval-2022 Task 6: iSarcasmEval, Intended Sarcasm Detection in English and Arabic.
```
@inproceedings{abu-farha-etal-2022-semeval,
    title = "{S}em{E}val-2022 Task 6: i{S}arcasm{E}val, Intended Sarcasm Detection in {E}nglish and {A}rabic",
    author = "Abu Farha, Ibrahim  and
      Oprea, Silviu Vlad  and
      Wilson, Steven  and
      Magdy, Walid",
    booktitle = "Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022)",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.semeval-1.111",
    pages = "802--814",
    abstract = "iSarcasmEval is the first shared task to target intended sarcasm detection: the data for this task was provided and labelled by the authors of the texts themselves. Such an approach minimises the downfalls of other methods to collect sarcasm data, which rely on distant supervision or third-party annotations. The shared task contains two languages, English and Arabic, and three subtasks: sarcasm detection, sarcasm category classification, and pairwise sarcasm identification given a sarcastic sentence and its non-sarcastic rephrase. The task received submissions from 60 different teams, with the sarcasm detection task being the most popular. Most of the participating teams utilised pre-trained language models. In this paper, we provide an overview of the task, data, and participating teams.",
}
```


