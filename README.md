# Sarcasm Detection in Text
## Introduction
This project is focused on building and training a Natural Language Processing model to detect sarcasm in textual data. Utilizing advanced machine learning techniques and NLP libraries, the goal is to accurately discern sarcastic statements in a given dataset.

## Usage (How to run)
There are three programs: subtask_A.py, subtask_B.py, and data_visualization_A.py. subtask_A.py and subtask_B.py are the programs used to train and
test the classifiers. To run these programs from the command terminal, you type 
'python "program name" "classifier method" "training data filename" "test data filename"'. The options for classifier methods are 'nb' for naive bayes,
'lr' for logistic regression, and 'baseline' for the most frequent baseline. Alternatively you can output to a file by typing
'python "program name" "classifier method" "training data filename" "test data filename" > "output filename"'

data_visualization_A.py is used to plot the different f1 scores for the different sized NB ngrams. Type "python data_visualization_A.py" in the 
terminal to run. You must make sure that you use standard output to save the output
from subtask_A.py to the filenames given inside the program. You also want to make sure in subtask_A.py that you are using the appropriate ngram_range values on 
line 87. In the github repo, there are already the calculated output files, so running the program itself should work.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Contact
For any queries or support, please contact us at email@email.com.
