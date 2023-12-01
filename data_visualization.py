import matplotlib.pyplot as plt

def clean_string(input_str):
    # Remove null bytes and whitespace from the input string
    return input_str.replace('\x00', '').strip()

def plot_f1_scores(file_paths, labels):
    plt.figure(figsize=(10, 6))

    # Create a list of x-values for all data points

    for i, file_path in enumerate(file_paths):
        with open(file_path, 'r') as file:
            lines = file.readlines()

            print('test')
            score = lines[2].split(',')[2]
            print(score)
            clean_score = clean_string(score)
            print(float(clean_score))
            #float_val = float(score)
            #print(float_val)
            # Extract F1 score from the .txt file
            f1_score = float(clean_score)
            plt.scatter(labels[i],f1_score, marker='o', color='black')

    plt.xlabel('Ngram Size')
    plt.ylabel('F1 Score')
    plt.xticks(range(1, len(file_paths) + 1), labels)
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    
    # Plotting evaluation metrics output to these files using subtask_A.py
    file_paths = ['nb1_english.txt', 'nb2_english.txt', 'nb3_english.txt', 'nb4_english.txt']
    labels = [1,2,3,4]
    plot_f1_scores(file_paths, labels)

    file_paths = ['nb1_Ar.txt', 'nb2_Ar.txt', 'nb3_Ar.txt', 'nb4_Ar.txt']
    labels = [1,2,3,4]
    plot_f1_scores(file_paths,labels)


