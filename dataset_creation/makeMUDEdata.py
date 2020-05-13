import json

prefix = "data/"
input_path = prefix + "train.txt"
label_path = prefix + "train_label.jsonl"
output_path = prefix + "train_correct.txt"

with open(input_path) as f:
    with open(label_path) as lf:
        with open(output_path,"w") as wf:
            for input_line, label_line in zip(f, lf):
                loaded_label = json.loads(label_line)
                tokens = input_line.strip().split()
                labels = {int(index): label for index, label in loaded_label.items()}
                tokens = input_line.strip().split()
                for i, label in labels.items():
                    tokens[i] = label
                wf.write(" ".join(tokens) + "\n")
