prefix = "data/"
input_path = prefix+"test.txt"
label_path = prefix+"test_label.jsonl"
output_path = prefix+"test_detection.tsv"

separator = " "

with open(input_path, "r") as f:
    with open(label_path, "r") as lf:
        with open(output_path,"w") as wf:
            wf.write("text_a\tlabel\n")
            for line, label_line in zip(f, lf):
                tokens = line.strip().split()
                label = eval(label_line.strip())
                label = {int(k):v for k, v in label.items()}
                                                                                                                    
                out_labels = ["O"]*len(tokens)
                for index in label:
                    out_labels[index] = "B-M"
                wf.write(separator.join(tokens)+"\t"+separator.join(out_labels)+"\n")