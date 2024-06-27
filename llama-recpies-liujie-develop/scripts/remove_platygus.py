import json

def remove(in_file, out_file):
    remove_count = 0
    with open(out_file, 'w') as fout:
        with open(in_file, 'r') as f:
            for line in f:
                if line.find("\"source\": \"platypus\"")>=0:
                    remove_count += 1
                    continue
                fout.write(line)
    print(remove_count)

remove('../data/unified_train_1016.jsonl','../data/unified_train_1016_clean.jsonl')
remove('../data/unified_valid_1016.jsonl','../data/unified_valid_1016_clean.jsonl')