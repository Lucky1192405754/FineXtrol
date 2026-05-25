import json

input_file = "./dataset/deleted_text_corpus_copy.json"
with open(input_file, "r") as f:
    data = json.load(f)

def remove_duplicate_sentences(data):
    """
    Remove duplicated sentences under each key so each sentence is kept only once.
    """
    unique_data = {}
    for key, sentences in data.items():
        seen = set()
        unique_sentences = []
        for sentence in sentences:
            if sentence not in seen:
                unique_sentences.append(sentence)
                seen.add(sentence)
        unique_data[key] = unique_sentences
    return unique_data

unique_data = remove_duplicate_sentences(data)

output_file = "./dataset/unique_text_corpus_copy.json"
with open(output_file, "w") as f:
    json.dump(unique_data, f, indent=4)

print(f"Duplicate sentences have been removed. The unique dataset has been saved to '{output_file}'.")