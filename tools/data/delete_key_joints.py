import json

input_file = "./dataset/copy_text_corpus_edited.json"
with open(input_file, "r") as f:
    data = json.load(f)

def remove_sentences_with_keyword(data, keyword):
    """
    Delete sentences containing the specified keywords from all shortest sentences and return statistics.
    """
    processed_data = {}
    total_before = 0 
    total_after = 0   
    removed_count = 0 

    for key, sentences in data.items():
        processed_sentences = []
        for sentence in sentences:
            total_before += 1
            if sentence and keyword not in sentence.lower():  
                processed_sentences.append(sentence)
                total_after += 1
            else:
                if sentence:
                    removed_count += 1
        if processed_sentences:
            processed_data[key] = processed_sentences
        else:
            processed_data[key] = [""]

    return processed_data, total_before, total_after, removed_count

keyword_to_remove = input("Enter the keyword to remove: ").strip().lower()

processed_data, total_before, total_after, removed_count = remove_sentences_with_keyword(data, keyword_to_remove)

with open(input_file, "w") as f:
    json.dump(processed_data, f, indent=4)

print("\nStatistics:")
print(f"Total sentences before removal: {total_before}")
print(f"Total sentences after removal: {total_after}")
print(f"Number of sentences removed due to '{keyword_to_remove}': {removed_count}")
print(f"The dataset has been updated and saved to '{input_file}'.")