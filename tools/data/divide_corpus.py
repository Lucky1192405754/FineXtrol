import json

input_file = "./dataset/copy_final_mirror_ori_humanml3d_posefix_annotations_interval0.5_pose_change_th1.0.json"
with open(input_file, "r") as f:
    data = json.load(f)

def split_sentences(data):
    """
    Split each sentence by periods into minimal sentence units.
    """
    processed_data = {}
    for key, sentences in data.items():
        processed_sentences = []
        for sentence in sentences:
            if sentence:
                min_units = [s.strip() + "." for s in sentence.split('.') if s.strip()]
                processed_sentences.extend(min_units)
            else:
                processed_sentences.append("")
        processed_data[key] = processed_sentences
    return processed_data

processed_data = split_sentences(data)

with open(input_file, "w") as f:
    json.dump(processed_data, f, indent=4)

print("Sentences have been split into minimum units and saved to the original file.")