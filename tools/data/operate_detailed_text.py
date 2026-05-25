import json
import re
from tqdm import tqdm

body_parts = [
    'body', 'toes', 'head', 'waist', 'torso', 'left chest', 'right chest',
    'left leg', 'right leg', 'left arm', 'right arm', 'left foot', 'right foot',
    'right hand', 'left hand', 'left elbow', 'right elbow', 'left heel', 'right heel',
    'left knee', 'right knee', 'left shoulder', 'right shoulder', 'left forearm',
    'right forearm', 'left thigh', 'right thigh', 'left hip', 'right hip',
    'left ankle', 'right ankle'
]

plural_to_parts = {
    'chests': ['left chest', 'right chest'],
    'legs': ['left leg', 'right leg'],
    'arms': ['left arm', 'right arm'],
    'feet': ['left foot', 'right foot'],
    'hands': ['left hand', 'right hand'],
    'elbows': ['left elbow', 'right elbow'],
    'heels': ['left heel', 'right heel'],
    'knees': ['left knee', 'right knee'],
    'shoulders': ['left shoulder', 'right shoulder'],
    'forearms': ['left forearm', 'right forearm'],
    'thighs': ['left thigh', 'right thigh'],
    'hips': ['left hip', 'right hip'],
    'ankles': ['left ankle', 'right ankle']
}

with open('./dataset/mirror_ori_humanml3d_posefix_annotations_interval0.5_pose_change_th1.0.json', 'r') as file:
    data = json.load(file)

total_modifications = 0

def process_text(text):
    global total_modifications

    if not text.strip():
        return text

    text = re.sub(r'\s*,\s*', '. ', text)
    text = re.sub(r'\s+and\s+', '. ', text)

    text = re.sub(r'\.\s*([a-z])', lambda match: '. ' + match.group(1).upper(), text)

    sentences = re.split(r'[.]\s*', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    expanded_sentences = []
    for sentence in sentences:
        modified = False
        for plural, parts in plural_to_parts.items():
            if plural in sentence:
                left_part = parts[0]
                right_part = parts[1]
                left_sentence = sentence.replace(plural, left_part)
                right_sentence = sentence.replace(plural, right_part)
                expanded_sentences.append(left_sentence)
                expanded_sentences.append(right_sentence)
                total_modifications += 1
                modified = True
        if not modified:
            expanded_sentences.append(sentence)
    
    return '. '.join(expanded_sentences) + '.'

processed_data = {}
for key, texts in tqdm(data.items(), desc="Processing data"):
    processed_data[key] = [process_text(text) if text.strip() else text for text in texts]

with open('./dataset/processed_mirror_ori_humanml3d_posefix_annotations_interval0.5_pose_change_th1.0.json', 'w') as file:
    json.dump(processed_data, file, indent=4)

print(f"Data processing is complete and has been saved to a new file.")
print(f"Modified {total_modifications} locations in total.")