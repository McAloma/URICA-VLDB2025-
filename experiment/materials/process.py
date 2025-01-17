import json
import random

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def sample_n_objects(data, n):
    return random.sample(data, n)

def main():
    input_file = "experiment/materials/query_region_infos_alpha.json"  
    output_file = "experiment/materials/query_region_infos_alpha_sample.json" 
    n = 20 

    data = load_json(input_file)
    sampled_data = sample_n_objects(data, n)
    save_json(sampled_data, output_file)

if __name__ == "__main__":
    main()
