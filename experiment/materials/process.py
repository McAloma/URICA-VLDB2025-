import json
import random

# 加载原始 JSON 文件
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 保存新的 JSON 文件
def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 随机抽取 n 个对象
def sample_n_objects(data, n):
    return random.sample(data, n)

def main():
    input_file = "experiment/materials/query_region_infos_alpha.json"  # 输入的 JSON 文件路径
    output_file = "experiment/materials/query_region_infos_alpha_sample.json"  # 输出的 JSON 文件路径
    n = 20  # 想要抽取的对象数量

    # 加载 JSON 文件
    data = load_json(input_file)

    # 随机抽取 n 个对象
    sampled_data = sample_n_objects(data, n)

    # 保存结果为新的 JSON 文件
    save_json(sampled_data, output_file)
    print(f"抽取的 {n} 个对象已保存至 {output_file}")

if __name__ == "__main__":
    main()
