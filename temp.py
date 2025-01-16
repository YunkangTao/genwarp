import json

# 文件路径
file_path = '/mnt/chenyang_lei/Datasets/easyanimate_dataset/EvaluationSet/Kubric-4D/gcd_validation.json'

# 读取JSON文件
with open(file_path, 'r') as file:
    data = json.load(file)

# 交换video_file_path和camera_file_path的值
for item in data:
    video_file_path = item['video_file_path']
    camera_file_path = item['camera_file_path']
    item['video_file_path'] = camera_file_path
    item['camera_file_path'] = video_file_path

# 将修改后的数据写回到文件中
with open(file_path, 'w') as file:
    json.dump(data, file, indent=4)

print("交换完成！")