import torch
import json

# 저장된 모델 파일 불러오기
checkpoint = torch.load("saved_models/food_classification_model.pth", map_location='cpu')

# 필요한 값 추출
result_data = {
    "train_accs": checkpoint["train_accs"],
    "val_accs": checkpoint["val_accs"],
    "train_losses": checkpoint["train_losses"],
    "val_losses": checkpoint["val_losses"]
}

# JSON 파일로 저장
with open("training_results.json", "w") as f:
    json.dump(result_data, f)

print("학습 결과가 'training_results.json' 파일로 저장되었습니다.")
