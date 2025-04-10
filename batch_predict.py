import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from cnn_ensemble_training import FoodCNN, load_saved_model

def predict_image(models, image_path, transform, device, class_names):
    """
    앙상블 모델을 사용하여 단일 이미지에 대한 예측을 수행하는 함수
    
    Args:
        models: 학습된 앙상블 모델 리스트
        image_path (str): 예측할 이미지 경로
        transform: 이미지 전처리 변환
        device: 모델이 있는 디바이스
        class_names: 클래스 이름 목록
    
    Returns:
        predicted_class: 예측된 클래스 이름
        confidence: 예측 신뢰도
        all_probabilities: 모든 클래스에 대한 확률
    """
    # 이미지 로드 및 전처리
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    # 앙상블 예측
    ensemble_outputs = []
    for model in models:
        model.eval()
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            ensemble_outputs.append(probabilities)
    
    # 앙상블 결과 평균
    ensemble_probabilities = torch.mean(torch.stack(ensemble_outputs), dim=0)
    confidence, predicted = torch.max(ensemble_probabilities, 1)
    
    predicted_class = class_names[predicted.item()]
    confidence = confidence.item()
    all_probabilities = ensemble_probabilities.squeeze().cpu().numpy()
    
    return predicted_class, confidence, all_probabilities

def process_directory(models, directory, transform, device, class_names, output_file):
    """
    디렉토리 내의 모든 이미지에 대한 앙상블 예측을 수행하는 함수
    
    Args:
        models: 학습된 앙상블 모델 리스트
        directory (str): 이미지가 있는 디렉토리 경로
        transform: 이미지 전처리 변환
        device: 모델이 있는 디바이스
        class_names: 클래스 이름 목록
        output_file (str): 결과를 저장할 CSV 파일 경로
    """
    results = []
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    for image_file in tqdm(image_files, desc="이미지 처리 중"):
        image_path = os.path.join(directory, image_file)
        try:
            predicted_class, confidence, all_probabilities = predict_image(
                models, image_path, transform, device, class_names
            )
            
            # 상위 3개 예측 클래스와 확률 추출
            top_k = min(3, len(class_names))
            top_indices = all_probabilities.argsort()[-top_k:][::-1]
            top_classes = [class_names[i] for i in top_indices]
            top_probs = all_probabilities[top_indices]
            
            # 결과 저장
            result = {
                '이미지 파일': image_file,
                '예측 클래스': predicted_class,
                '신뢰도': confidence
            }
            
            # 상위 예측 클래스와 확률 추가
            for i in range(top_k):
                result[f'상위{i+1} 클래스'] = top_classes[i]
                result[f'상위{i+1} 확률'] = top_probs[i]
            
            results.append(result)
            
        except Exception as e:
            print(f"이미지 '{image_file}' 처리 중 오류 발생: {e}")
    
    # 결과를 DataFrame으로 변환하고 CSV로 저장
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"예측 결과가 '{output_file}'에 저장되었습니다.")
    
    return df

def main():
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='음식 이미지 배치 분류 예측')
    parser.add_argument('--model_paths', type=str, nargs='+', required=True,
                        help='학습된 앙상블 모델 파일 경로들')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='예측할 이미지가 있는 디렉토리 경로')
    parser.add_argument('--output_file', type=str, default='./results/batch_predictions.csv',
                        help='결과를 저장할 CSV 파일 경로')
    args = parser.parse_args()
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"디바이스: {device}")
    
    # 앙상블 모델 로드
    models = []
    try:
        for model_path in args.model_paths:
            model, _, _, _, _, _, class_names = load_saved_model(model_path, device)
            models.append(model)
        print(f"앙상블 모델이 성공적으로 로드되었습니다. 모델 수: {len(models)}, 클래스 수: {len(class_names)}")
    except FileNotFoundError as e:
        print(f"모델 파일을 찾을 수 없습니다: {e}")
        return
    
    # 이미지 전처리 변환 정의
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 출력 디렉토리 생성
    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 배치 처리 실행
    process_directory(models, args.input_dir, transform, device, class_names, args.output_file)

if __name__ == '__main__':
    main() 