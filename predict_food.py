import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import argparse
import matplotlib.pyplot as plt
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
    original_image = image.copy()
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
    
    return predicted_class, confidence, all_probabilities, original_image

def visualize_prediction(image, predicted_class, confidence, all_probabilities, class_names):
    """
    예측 결과를 시각화하는 함수
    
    Args:
        image: 원본 이미지
        predicted_class: 예측된 클래스 이름
        confidence: 예측 신뢰도
        all_probabilities: 모든 클래스에 대한 확률
        class_names: 클래스 이름 목록
    """
    plt.figure(figsize=(12, 5))
    
    # 이미지 표시
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f'예측: {predicted_class} (신뢰도: {confidence:.2f})')
    plt.axis('off')
    
    # 확률 막대 그래프
    plt.subplot(1, 2, 2)
    top_k = min(5, len(class_names))  # 상위 5개 클래스만 표시
    top_indices = all_probabilities.argsort()[-top_k:][::-1]
    top_classes = [class_names[i] for i in top_indices]
    top_probs = all_probabilities[top_indices]
    
    plt.barh(range(top_k), top_probs, color='skyblue')
    plt.yticks(range(top_k), top_classes)
    plt.xlabel('확률')
    plt.title('상위 예측 클래스')
    
    plt.tight_layout()
    plt.savefig('prediction_result.png')
    plt.close()

def main():
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='음식 이미지 분류 예측')
    parser.add_argument('--model_paths', type=str, nargs='+', required=True,
                        help='학습된 앙상블 모델 파일 경로들')
    parser.add_argument('--image_path', type=str, required=True,
                        help='예측할 이미지 파일 경로')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='결과를 저장할 디렉토리')
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
    
    # 이미지 예측
    try:
        predicted_class, confidence, all_probabilities, original_image = predict_image(
            models, args.image_path, transform, device, class_names
        )
        
        # 결과 출력
        print(f"\n이미지 '{args.image_path}' 예측 결과:")
        print(f"예측 클래스: {predicted_class}")
        print(f"신뢰도: {confidence:.4f}")
        
        # 결과 시각화
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        visualize_prediction(original_image, predicted_class, confidence, all_probabilities, class_names)
        print(f"예측 결과가 '{args.output_dir}/prediction_result.png'에 저장되었습니다.")
        
    except Exception as e:
        print(f"예측 중 오류 발생: {e}")

if __name__ == '__main__':
    main() 