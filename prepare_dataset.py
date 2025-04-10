import os
import shutil
import random
from tqdm import tqdm

# 데이터셋 경로 설정
SOURCE_DIR = './dataset/food-101/images'  # Food-101 데이터셋 경로
TRAIN_DIR = './food_dataset/train'
VAL_DIR = './food_dataset/val'

# 설정
IMAGES_PER_CLASS = 1000  # 클래스당 이미지 수
VAL_RATIO = 0.2  # 검증 데이터 비율

def create_directory_structure():
    """학습 및 검증 디렉토리 구조 생성"""
    # 기존 디렉토리가 있으면 삭제
    if os.path.exists(TRAIN_DIR):
        shutil.rmtree(TRAIN_DIR)
    if os.path.exists(VAL_DIR):
        shutil.rmtree(VAL_DIR)
    
    # 새 디렉토리 생성
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    
    # 각 클래스에 대한 디렉토리 생성
    for class_name in os.listdir(SOURCE_DIR):
        if os.path.isdir(os.path.join(SOURCE_DIR, class_name)):
            os.makedirs(os.path.join(TRAIN_DIR, class_name), exist_ok=True)
            os.makedirs(os.path.join(VAL_DIR, class_name), exist_ok=True)

def split_dataset():
    """데이터셋을 학습용과 검증용으로 분할"""
    print("데이터셋 분할 시작...")
    
    # 디렉토리 구조 생성
    create_directory_structure()
    
    # 각 클래스별로 이미지 분할
    for class_name in tqdm(os.listdir(SOURCE_DIR), desc="클래스 처리 중"):
        class_dir = os.path.join(SOURCE_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        # 클래스 디렉토리의 모든 이미지 파일 가져오기
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
        
        # 이미지가 1000개보다 많으면 랜덤하게 1000개만 선택
        if len(image_files) > IMAGES_PER_CLASS:
            image_files = random.sample(image_files, IMAGES_PER_CLASS)
        elif len(image_files) < IMAGES_PER_CLASS:
            print(f"경고: {class_name} 클래스의 이미지가 {len(image_files)}개로, 목표인 {IMAGES_PER_CLASS}개보다 적습니다.")
        
        # 이미지 파일 섞기
        random.shuffle(image_files)
        
        # 검증 데이터 수 계산
        val_size = int(len(image_files) * VAL_RATIO)
        
        # 검증 데이터 분할
        val_files = image_files[:val_size]
        train_files = image_files[val_size:]
        
        # 학습 데이터 복사
        for img_file in tqdm(train_files, desc=f"{class_name} 학습 데이터 복사 중", leave=False):
            src_path = os.path.join(class_dir, img_file)
            dst_path = os.path.join(TRAIN_DIR, class_name, img_file)
            shutil.copy2(src_path, dst_path)
        
        # 검증 데이터 복사
        for img_file in tqdm(val_files, desc=f"{class_name} 검증 데이터 복사 중", leave=False):
            src_path = os.path.join(class_dir, img_file)
            dst_path = os.path.join(VAL_DIR, class_name, img_file)
            shutil.copy2(src_path, dst_path)
    
    print("\n데이터셋 분할 완료!")
    
    # 데이터셋 통계 출력
    print("\n데이터셋 통계:")
    train_count = sum([len(os.listdir(os.path.join(TRAIN_DIR, d))) for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
    val_count = sum([len(os.listdir(os.path.join(VAL_DIR, d))) for d in os.listdir(VAL_DIR) if os.path.isdir(os.path.join(VAL_DIR, d))])
    num_classes = len([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
    
    print(f"총 클래스 수: {num_classes}")
    print(f"클래스당 목표 이미지 수: {IMAGES_PER_CLASS}")
    print(f"학습 데이터 수: {train_count} (클래스당 평균: {train_count/num_classes:.1f}개)")
    print(f"검증 데이터 수: {val_count} (클래스당 평균: {val_count/num_classes:.1f}개)")
    print(f"전체 데이터 수: {train_count + val_count}")
    print(f"학습:검증 비율 = {train_count/(train_count+val_count):.1%}:{val_count/(train_count+val_count):.1%}")

if __name__ == "__main__":
    # 재현성을 위한 시드 설정
    random.seed(42)
    
    # 데이터셋 분할 실행
    split_dataset() 