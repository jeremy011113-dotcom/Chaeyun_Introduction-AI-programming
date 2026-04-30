import json
with open('week5_2_mnist_cnn.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if '✏️ 여기에 도전 과제 코드를 작성하세요!' in source:
            new_source = """# ✏️ 여기에 도전 과제 코드를 작성하세요!

# == 기초 2번: Dropout 비율 변경 (0.3, 0.7) ===
# Dropout을 0.3, 0.7로 변경하여 학습 결과를 확인하는 코드입니다.
dropout_rates = [0.3, 0.7]

print("--- 🟢 기초 2번 도전 과제: Dropout 비율 비교 실험 시작 ---")

for dr in dropout_rates:
    print(f"\\n▶ 실험 진행 중: Dropout Rate = {dr}")
    
    # 1. 모델 생성 (새로운 dropout 비율 적용)
    model_challenge = MnistCNN(dropout_rate=dr).to(device)
    
    # 2. 학습 설정
    criterion_challenge = nn.CrossEntropyLoss()
    optimizer_challenge = optim.Adam(model_challenge.parameters(), lr=0.001)
    
    # 빠른 실험을 위해 Epoch을 5로 설정 (필요시 늘릴 수 있습니다)
    NUM_EPOCHS_CHALLENGE = 5 
    
    for epoch in range(1, NUM_EPOCHS_CHALLENGE + 1):
        # 훈련
        model_challenge.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer_challenge.zero_grad()
            outputs = model_challenge(images)
            loss = criterion_challenge(outputs, labels)
            loss.backward()
            optimizer_challenge.step()
        
        # 5번째 Epoch(마지막)에서만 결과 출력
        if epoch == NUM_EPOCHS_CHALLENGE:
            model_challenge.eval()
            tr_loss, tr_correct, tr_total = 0.0, 0, 0
            te_loss, te_correct, te_total = 0.0, 0, 0
            
            with torch.no_grad():
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model_challenge(images)
                    tr_loss += criterion_challenge(outputs, labels).item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    tr_correct += (predicted == labels).sum().item()
                    tr_total += images.size(0)
                    
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model_challenge(images)
                    te_loss += criterion_challenge(outputs, labels).item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    te_correct += (predicted == labels).sum().item()
                    te_total += images.size(0)
            
            tr_acc = tr_correct / tr_total
            te_acc = te_correct / te_total
            tr_loss = tr_loss / tr_total
            te_loss = te_loss / te_total
            
            print(f"   [결과] Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc*100:.1f}%")
            print(f"   [결과] Test Loss:  {te_loss:.4f} | Test Acc:  {te_acc*100:.1f}%")
            print(f"   => Train/Test 정확도 차이: {(tr_acc - te_acc)*100:.2f}%p (과적합 지표)")

print("\\n✅ Dropout 실험 완료!")
"""
            # Split the new_source back into list of lines for jupyter
            cell['source'] = [line + '\n' for line in new_source.strip("\n").split('\n')]
            break

with open('week5_2_mnist_cnn.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
