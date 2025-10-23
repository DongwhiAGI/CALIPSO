import torch

def convert_ordinal_to_class_preds(ordinal_logits: torch.Tensor) -> torch.Tensor:
    """
    순서 회귀 모델의 로짓을 최종 클래스 예측으로 변환합니다.
    P(y > k) > 0.5 인 k의 개수를 세어 예측 클래스를 결정합니다.
    """
    # 로짓에 sigmoid를 적용하여 (0, 1) 사이의 확률로 변환
    probs = torch.sigmoid(ordinal_logits)
    
    # 확률이 0.5보다 큰 경우를 모두 더해서 예측 클래스를 만듦
    # 예: [0.9, 0.8, 0.2] -> True, True, False -> 2. 즉, 클래스 2로 예측
    preds = (probs > 0.5).sum(dim=1)
    return preds

def convert_ordinal_to_preds_and_probs(ordinal_logits: torch.Tensor):
    """
    순서 회귀 로짓을 최종 예측 클래스와 해당 클래스의 확률로 변환합니다.
    """
    # 로짓을 (0, 1) 사이의 누적 확률 P(y > k)로 변환
    cum_probs = torch.sigmoid(ordinal_logits)
    
    # 예측 클래스는 P(y > k) > 0.5 인 k의 개수
    # 예: [0.9, 0.8, 0.2] -> True, True, False -> 2. 즉, 클래스 2
    preds = (cum_probs > 0.5).sum(dim=1)
    
    # 각 클래스 k가 정답일 확률 P(y = k) 계산
    # P(y=k) = P(y>k-1) - P(y>k)
    # 계산을 위해 양쪽에 P(y>-1)=1 과 P(y>N-1)=0 을 추가
    B, N_minus_1 = cum_probs.shape
    ones = torch.ones((B, 1), device=cum_probs.device)
    zeros = torch.zeros((B, 1), device=cum_probs.device)
    
    padded_probs = torch.cat([ones, cum_probs, zeros], dim=1) # (B, N+1)
    
    # 각 클래스에 대한 확률
    # ex) P(y=0)=P(y>-1)-P(y>0)=1-P(y>0), P(y=1)=P(y>0)-P(y>1), ...
    class_probs = padded_probs[:, :-1] - padded_probs[:, 1:]
    
    # 최종 예측 클래스에 해당하는 확률을 가져옴
    # preds.unsqueeze(1) -> (B, 1) 형태의 인덱스 텐서
    pred_probs = torch.gather(class_probs, 1, preds.unsqueeze(1)).squeeze(1)
    
    return preds, pred_probs

def calculate_accuracy(logits_x, logits_y, logits_z, labels):
    """3축 로짓과 라벨로부터 축별/완전 일치 정확도를 위한 정답 개수를 계산합니다."""
    # 1. 각 축에 대한 예측 클래스 구하기 (가장 높은 logit 값의 인덱스)
    preds_x = torch.argmax(logits_x, dim=1)
    preds_y = torch.argmax(logits_y, dim=1)
    preds_z = torch.argmax(logits_z, dim=1)

    # 2. 축별 정답 개수 계산
    correct_x = (preds_x == labels[:, 0]).sum().item()
    correct_y = (preds_y == labels[:, 1]).sum().item()
    correct_z = (preds_z == labels[:, 2]).sum().item()

    # 3. 완전 일치(X, Y, Z 모두 정답) 개수 계산
    exact_match_mask = (preds_x == labels[:, 0]) & \
                       (preds_y == labels[:, 1]) & \
                       (preds_z == labels[:, 2])
    correct_exact = exact_match_mask.sum().item()

    return correct_x, correct_y, correct_z, correct_exact