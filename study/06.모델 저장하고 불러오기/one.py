import torch
import torchvision.models as models

model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')

model = models.vgg16() # 여기서는 ``weights`` 를 지정하지 않았으므로, 학습되지 않은 모델을 생성합니다.
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

torch.save(model, 'model.pth')
model = torch.load('model.pth')