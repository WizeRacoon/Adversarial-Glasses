import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from insightface.app import FaceAnalysis

def load_resnet(device: torch.device) -> InceptionResnetV1:
    """Loads a pre-trained, frozen InceptionResnetV1 model."""
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model

def load_mtcnn(device: torch.device) -> MTCNN:
    """Loads MTCNN for dynamic face alignment and cropping."""
    return MTCNN(image_size=224, margin=20, keep_all=False, post_process=False, device=device)

def load_arcface(device: torch.device) -> FaceAnalysis:
    """Loads the ArcFace buffalo_l model for black-box transferability evaluation."""
    providers = ['CUDAExecutionProvider'] if device.type == 'cuda' else ['CPUExecutionProvider']
    app = FaceAnalysis(name='buffalo_l', providers=providers)
    
    # ctx_id=0 uses the first GPU, -1 forces CPU
    ctx_id = 0 if device.type == 'cuda' else -1
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app