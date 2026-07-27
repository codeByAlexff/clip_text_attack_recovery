import torch
import clip


def load_model(model_name="ViT-B/32"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device)
    return model, preprocess, device

model, preprocess, device = load_model()

def classify_image(image, labels, top_k=5):
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in labels]).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)


    image_features /= image_features.norm(dim=1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    k = min(top_k, len(labels))
    values, indices = similarity[0].topk(k)

    results = []
    for value, index in zip(values, indices):
        results.append((labels[index], value.item() * 100))
    return results
