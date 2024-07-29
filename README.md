<h2 align="center"> <a href="https://www.sciencedirect.com/science/article/pii/S0045790624004014">Advancing Vietnamese Visual Question Answering
with Transformer and Convolutional Integration</a></h2>
<h5 align="center"> We would be grateful if you like our project and give us a star ⭐ on GitHub.  </h2>

<h5 align="center">

## Getting Started
It currently includes code and models at 
[here](beit3/HCMUS/).
### Installation
```bash
git clone https://github.com/ngocson1042002/ViVQA.git
cd ViVQA/beit3/HCMUS
pip install salesforce-lavis
pip install torchscale timm underthesea efficientnet_pytorch
pip install --upgrade transformers
```
### Run model
We support our work with Hugging Face model [ngocson2002/vivqa-model](https://huggingface.co/ngocson2002/vivqa-model).
```python
from transformers import AutoModel
from transformers import AutoTokenizer
from processor import Processor
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModel.from_pretrained("ngocson2002/vivqa-model", trust_remote_code=True).to(device)
processor = Processor()

image = Image.open('./ViVQA/demo/1.jpg').convert('RGB')
question = "màu áo của con chó là gì?"

inputs = processor(image, question, return_tensors='pt')
inputs["image"] = inputs["image"].unsqueeze(0)

model.eval()
with torch.no_grad():
    output = model(**inputs)
    logits = output.logits
    idx = logits.argmax(-1).item()

print("Predicted answer:", model.config.id2label[idx]) # prints: màu đỏ
```
## Authors
- Ngoc-Son Nguyen
([nngocson01042002@gmail.com](mailto:nngocson01042002@gmail.com))
- Van-Son Nguyen
([ngvanson1812@gmail.com](mailto:ngvanson1812@gmail.com))
- Tung Le ([lttung@fit.hcmus.edu.vn](mailto:lttung@fit.hcmus.edu.vn))
### Affiliations
- Faculty of Information Technology, University of Science, Ho Chi Minh, Vietnam
- Vietnam National University, Ho Chi Minh, Vietnam
## Contact
Contact Ngoc-Son Nguyen ([nngocson01042002@gmail.com](mailto:nngocson01042002@gmail.com)) if you have any questions.
## Citation
If you find our work useful for your research, please cite using this BibTeX:
```bibtex
@article{NGUYEN2024109474,
    title = {Advancing Vietnamese Visual Question Answering with Transformer and Convolutional Integration},
    journal = {Computers and Electrical Engineering},
    volume = {119},
    pages = {109474},
    year = {2024},
    issn = {0045-7906},
    doi = {https://doi.org/10.1016/j.compeleceng.2024.109474},
    url = {https://www.sciencedirect.com/science/article/pii/S0045790624004014},
    author = {Ngoc Son Nguyen and Van Son Nguyen and Tung Le},
    keywords = {Visual question answering, ViVQA, EfficientNet, BLIP-2, Convolutional},
    abstract = {Visual Question Answering (VQA) has recently emerged as a potential research domain, captivating the interest of many in the field of artificial intelligence and computer vision. Despite the prevalence of approaches in English, there is a notable lack of systems specifically developed for certain languages, particularly Vietnamese. This study aims to bridge this gap by conducting comprehensive experiments on the Vietnamese Visual Question Answering (ViVQA) dataset, demonstrating the effectiveness of our proposed model. In response to community interest, we have developed a model that enhances image representation capabilities, thereby improving overall performance in the ViVQA system. Therefore, we propose AViVQA-TranConI (Advancing Vietnamese Visual Question Answering with Transformer and Convolutional Integration). AViVQA-TranConI integrates the Bootstrapping Language-Image Pre-training with frozen unimodal models (BLIP-2) and the convolutional neural network EfficientNet to extract and process both local and global features from images. This integration leverages the strengths of transformer-based architectures for capturing comprehensive contextual information and convolutional networks for detailed local features. By freezing the parameters of these pre-trained models, we significantly reduce the computational cost and training time, while maintaining high performance. This approach significantly improves image representation and enhances the performance of existing VQA systems. We then leverage a multi-modal fusion module based on a general-purpose multi-modal foundation model (BEiT-3) to fuse the information between visual and textual features. Our experimental findings demonstrate that AViVQA-TranConI surpasses competing baselines, achieving promising performance. This is particularly evident in its accuracy of 71.04% on the test set of the ViVQA dataset, marking a significant advancement in our research area. The code is available at https://github.com/nngocson2002/ViVQA.}
}
```
