import base64
import io
import requests
from openai import OpenAI
from PIL import Image

# Configure OpenAI client
client = OpenAI(
    api_key="https://t.me/evilfreelancer",
    base_url="https://api.rpa.icu"
)

# Step 1: Download random cat image
resp = requests.get("https://cataas.com/cat", timeout=30)
resp.raise_for_status()

# Step 2: Open image via PIL
img = Image.open(io.BytesIO(resp.content)).convert("RGB")

# Step 3: Convert to base64 Data URL
buf = io.BytesIO()
img.save(buf, format="JPEG")
b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
data_url = f"data:image/jpeg;base64,{b64_str}"

# Step 4: Send to embeddings API with modality=image
resp_img = client.embeddings.create(
    model="jina-clip-v2",
    input=[data_url],
    encoding_format="float",
)

# Step 5: Output vector length and first few numbers
vec = resp_img.data[0].embedding
print(f"Embedding length: {len(vec)}")
print("First 8 dims:", vec[:8])
