import base64
import io
import os
from re import U
import re
from typing import List, Tuple, Union
from urllib import response
from byaldi import RAGMultiModalModel
from byaldi.objects import Result
from byaldi.vlms import OpenAI
from PIL import Image

from dotenv import load_dotenv

load_dotenv()

# Optionally, you can specify an `index_root`, which is where it'll save the index. It defaults to ".byaldi/".
RAG = RAGMultiModalModel.from_pretrained(
    "vidore/colqwen2.5-v0.2"
)

# Initialize VLM
vlm = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    model="Qwen2.5-VL-7B-Instruct"
)

RAG.index(
    input_path="./examples/docs/ACL-3.pdf",
    index_name="attention",
    store_collection_with_index=True,
    overwrite=True
)

query = "What are the key contributions of this paper?"

results: Union[List[Result], List[List[Result]]] = RAG.search(query, k=1)

def generate_response(query: str, search_results: List[Result], **kwargs) -> Tuple[str, List[Image.Image]]:
    
    pil_images = []
    for result in search_results:
        image_data = result.base64
        if isinstance(image_data, Image.Image):
            pil_images.append(image_data)
        elif isinstance(image_data, str):
            # Assume it's base64 encoded
            pil_images.append(Image.open(io.BytesIO(base64.b64decode(image_data))))
        elif isinstance(image_data, bytes):
            pil_images.append(Image.open(io.BytesIO(image_data)))
        else:
            raise ValueError(f"Unexpected image type: {type(image_data)}")

    try:
        # Prepare context for VLM
        context = f"Query: {query}\n\nRelevant image information:\n"
        for i, result in enumerate(search_results, 1):
            context += f"Image {i}: From document '{result.doc_id}', page {result.page_num}\n"
            if result.metadata is not None:
                context += f"Metadata: {result.metadata}\n"
            # if "page_text" in result:
            #     context += f"Page text: {result['page_text'][:500]}...\n\n"

        # Generate response using VLM
        vlm_response = vlm.query(context, pil_images, max_tokens=500)

        return vlm_response, pil_images
    except Exception as e:
        return f"Error generating response: {str(e)}", []


response, images = generate_response(query, results)

print(response)