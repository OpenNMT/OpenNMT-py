# This code is based on the following source:
# https://github.com/hobodrifterdavid/nllb-docker-rest

from typing import List
import ctranslate2
import transformers
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
import uvicorn
from asyncio.locks import Lock

app = FastAPI()

# Initialize the translator and tokenizer outside of the function to reuse and avoid overhead.
src_lang = "eng_Latn"
tgt_lang = "zho_Hans"
translator = ctranslate2.Translator("/root/autodl-tmp/nllb-200-3.3B-converted", device="cuda") # If multi gpus , device_index = [0,1,2,3] will be useful.
tokenizer = transformers.AutoTokenizer.from_pretrained("/root/autodl-tmp/ct2fast-nllb-200-3.3B", src_lang=src_lang)

# Define the request and response models for the FastAPI endpoint.
class TranslationRequest(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    translations: List[str]

# Initialize a queue for batching translation requests and a dictionary for results.
request_queue = asyncio.Queue()
results = {}
MAX_BATCH_SIZE = 100
TIMEOUT = 0.1  # Time to wait for accumulating enough batch items.
batch_ready_event = asyncio.Event()
results_lock = Lock()  # Lock to synchronize access to the results dictionary.

async def batch_processor():
    """Asynchronously process translation requests in batches."""
    while True:
        try:
            await asyncio.wait_for(batch_ready_event.wait(), timeout=TIMEOUT)
        except asyncio.TimeoutError:
            pass

        # Accumulate translation requests for batching.
        batched_items = []
        identifiers = []
        while not request_queue.empty() and len(batched_items) < MAX_BATCH_SIZE:
            uid, text = await request_queue.get()
            batched_items.append(text)
            identifiers.append(uid)

        # If there are items to translate, process them.
        if batched_items:
            print(f"Translating a batch of {len(batched_items)} items.")
            try:
                translations = translate_batch(batched_items)
                async with results_lock:
                    for uid, translation in zip(identifiers, translations):
                        if uid in results:
                            results[uid]["translation"] = translation
                            event = results[uid]["event"]
                            event.set()
            except Exception as e:
                # Handle translation errors.
                print(f"Error during translation: {e}")

        # Reset the event to wait for the next batch.
        batch_ready_event.clear()

@app.on_event("startup")
async def startup_event():
    """On server startup, initialize the batch processor."""
    asyncio.create_task(batch_processor())

@app.post("/translate", response_model=TranslationResponse)
async def translate_endpoint(request: TranslationRequest):
    """Endpoint to handle translation requests."""
    result_event = asyncio.Event()
    unique_id = str(id(result_event))
    async with results_lock:
        results[unique_id] = {"event": result_event, "translation": None}
    await request_queue.put((unique_id, request.text))

    # Trigger the batch_ready_event if the queue size reaches the defined threshold.
    if request_queue.qsize() >= MAX_BATCH_SIZE:
        batch_ready_event.set()

    # Wait for the translation result.
    await result_event.wait()
    async with results_lock:
        translation = results.pop(unique_id, {}).get("translation", "")
    return {"translations": [translation]}

def translate_batch(texts: List[str]) -> List[str]:
    """Translate a batch of texts using the pre-initialized translator and tokenizer."""
    # Tokenize source texts.
    sources = [tokenizer.convert_ids_to_tokens(tokenizer.encode(text)) for text in texts]
    # Set target language prefixes.
    target_prefixes = [[tgt_lang] for _ in texts]
    # Translate in batches.
    batch_results = translator.translate_batch(sources, target_prefix=target_prefixes, max_batch_size=MAX_BATCH_SIZE)
    translations = []
    for result in batch_results:
        target = result.hypotheses[0][1:]
        translations.append(tokenizer.decode(tokenizer.convert_tokens_to_ids(target)))
    return translations

if __name__ == '__main__':
    # Start the FastAPI server.
    uvicorn.run(app=app, host='0.0.0.0', port=8000)
