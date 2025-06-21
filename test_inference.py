import torch
import chromadb
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams
from client import Client
from prompt_gen.inference_prompt_generator import InferencePromptGenerator

def main():
    if not torch.cuda.is_available():
        print("Fatal: NVIDIA GPU not found. This script requires a CUDA-enabled GPU.")
        return

    title = "A Study of JNJ-75276617 in Participants With Relapsed or Refractory Multiple Myeloma"
    description = "The purpose of this study is to characterize the safety and tolerability of JNJ-75276617 and to identify the recommended Phase 2 dose(s) (RP2D[s]) and schedule for JNJ-75276617 monotherapy and JNJ-75276617 in combination with daratumumab and dexamethasone."

    print("--- Initializing Models and Database ---")
    print(f"Title: {title}")
    
    llm = LLM(
        model="biodatlab/ec-raft",
        tensor_parallel_size=1,
        max_model_len=25000,
        dtype="float16"
    )
    embed_model = SentenceTransformer("malteos/scincl")
    collection = chromadb.PersistentClient(path="./clinical_trials_chroma_all").get_collection("clinical_trials_studies")
    
    client = Client(collection.client, embed_model, collection)
    inference_generator = InferencePromptGenerator(client)
    sampling_params = SamplingParams(max_tokens=4096, min_p=0.03, temperature=0.3)

    print("--- Generating Prompt ---")
    messages = inference_generator.generate_inference_messages(title, description, "user_input", 4)
    formatted_prompt = llm.get_tokenizer().apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        tokenize=False
    )

    print("--- Generating Response ---")
    outputs = llm.generate([formatted_prompt], sampling_params)
    response = outputs[0].outputs[0].text
    
    print("\n--- Eligibility Criteria by EC-RAFT ---")
    print(response)
    print("---------------------------------------")
    print("\n✅ Inference complete.")

if __name__ == "__main__":
    main()