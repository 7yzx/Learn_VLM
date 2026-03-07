from datasets import load_dataset
from byaldi import RAGMultiModalModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain_deepseek import ChatDeepSeek
from tqdm.auto import tqdm
import pandas as pd
import torch, json, os, gc
from rerankers import Reranker
from utils.utils import find_files,pdf_folder_to_images,clear_memory,save_images_to_local_wo_resize,save_images_to_local,load_png_images,images_to_base64,get_grouped_images,process_ranker_results

MODEL_PATH = "/archive/share/cql/LLM-FoR-ALL/mini_vlm/models/qwen2.5vl"
PDF_PATH = "/archive/share/cql/LLM-FoR-ALL/mini_vlm/data/pdf"
RERANK_MODEL_PATH = "/archive/share/cql/LLM-FoR-ALL/mini_vlm/models/MonoQwen2-VL-v0.1"
RETRIEVAL_MODEL_PATH = "/archive/share/cql/LLM-FoR-ALL/mini_vlm/models/colqwen2-v1.0"
ORIGIN_DATA_PATH = "/archive/share/cql/LLM-FoR-ALL/mini_vlm/data/mrag/pdfvqa"
DATA_PATH = "/archive/share/cql/LLM-FoR-ALL/mini_vlm/data/mrag"
LOG_PATH = "/archive/share/cql/LLM-FoR-ALL/mini_vlm/logs"
IMAGE_PATH = "/archive/share/cql/LLM-FoR-ALL/mini_vlm/data/mrag/images"
TMP_PATH = "/archive/share/cql/aaa/tmp"
ANS_PATH = os.path.join(LOG_PATH, "eval.json")
DS_API_KEY = "your_key"
SUBSET = 50
PREPROCESS = 1
INPUT_PDF = -1

if PREPROCESS>0:
    directories = ['data']
    data_files = find_files(directories,ORIGIN_DATA_PATH)
    dataset = load_dataset("parquet", data_files=data_files, split='train', cache_dir=TMP_PATH) 
    if SUBSET > 0:
        dataset = dataset.select(range(SUBSET))
    save_images_to_local_wo_resize(dataset,'page',IMAGE_PATH)
    directories = ['mypdfqa']
    data_files = find_files(directories,DATA_PATH)
    dataset = load_dataset("parquet", data_files=data_files, split='train', cache_dir=TMP_PATH) 
    if SUBSET > 0:
        dataset = dataset.select(range(SUBSET))
if INPUT_PDF>0:
    pdf_folder_to_images(input_folder=PDF_PATH)

all_images = load_png_images(IMAGE_PATH)

retrieval_model = RAGMultiModalModel.from_pretrained(RETRIEVAL_MODEL_PATH)
retrieval_model.index(input_path=IMAGE_PATH, index_name="paper_index", store_collection_with_index=False, overwrite=True)

# import IPython;IPython.embed();
reranker_model = Reranker(RERANK_MODEL_PATH, device="cuda")

vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16
)
vl_model.eval()
min_pixels = 224 * 224
max_pixels = 448 * 448
vl_model_processor = AutoProcessor.from_pretrained(
    MODEL_PATH, min_pixels=min_pixels, max_pixels=max_pixels
)

clear_memory()

def answer_with_mrag(text_query:str,
                     retrival_top_k:int = 3,
                     reranker_top_k:int = 1,
                     max_new_tokens:int = 500):
    results = retrieval_model.search(text_query, k=retrival_top_k)
    grouped_images = get_grouped_images(results, all_images)
    base64_list = images_to_base64(grouped_images)
    results_rank = reranker_model.rank(text_query, base64_list)
    grouped_images_rank = process_ranker_results(results_rank, grouped_images, top_k=reranker_top_k)

    chat_template = [
        {
            "role": "user",
            "content": [{"type": "image", "image": image} for image in grouped_images_rank]
            + [{"type": "text", "text": text_query}],
        }
    ]
    text = vl_model_processor.apply_chat_template(chat_template, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(chat_template)
    inputs = vl_model_processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = vl_model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]

    output_text = vl_model_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    response = output_text[0]
    return response,results,results_rank

# text_query = "介绍colpali"
# print(answer_with_mrag(text_query))

def gen_answers(dataset, output_json=ANS_PATH):
    list_answers = []
    for image_id, item in enumerate(tqdm(dataset, desc="Generating")):
        questions = item["questions"]
        answers = item["answers"]
        for question,answer in zip(questions,answers):
            try:
                gen_ans, retrieval_hits, ranking_scores = answer_with_mrag(question)
            except torch.cuda.OutOfMemoryError:
                clear_memory(); gc.collect()
                gen_ans = "[ERROR] OOM"
            list_answers.append(
                {
                    "image": f"image_{image_id}.png",
                    "question": question,
                    "true_answer": answer,
                    "generated_answer": gen_ans,
                    # "retrieval_results": retrieval_hits,
                    # "rerank_scores": ranking_scores,
                }
            )
            with open(output_json, "w") as f:
                json.dump(list_answers, f, ensure_ascii=False, indent=2)
    return list_answers

EVALUATION_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}\"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
[Is the response correct, accurate, and factual based on the reference answer?]
Score 1: The response is completely incorrect, inaccurate, and/or not factual.
Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
Score 3: The response is somewhat correct, accurate, and/or factual.
Score 4: The response is mostly correct, accurate, and factual.
Score 5: The response is completely correct, accurate, and factual.

###Feedback:"""

evaluation_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="You are a fair evaluator language model."),
        HumanMessagePromptTemplate.from_template(EVALUATION_PROMPT),
    ]
)

eval_chat_model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    api_key=DS_API_KEY
)

def evaluate_answers(
    answer_path: str,
    eval_chat_model,
    evaluation_prompt_template,
    field_prefix="deepseek-chat"
):
    answers = json.load(open(answer_path))
    for ex in tqdm(answers, desc="Evaluating"):
        if f"eval_score_{field_prefix}" in ex:
            continue 

        eval_prompt = evaluation_prompt_template.format_messages(
            instruction=ex["question"],
            response=ex["generated_answer"],
            reference_answer=ex["true_answer"],
        )
        eval_result = eval_chat_model.invoke(eval_prompt)
        try:
            feedback, score = [s.strip() for s in eval_result.content.split("[RESULT]")]
            score = int(score)
        except Exception as e:
            feedback, score = f"ParseError: {e}\n{eval_result.content}", -1

        ex[f"eval_score_{field_prefix}"] = score
        ex[f"eval_feedback_{field_prefix}"] = feedback

        with open(answer_path, "w") as f:
            json.dump(answers, f, ensure_ascii=False, indent=2)
            
if __name__ == '__main__':
    gen_answers(dataset)
    evaluate_answers(ANS_PATH,eval_chat_model,evaluation_prompt_template)
    result = pd.DataFrame(json.load(open(ANS_PATH, "r")))
    result["eval_score_gpt4"] = result["eval_score_gpt4"].apply(lambda x: float(x) if isinstance(x, int) else 1)
    result["eval_score_gpt4"] = (result["eval_score_gpt4"] - 1) / 4.0
    print(result["eval_score_gpt4"].mean())
    