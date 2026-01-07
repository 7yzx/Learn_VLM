from datasets import load_dataset
from byaldi import RAGMultiModalModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from langchain_deepseek import ChatDeepSeek
from tqdm.auto import tqdm
import pandas as pd
from PIL import Image
import torch, json, os, gc
from rerankers import Reranker
from utils.utils import find_files,pdf_folder_to_images,save_images_to_local,load_png_images,vlm_generate,vlm_generate_multi
from utils.templates import QA_generation_prompt,question_groundedness_critique_prompt,question_standalone_critique_prompt

MODEL_PATH = "/archive/share/cql/LLM-FoR-ALL/mini_vlm/models/qwen2.5vl"
DATA_PATH = "/archive/share/cql/LLM-FoR-ALL/mini_vlm/data/mrag/pdfvqa"
LOG_PATH = "/archive/share/cql/LLM-FoR-ALL/mini_vlm/logs"
IMAGE_PATH = "/archive/share/cql/LLM-FoR-ALL/mini_vlm/data/mrag/images"
TMP_PATH = "/archive/share/cql/aaa/tmp"
ANS_PATH = os.path.join(LOG_PATH, "eval.json")
DS_API_KEY = "sk-a30bad8dd8e84a3793fad548613df9a3"
SUBSET = 50
PREPROCESS = 1
INPUT_PDF = -1

if PREPROCESS>0:
    directories = ['data']
    data_files = find_files(directories,DATA_PATH)
    dataset = load_dataset("parquet", data_files=data_files, split='train', cache_dir=TMP_PATH) 
    if SUBSET > 0:
        dataset = dataset.select(range(SUBSET))
    save_images_to_local(dataset,'page',IMAGE_PATH)
if INPUT_PDF>0:
    pdf_folder_to_images(input_folder=PDF_PATH)

all_images = load_png_images(IMAGE_PATH)

vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16
)
vl_model.eval()
min_pixels = 224 * 224
max_pixels = 448 * 448
processor = AutoProcessor.from_pretrained(
    MODEL_PATH, min_pixels=min_pixels, max_pixels=max_pixels
)   
# print(vlm_generate_multi("描述这张图片","/archive/share/cql/LLM-FoR-ALL/mini_vlm/data/mrag/images/image_0.png"))

questions_final = [None] * len(dataset)
answers_final   = [None] * len(dataset)

for image_id, image_data in tqdm(enumerate(dataset), total=len(dataset), desc="preprocessing images"):
    image = image_data['page']
    if isinstance(image, str):
        image = Image.open(image)
    output_QA_couples = vlm_generate_multi(vl_model=vl_model,processor=processor,prompt=QA_generation_prompt,img_path=None,image=image,n=7)
    candidates = []
    for output_QA_couple in output_QA_couples:
        try:
            question = output_QA_couple.split("Factoid question: ")[-1].split("Answer: ")[0]
            answer = output_QA_couple.split("Answer: ")[-1]
            candidates.append((question,answer))
        except:
            continue
    questions,answers = [],[]
    for question,answer in candidates:
        evaluations = {
            "groundedness": vlm_generate(
                vl_model=vl_model,
                processor=processor,
                prompt=question_groundedness_critique_prompt.format(question=question),
                image=image
            )[0],
            "standalone": vlm_generate(
                vl_model=vl_model,
                processor=processor,
                prompt=question_standalone_critique_prompt.format(question=question),
                image=image
            )[0],
        }
        # print(evaluations)
        try:
            scores = []
            for criterion, evaluation in evaluations.items():
                score, eval = (
                    int(evaluation.split("Total rating: ")[-1].strip()),
                    evaluation.split("Total rating: ")[-2].split("Evaluation: ")[1],
                )
                scores.append(score)
        except Exception as e:
            continue
        # print("score:",sum(scores)/len(scores))
        if sum(scores)/len(scores)>=4:
            questions.append(question)
            answers.append(answer)
            
    questions_final[image_id] = questions
    answers_final[image_id] = answers
    # print(len(answers))
    
dataset = dataset.remove_columns(
    [c for c in ["questions", "answers"] if c in dataset.column_names]
)
dataset = dataset.add_column("questions", questions_final)
dataset = dataset.add_column("answers",   answers_final)
df = dataset.to_pandas()
output_path = "/archive/share/cql/LLM-FoR-ALL/mini_vlm/data/mrag/mypdfqa/pdf_qa.parquet"
df.to_parquet(output_path, index=False)
    
        