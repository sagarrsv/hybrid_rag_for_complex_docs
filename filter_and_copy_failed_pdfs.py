import os
import re
import shutil
from pathlib import Path

# 1. Paths (update these two paths for your Mac)
SOURCE_DIR = Path("/Users/sagar_rsv/Desktop/multimodal_doc_crag/modern_ai_corpus_raw_pdfs")
TARGET_DIR = Path("//Users/sagar_rsv/Desktop/multimodal_doc_crag/modern_ai_corpus_retry_pdfs")

# 2. Your list of 36 document IDs needing visual retry
FAILED_DOC_IDS = [
    "2005.14165_Language_Models_are_FewShot_Learners",
    "2007.07399_Bringing_the_People_Back_In_Contesting_Bench",
    "2010.11929_An_Image_is_Worth_16x16_Words_Transformers_f",
    "2101.03961_Switch_Transformers_Scaling_to_Trillion_Para",
    "2103.00020_Learning_Transferable_Visual_Models_From_Natu",
    "2104.14294_Emerging_Properties_in_SelfSupervised_Vision",
    "2106.09685_LoRA_LowRank_Adaptation_of_Large_Language_M",
    "2201.11903_ChainofThought_Prompting_Elicits_Reasoning_",
    "2203.02155_Training_language_models_to_follow_instructio",
    "2203.15556_Training_ComputeOptimal_Large_Language_Model",
    "2204.05862_Training_a_Helpful_and_Harmless_Assistant_wit",
    "2204.14198_Flamingo_a_Visual_Language_Model_for_FewSho",
    "2211.05100_BLOOM_A_176BParameter_OpenAccess_Multiling",
    "2304.02643_Segment_Anything",
    "2304.07193_DINOv2_Learning_Robust_Visual_Features_witho",
    "2304.08485_Visual_Instruction_Tuning",
    "2305.18290_Direct_Preference_Optimization_Your_Language",
    "2305.20050_Lets_Verify_Step_by_Step",
    "2307.09288_Llama_2_Open_Foundation_and_FineTuned_Chat_",
    "2308.09687_Graph_of_Thoughts_Solving_Elaborate_Problems",
    "2309.16609_Qwen_Technical_Report",
    "2310.11511_SelfRAG_Learning_to_Retrieve_Generate_and",
    "2310.12469_Entropy_and_de_Haasvan_Alphen_oscillations_o",
    "2401.06066_DeepSeekMoE_Towards_Ultimate_Expert_Speciali",
    "2401.10774_Medusa_Simple_LLM_Inference_Acceleration_Fra",
    "2402.03300_DeepSeekMath_Pushing_the_Limits_of_Mathemati",
    "2404.08471_Revisiting_Feature_Prediction_for_Learning_Vi",
    "2405.01168_Remote_Nucleation_and_Stationary_Domain_Walls",
    "2405.21060_Transformers_are_SSMs_Generalized_Models_and",
    "2407.07726_PaliGemma_A_versatile_3B_VLM_for_transfer",
    "2407.21783_The_Llama_3_Herd_of_Models",
    "2408.03314_Scaling_LLM_TestTime_Compute_Optimally_can_b",
    "2409.12191_Qwen2VL_Enhancing_VisionLanguage_Models_P",
    "2412.03555_PaliGemma_2_A_Family_of_Versatile_VLMs_for_T",
    "2412.05271_Expanding_Performance_Boundaries_of_OpenSour",
    "2501.12948_DeepSeekR1_Incentivizing_Reasoning_Capabili",
]

TARGET_DIR.mkdir(parents=True, exist_ok=True)

# 3. Cache source files
source_files = list(SOURCE_DIR.glob("*.pdf"))
print(f"Scanning {len(source_files)} PDFs in: {SOURCE_DIR}")

copied_count = 0
not_found = []

for full_id in FAILED_DOC_IDS:
    # Extract the numeric arxiv ID prefix (e.g. '2005.14165')
    match = re.match(r"^(\d{4}\.\d{4,5})", full_id)
    arxiv_prefix = match.group(1) if match else full_id

    # Locate the matching PDF in the source directory
    matched_file = None
    for file_path in source_files:
        if file_path.name.startswith(arxiv_prefix):
            matched_file = file_path
            break

    if matched_file:
        dest_file = TARGET_DIR / matched_file.name
        shutil.copy2(matched_file, dest_file)
        print(f"Copied: {matched_file.name}")
        copied_count += 1
    else:
        not_found.append(full_id)

print("\n" + "=" * 50)
print(f"Successfully copied: {copied_count}/{len(FAILED_DOC_IDS)} PDFs")
if not_found:
    print(f"Warning: {len(not_found)} files could not be located:")
    for f in not_found:
        print(f" - {f}")
print("=" * 50)