import os
import sys
import json
import time
from openai import OpenAI

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

input_file_path = os.path.join('ds.jsonl')
output_file_path = os.path.join('ds_score_use_ds.jsonl')

client = OpenAI(
    base_url="...",
    api_key="..."
)

processed_records = []
start_index = 0

if os.path.exists(output_file_path):
    print(f"Detected existing output file: {output_file_path}")
    with open(output_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    processed_records.append(record)
                except json.JSONDecodeError:
                    continue
    
    start_index = len(processed_records)
    print(f"Already processed {start_index} records, will continue from record {start_index + 1}")
else:
    print("No existing output file found, will start from the beginning")

total_processed = start_index
success_count = 0
error_count = 0
current_batch_processed = 0

all_input_records = []
with open(input_file_path, 'r', encoding='utf-8') as fin:
    for line in fin:
        line = line.strip()
        if line:
            try:
                record = json.loads(line)
                all_input_records.append(record)
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}, line content: {line}")
                continue

total_records = len(all_input_records)
remaining_records = total_records - start_index

print(f"Total records: {total_records}")
print(f"Remaining records to process: {remaining_records}")

if remaining_records <= 0:
    print("All records have been processed!")
    exit()

with open(output_file_path, 'a', encoding='utf-8') as fout:
    for i in range(start_index, total_records):
        record = all_input_records[i]
        
        mashup_description = record.get("prompt", "")
        completion = record.get("completion", [])
        target_api_description = record.get("target_api_description", "")
        existing_reason = record.get("reason", "")
        
        if isinstance(completion, list):
            completion_str = ', '.join(completion)
        else:
            completion_str = str(completion)
        
        prompt_message = f"""API Reason Evaluation Task
You are given a mashup description, recommended APIs, their corresponding descriptions, and the reason for the recommendation.
#Instruction
Please evaluate the recommendation by:
1. List all the APIs both in the "Recommended APIs" and in the reason. The API names in the reason are already enclosed in angle brackets (e.g., <API_sendgrid>).
2. For EACH API, evaluate the reason based on three criteria independently and provide:
Criterion 1 Score (0-1): How well the reason aligns with the mashup requirements.
Criterion 2 Score (0-1): How well the reason aligns with the API description.
Criterion 3 Score (0-1): The readability of the reasoning.
A concise justification for each criterion
3. Finally, provide an overall score (0 to 1) calculated as the average of all criteria scores across ALL APIs.
If any API doesn't have a reason or any API is in the reason but not listed in "Recommended APIs", it should be given 0 for all criteria.
#Output Format
Please respond in the following format:
{{
"api_evaluations": [
{{
"api_name": "<API name>",
"criterion_1_score": "<score 0-1>",
"criterion_1_reason": "<reason>",
"criterion_2_score": "<score 0-1>",
"criterion_2_reason": "<reason>",
"criterion_3_score": "<score 0-1>",
"criterion_3_reason": "<reason>"
}}, ...
],
"overall_score": "<score 0-1>"
}}
#Input
Mashup: {mashup_description}
Recommended APIs: {completion_str}
Recommended APIs Description: {target_api_description}
Reason: {existing_reason}"""

        max_retries = 5
        retry_count = 0
        score = None
        score_full = None

        while retry_count < max_retries:
            try:
                print(f"Processing record {i + 1}/{total_records} (batch #{current_batch_processed + 1}), attempt {retry_count + 1}...")

                response = client.chat.completions.create(
                    model="...",
                    messages=[
                        {"role": "user", "content": prompt_message}
                    ]
                )
                score_response = response.choices[0].message.content.strip()

                try:
                    import re
                    json_match = re.search(r'\{[\s\S]*\}', score_response)
                    if json_match:
                        json_str = json_match.group(0)
                        score_data = json.loads(json_str)

                        overall_score = score_data.get("overall_score")
                        if overall_score is not None:
                            overall_score = float(overall_score)
                        if overall_score is not None and 0 <= overall_score <= 1:
                            score = overall_score
                            score_full = score_data
                            success_count += 1
                            print(f"Successfully obtained score: {score}")
                            break
                        else:
                            print(f"Overall score out of range or missing: {overall_score}")
                            retry_count += 1
                    else:
                        print(f"Unable to extract JSON from response: {score_response[:100]}...")
                        retry_count += 1
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Unable to parse response: {e}")
                    print(f"Response: {score_response[:100]}...")
                    retry_count += 1

                if retry_count < max_retries and score is None:
                    print(f"Waiting 2 seconds before retry...")
                    time.sleep(2)

            except Exception as e:
                retry_count += 1
                print(f"Request exception on attempt {retry_count}: {e}")
                if retry_count < max_retries:
                    print(f"Waiting 2 seconds before retry...")
                    time.sleep(2)

        if retry_count >= max_retries or score is None:
            score = -1
            score_full = {"error": "Failed to obtain score after maximum retries"}
            error_count += 1
            print(f"Record processing failed, maximum retries reached")

        record["score"] = score
        record["score_full"] = score_full
        
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()
        
        total_processed += 1
        current_batch_processed += 1
        print(f"Record {i + 1} (Mashup: {mashup_description[:50]}...) processed and written.")
        print(f"Total progress: {total_processed}/{total_records} ({total_processed/total_records*100:.1f}%)")
        
        time.sleep(1)

print(f"\nProcessing completed!")
print(f"Total records: {total_records}")
print(f"Records processed in this session: {current_batch_processed}")
print(f"Successfully processed in this session: {success_count}")
print(f"Error records in this session: {error_count}")
print(f"Total processed records: {total_processed}")
print(f"Output file: {output_file_path}")
