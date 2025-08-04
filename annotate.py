import os
import sys
import json
import time
from openai import OpenAI


curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


input_file_path = os.path.join(rootPath, 'preprocess', 'train_data.jsonl')
output_file_path = os.path.join(rootPath, 'preprocess', 'train_data_reason_ds.jsonl')


client = OpenAI(
    base_url="...",
    api_key="..."
)


total_processed = 0
success_count = 0
error_count = 0


with open(input_file_path, 'r', encoding='utf-8') as fin, \
     open(output_file_path, 'w', encoding='utf-8') as fout:
    
    for line in fin:
        line = line.strip()
        if not line:
            continue
        
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"JSON error: {e} ,line: {line}")
            continue
        
    
        mashup_description = record.get("prompt", "")
        completion = record.get("completion", [])
        target_api_description = record.get("target_api_description", "")
        

        if isinstance(completion, list):
            completion_str = ', '.join(completion)
        else:
            completion_str = str(completion)
        

        prompt_message = f"""API Recommendation Reasoning Task
Reason in English, concisely and accurately explain why the mashup uses each target API, based on the descriptions of the mashup and the target APIs.
When you mention an API name, wrap it with three asterisks exactly as given in Target APIs,
for example: ***sendgrid***.

Mashup: {mashup_description}
Target APIs: {completion_str}
Target APIs Descriptions: {target_api_description}

Reason:
"""
        
    #    max 5
        max_retries = 5
        retry_count = 0
        reason = None
        
        while retry_count < max_retries:
            try:
                print(f"record {total_processed + 1},try{retry_count + 1} 次...")
                
                response = client.chat.completions.create(
                    model="",
                    messages=[
                        {"role": "user", "content": prompt_message}
                    ]
                )
                reason = response.choices[0].message.content
                

                if reason and not reason.startswith("Error:"):
                    success_count += 1
                    print(f" reason: {reason[:100]}...")
                    break
                else:
                    print(f"{reason[:50] if reason else 'None'}...")
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"retry...")
                        time.sleep(2)
                
            except Exception as e:
                retry_count += 1
                print(f"{retry_count} error: {e}")
                if retry_count < max_retries:
                    print(f"retry...")
                    time.sleep(2)
        
        
        if retry_count >= max_retries or not reason or reason.startswith("Error:"):
            reason = "Error: Failed to generate reason after multiple retries"
            error_count += 1
            print(f"max entry")
        
        # After running this code, the annotated dataset still needs to be manually cleaned!!!”

        if reason:
            reason = reason.strip().replace('\n', ' ').replace('\r', ' ')
        

        record["reason"] = reason
        

        json_line = json.dumps(record, ensure_ascii=False, separators=(',', ':'))
        fout.write(json_line + "\n")
        fout.flush()  
        
        total_processed += 1
        print(f"record {total_processed}(Mashup:{mashup_description[:50]}...)written.")
        

        time.sleep(1)

print(f"\ndone!")
print(f"total: {total_processed}")
print(f"success: {success_count}")
print(f"error: {error_count}")
print(f"output_file: {output_file_path}")

