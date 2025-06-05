"""
This file is the BFCL-V2, both composite and live-only.

There are three global variables you need to set:
- DIR: the score directory
- SUFFIX: the suffix of the output file, it will be saved as error_type_{TYPE}_{SUFFIX}.csv
- LIVE_ONLY: whether to only include live models (True for live only, False for composite of live and non-live)

This will save into ./analysis/error_type_{TYPE}_{SUFFIX}.csv

This script deprecates error_type_helper.py (v1)
"""

import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

DIR = "score_Sep30"
SUFFIX = "BFCL-Live,ICLRSep30"
LIVE_ONLY = True

json_files = [
    f for f in glob.glob(DIR + "/**/*.json", recursive=True) if "multi_turn" not in f
]

rows = []
error_mapping = {
    "type_error:nested": "Type Error: Composite Type Mismatch",
    "type_error:simple": "Type Error: Primitive Type Mismatch",
    "value_error:dict_items": "Value Error: Invalid Dictionary Items",
    "value_error:dict_key": "Value Error: Invalid Dictionary Key",
    "value_error:dict_value": "Value Error: Invalid Dictionary Value",
    "value_error:string": "Value Error: Wrong String Value",
    "value_error:list_dict_count": "Value Error: Incorrect List/Dictionary Count",
    "value_error:others": "Value Error: Wrong int/float/bool/variable Value",
    "value_error:exec_result_count": "Executable Function Error (REST): Incorrect Execution Result List Length",
    "dict_checker:unclear": "Value Error: Dictionary Misc",
    "list_dict_checker:unclear": "Value Error: List Dictionary Misc",
    "simple_function_checker:wrong_func_name": "Simple Function Error: Incorrect Function Name",
    "simple_function_checker:missing_required": "Simple Function Error: Missing Required Parameter",
    "simple_function_checker:unexpected_param": "Simple Function Error: Unexpected Parameter Encountered",
    "simple_function_checker:missing_optional": "Simple Function Error: Missing Optional Parameter",
    "simple_function_checker:unclear": "Simple Function Error: Misc",
    "simple_function_checker:wrong_count": "Simple Function Error: Unexpected Argument Count",
    "type_error:java": "Type Error: Java Type Mismatch",
    "type_error:js": "Type Error: JavaScript Type Mismatch",
    "parallel_function_checker_enforce_order:wrong_count": "Parallel Function Error: Unexpected Function Count",
    "parallel_function_checker_enforce_order:cannot_find_description": "Parallel Function Error: Missing Function Description",
    "parallel_function_checker_no_order:wrong_count": "Parallel Function Error: Unexpected Function Count",
    "parallel_function_checker_no_order:cannot_find_description": "Parallel Function Error: Missing Function Description",
    "parallel_function_checker_no_order:cannot_find_match": "Parallel Function Error: No Match Found",
    "executable_checker:wrong_result_type": "Executable Function Error (Non REST): Incorrect Result Type",
    "executable_checker:wrong_result_type:dict_length": "Executable Function Error (Non REST): Incorrect Dictionary Length",
    "executable_checker:wrong_result_type:dict_key_not_found": "Executable Function Error (Non REST): Dictionary Key Not Found",
    "executable_checker:wrong_result_type:dict_extra_key": "Executable Function Error (Non REST): Extra Dictionary Key",
    "executable_checker:wrong_result_type:list_length": "Executable Function Error (Non REST): Incorrect List Length",
    "executable_checker:unclear": "Executable Function Error (Non REST): Misc",
    "executable_checker:execution_error": "Executable Function Error (Non REST): Runtime Execution Error",
    "executable_checker:wrong_result": "Executable Function Error (Non REST): Incorrect Execution Result",
    "executable_checker:wrong_result_real_time": "Executable Function Error (Non REST): Incorrect Execution Result for Real Time",
    "executable_checker:cannot_find_match": "Executable Function Error (Non REST): No Match Found",
    "parallel_function_checker_no_order:wrong_count": "Parallel Function Error: Unexpected Function Count",
    "parallel_function_checker_no_order:cannot_find_description": "Parallel Function Error: Missing Function Description",
    "value_error:exec_result_rest_count": "Value Error: Execution Result Count",
    "value_error:list/tuple": "Value Error: Invalid List/Tuple Format",
    "simple_exec_checker:wrong_count": "Executable Function Error (Non REST): Incorrect Function Count for Simple Execution",
    "executable_checker_rest:execution_error": "Executable Function Error (REST): Runtime Execution Error",
    "executable_checker_rest:wrong_key": "Executable Function Error (REST): Incorrect Response Key",
    "executable_checker_rest:wrong_type": "Executable Function Error (REST): Incorrect Response Type",
    "executable_checker_rest:response_format_error": "Executable Function Error (REST): Response Format Error",
    "executable_checker_rest:wrong_status_code": "Executable Function Error (REST): Incorrect HTTP Status Code",
    "executable_checker_rest:cannot_get_status_code": "Executable Function Error (REST): Cannot Get HTTP Status Code",
    "executable_decoder:decoder_failed": "Function Parsing Error: Decoding Failure",
    "executable_decoder:rest_wrong_output_format": "Function Parsing Error: Incorrect REST Parsing Output Format",
    "executable_decoder:wrong_output_format": "Function Parsing Error: Incorrect Parsing Output Format",
    "ast_decoder:decoder_failed": "Function Parsing Error: Decoding Failure",
    "ast_decoder:decoder_wrong_output_format": "Function Parsing Error: Incorrect Decoder Output Format",
    "irrelevance_error:decoder_success": "Irrelevance Error: Succeeded with Irrelevant Func Doc",
    "relevance_error:decoder_failed": "Relevance Error: Decoding Failure",
    "multiple_function_checker:wrong_count": "Multiple Function Error: Incorrect Function Count",
}

test_category_mapping = {
    "multiple_function": "ast_multiple",
    "parallel_multiple_function": "ast_parallel_multiple",
    "parallel_function": "ast_parallel",
    "simple": "ast_simple",
    "java": "ast_java",
    "javascript": "ast_js",
    "executable_parallel_multiple_function": "exec_parallel_multiple",
    "executable_multiple_function": "exec_multiple",
    "executable_simple": "exec_simple",
    "executable_parallel_function": "exec_parallel",
    "rest": "exec_rest",
}


MODEL_METADATA_MAPPING = {
    "o1-preview-2024-09-12": [
        "o1-preview-2024-09-12 (Prompt)",
        "https://openai.com/index/introducing-openai-o1-preview/",
        "OpenAI",
        "Proprietary",
    ],
    "o1-mini-2024-09-12": [
        "o1-mini-2024-09-12 (Prompt)",
        "https://openai.com/index/openai-o1-mini-advancing-cost-efficient-reasoning/",
        "OpenAI",
        "Proprietary",
    ],
    "gpt-4o-2024-08-06": [
        "GPT-4o-2024-08-06 (Prompt)",
        "https://openai.com/index/hello-gpt-4o/",
        "OpenAI",
        "Proprietary",
    ],
    "gpt-4o-2024-08-06-FC": [
        "GPT-4o-2024-08-06 (FC)",
        "https://openai.com/index/hello-gpt-4o/",
        "OpenAI",
        "Proprietary",
    ],
    "gpt-4o-2024-05-13-FC": [
        "GPT-4o-2024-05-13 (FC)",
        "https://openai.com/index/hello-gpt-4o/",
        "OpenAI",
        "Proprietary",
    ],
    "gpt-4o-2024-05-13": [
        "GPT-4o-2024-05-13 (Prompt)",
        "https://openai.com/index/hello-gpt-4o/",
        "OpenAI",
        "Proprietary",
    ],
    "gpt-4o-mini-2024-07-18": [
        "GPT-4o-mini-2024-07-18 (Prompt)",
        "https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/",
        "OpenAI",
        "Proprietary",
    ],
    "gpt-4o-mini-2024-07-18-FC": [
        "GPT-4o-mini-2024-07-18 (FC)",
        "https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/",
        "OpenAI",
        "Proprietary",
    ],
    "gpt-4-1106-preview-FC": [
        "GPT-4-1106-Preview (FC)",
        "https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo",
        "OpenAI",
        "Proprietary",
    ],
    "gpt-4-1106-preview": [
        "GPT-4-1106-Preview (Prompt)",
        "https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo",
        "OpenAI",
        "Proprietary",
    ],
    "gpt-4-0125-preview-FC": [
        "GPT-4-0125-Preview (FC)",
        "https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo",
        "OpenAI",
        "Proprietary",
    ],
    "gpt-4-0125-preview": [
        "GPT-4-0125-Preview (Prompt)",
        "https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo",
        "OpenAI",
        "Proprietary",
    ],
    "gpt-4-turbo-2024-04-09-FC": [
        "GPT-4-turbo-2024-04-09 (FC)",
        "https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo",
        "OpenAI",
        "Proprietary",
    ],
    "gpt-4-turbo-2024-04-09": [
        "GPT-4-turbo-2024-04-09 (Prompt)",
        "https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo",
        "OpenAI",
        "Proprietary",
    ],
    "gorilla-openfunctions-v2": [
        "Gorilla-OpenFunctions-v2 (FC)",
        "https://gorilla.cs.berkeley.edu/blogs/7_open_functions_v2.html",
        "Gorilla LLM",
        "Apache 2.0",
    ],
    "claude-3-opus-20240229-FC": [
        "Claude-3-Opus-20240229 (FC tools-2024-04-04)",
        "https://www.anthropic.com/news/claude-3-family",
        "Anthropic",
        "Proprietary",
    ],
    "claude-3-opus-20240229": [
        "Claude-3-Opus-20240229 (Prompt)",
        "https://www.anthropic.com/news/claude-3-family",
        "Anthropic",
        "Proprietary",
    ],
    "open-mistral-nemo-2407": [
        "Open-Mistral-Nemo-2407 (Prompt)",
        "https://mistral.ai/news/mistral-nemo/",
        "Mistral AI",
        "Proprietary",
    ],
    "open-mistral-nemo-2407-FC": [
        "Open-Mistral-Nemo-2407 (FC)",
        "https://mistral.ai/news/mistral-nemo/",
        "Mistral AI",
        "Proprietary",
    ],
    "open-mixtral-8x22b": [
        "Open-Mixtral-8x22b (Prompt)",
        "https://mistral.ai/news/mixtral-8x22b/",
        "Mistral AI",
        "Proprietary",
    ],
    "open-mixtral-8x22b-FC": [
        "Open-Mixtral-8x22b (FC)",
        "https://mistral.ai/news/mixtral-8x22b/",
        "Mistral AI",
        "Proprietary",
    ],
    "open-mixtral-8x7b": [
        "Open-Mixtral-8x7b (Prompt)",
        "https://mistral.ai/news/mixtral-of-experts/",
        "Mistral AI",
        "Proprietary",
    ],
    "mistral-medium-2312": [
        "Mistral-Medium-2312 (Prompt)",
        "https://docs.mistral.ai/guides/model-selection/",
        "Mistral AI",
        "Proprietary",
    ],
    "mistral-small-2402": [
        "Mistral-Small-2402 (Prompt)",
        "https://docs.mistral.ai/guides/model-selection/",
        "Mistral AI",
        "Proprietary",
    ],
    "mistral-large-2407": [
        "mistral-large-2407 (Prompt)",
        "https://mistral.ai/news/mistral-large-2407/",
        "Mistral AI",
        "Proprietary",
    ],
    "claude-3-sonnet-20240229-FC": [
        "Claude-3-Sonnet-20240229 (FC tools-2024-04-04)",
        "https://www.anthropic.com/news/claude-3-family",
        "Anthropic",
        "Proprietary",
    ],
    "claude-3-sonnet-20240229": [
        "Claude-3-Sonnet-20240229 (Prompt)",
        "https://www.anthropic.com/news/claude-3-family",
        "Anthropic",
        "Proprietary",
    ],
    "claude-3-haiku-20240307-FC": [
        "Claude-3-Haiku-20240307 (FC tools-2024-04-04)",
        "https://www.anthropic.com/news/claude-3-family",
        "Anthropic",
        "Proprietary",
    ],
    "claude-3-haiku-20240307": [
        "Claude-3-Haiku-20240307 (Prompt)",
        "https://www.anthropic.com/news/claude-3-family",
        "Anthropic",
        "Proprietary",
    ],
    "claude-3-5-sonnet-20240620-FC": [
        "Claude-3.5-Sonnet-20240620 (FC)",
        "https://www.anthropic.com/news/claude-3-5-sonnet",
        "Anthropic",
        "Proprietary",
    ],
    "claude-3-5-sonnet-20240620": [
        "Claude-3.5-Sonnet-20240620 (Prompt)",
        "https://www.anthropic.com/news/claude-3-5-sonnet",
        "Anthropic",
        "Proprietary",
    ],
    "gpt-3.5-turbo-0125-FC": [
        "GPT-3.5-Turbo-0125 (FC)",
        "https://platform.openai.com/docs/models/gpt-3-5-turbo",
        "OpenAI",
        "Proprietary",
    ],
    "gpt-3.5-turbo-0125": [
        "GPT-3.5-Turbo-0125 (Prompt)",
        "https://platform.openai.com/docs/models/gpt-3-5-turbo",
        "OpenAI",
        "Proprietary",
    ],
    "meetkai/functionary-small-v3.1-FC": [
        "Functionary-Small-v3.1 (FC)",
        "https://huggingface.co/meetkai/functionary-small-v3.1",
        "MeetKai",
        "MIT",
    ],
    "meetkai/functionary-small-v3.2-FC": [
        "Functionary-Small-v3.2 (FC)",
        "https://huggingface.co/meetkai/functionary-small-v3.2",
        "MeetKai",
        "MIT",
    ],
    "meetkai/functionary-medium-v3.1-FC": [
        "Functionary-Medium-v3.1 (FC)",
        "https://huggingface.co/meetkai/functionary-medium-v3.1",
        "MeetKai",
        "MIT",
    ],
    "claude-2.1": [
        "Claude-2.1 (Prompt)",
        "https://www.anthropic.com/news/claude-2-1",
        "Anthropic",
        "Proprietary",
    ],
    "mistral-tiny-2312": [
        "Mistral-tiny-2312 (Prompt)",
        "https://docs.mistral.ai/guides/model-selection/",
        "Mistral AI",
        "Proprietary",
    ],
    "claude-instant-1.2": [
        "Claude-instant-1.2 (Prompt)",
        "https://www.anthropic.com/news/releasing-claude-instant-1-2",
        "Anthropic",
        "Proprietary",
    ],
    "mistral-small-2402-FC": [
        "Mistral-small-2402 (FC)",
        "https://docs.mistral.ai/guides/model-selection/",
        "Mistral AI",
        "Proprietary",
    ],
    "mistral-large-2407-FC": [
        "mistral-large-2407 (FC)",
        "https://mistral.ai/news/mistral-large-2407/",
        "Mistral AI",
        "Proprietary",
    ],
    "Nexusflow-Raven-v2": [
        "Nexusflow-Raven-v2 (FC)",
        "https://huggingface.co/Nexusflow/NexusRaven-V2-13B",
        "Nexusflow",
        "Apache 2.0",
    ],
    "firefunction-v1-FC": [
        "FireFunction-v1 (FC)",
        "https://huggingface.co/fireworks-ai/firefunction-v1",
        "Fireworks",
        "Apache 2.0",
    ],
    "firefunction-v2-FC": [
        "FireFunction-v2 (FC)",
        "https://huggingface.co/fireworks-ai/firefunction-v2",
        "Fireworks",
        "Apache 2.0",
    ],
    "gemini-1.5-pro-002": [
        "Gemini-1.5-Pro-002 (Prompt)",
        "https://deepmind.google/technologies/gemini/pro/",
        "Google",
        "Proprietary",
    ],
    "gemini-1.5-pro-002-FC": [
        "Gemini-1.5-Pro-002 (FC)",
        "https://deepmind.google/technologies/gemini/pro/",
        "Google",
        "Proprietary",
    ],
    "gemini-1.5-pro-preview-0514-FC": [
        "Gemini-1.5-Pro-Preview-0514 (FC)",
        "https://deepmind.google/technologies/gemini/pro/",
        "Google",
        "Proprietary",
    ],
    "gemini-1.5-pro-preview-0514": [
        "Gemini-1.5-Pro-Preview-0514 (Prompt)",
        "https://deepmind.google/technologies/gemini/pro/",
        "Google",
        "Proprietary",
    ],
    "gemini-1.5-flash-preview-0514-FC": [
        "Gemini-1.5-Flash-Preview-0514 (FC)",
        "https://deepmind.google/technologies/gemini/flash/",
        "Google",
        "Proprietary",
    ],
    "gemini-1.5-flash-preview-0514": [
        "Gemini-1.5-Flash-Preview-0514 (Prompt)",
        "https://deepmind.google/technologies/gemini/flash/",
        "Google",
        "Proprietary",
    ],
    "gemini-1.5-pro-preview-0409-FC": [
        "Gemini-1.5-Pro-Preview-0409 (FC)",
        "https://deepmind.google/technologies/gemini/#introduction",
        "Google",
        "Proprietary",
    ],
    "gemini-1.0-pro-FC": [
        "Gemini-1.0-Pro-001 (FC)",
        "https://deepmind.google/technologies/gemini/#introduction",
        "Google",
        "Proprietary",
    ],
    "gemini-1.0-pro": [
        "Gemini-1.0-Pro-001 (Prompt)",
        "https://deepmind.google/technologies/gemini/#introduction",
        "Google",
        "Proprietary",
    ],
    "gpt-4-0613-FC": [
        "GPT-4-0613 (FC)",
        "https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo",
        "OpenAI",
        "Proprietary",
    ],
    "gpt-4-0613": [
        "GPT-4-0613 (Prompt)",
        "https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo",
        "OpenAI",
        "Proprietary",
    ],
    "deepseek-ai/deepseek-coder-6.7b-instruct": [
        "Deepseek-v1.5 (Prompt)",
        "https://huggingface.co/deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        "Deepseek",
        "Deepseek License",
    ],
    "google/gemma-7b-it": [
        "Gemma-7b-it (Prompt)",
        "https://blog.google/technology/developers/gemma-open-models/",
        "Google",
        "gemma-terms-of-use",
    ],
    "glaiveai/glaive-function-calling-v1": [
        "Glaive-v1 (FC)",
        "https://huggingface.co/glaiveai/glaive-function-calling-v1",
        "Glaive",
        "cc-by-sa-4.0",
    ],
    "databricks-dbrx-instruct": [
        "DBRX-Instruct (Prompt)",
        "https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm",
        "Databricks",
        "Databricks Open Model",
    ],
    "NousResearch/Hermes-2-Pro-Llama-3-8B": [
        "Hermes-2-Pro-Llama-3-8B (FC)",
        "https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B",
        "NousResearch",
        "apache-2.0",
    ],
    "NousResearch/Hermes-2-Pro-Llama-3-70B": [
        "Hermes-2-Pro-Llama-3-70B (FC)",
        "https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-70B",
        "NousResearch",
        "apache-2.0",
    ],
    "NousResearch/Hermes-2-Pro-Mistral-7B": [
        "Hermes-2-Pro-Mistral-7B (FC)",
        "https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B",
        "NousResearch",
        "apache-2.0",
    ],
    "NousResearch/Hermes-2-Theta-Llama-3-8B": [
        "Hermes-2-Theta-Llama-3-8B (FC)",
        "https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B",
        "NousResearch",
        "apache-2.0",
    ],
    "NousResearch/Hermes-2-Theta-Llama-3-70B": [
        "Hermes-2-Theta-Llama-3-70B (FC)",
        "https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-70B",
        "NousResearch",
        "apache-2.0",
    ],
    "meta-llama/Meta-Llama-3-8B-Instruct": [
        "Meta-Llama-3-8B-Instruct (Prompt)",
        "https://llama.meta.com/llama3",
        "Meta",
        "Meta Llama 3 Community",
    ],
    "meta-llama/Meta-Llama-3-70B-Instruct": [
        "Meta-Llama-3-70B-Instruct (Prompt)",
        "https://llama.meta.com/llama3",
        "Meta",
        "Meta Llama 3 Community",
    ],
    "meta-llama/Llama-3.1-8B-Instruct": [
        "Llama-3.1-8B-Instruct (Prompt)",
        "https://llama.meta.com/llama3",
        "Meta",
        "Meta Llama 3 Community",
    ],
    "meta-llama/Llama-3.1-70B-Instruct": [
        "Llama-3.1-70B-Instruct (Prompt)",
        "https://llama.meta.com/llama3",
        "Meta",
        "Meta Llama 3 Community",
    ],
    "meta-llama/Llama-3.2-1B-Instruct": [
        "Llama-3.2-1B-Instruct (Prompt)",
        "https://llama.meta.com/llama3",
        "Meta",
        "Meta Llama 3 Community",
    ],
    "meta-llama/Llama-3.2-3B-Instruct": [
        "Llama-3.2-3B-Instruct (Prompt)",
        "https://llama.meta.com/llama3",
        "Meta",
        "Meta Llama 3 Community",
    ],
    "meta-llama/Llama-3.1-8B-Instruct-FC": [
        "Llama-3.1-8B-Instruct (FC)",
        "https://llama.meta.com/llama3",
        "Meta",
        "Meta Llama 3 Community",
    ],
    "meta-llama/Llama-3.1-70B-Instruct-FC": [
        "Llama-3.1-70B-Instruct (FC)",
        "https://llama.meta.com/llama3",
        "Meta",
        "Meta Llama 3 Community",
    ],
    "meta-llama/Llama-3.2-1B-Instruct-FC": [
        "Llama-3.2-1B-Instruct (FC)",
        "https://llama.meta.com/llama3",
        "Meta",
        "Meta Llama 3 Community",
    ],
    "meta-llama/Llama-3.2-3B-Instruct-FC": [
        "Llama-3.2-3B-Instruct (FC)",
        "https://llama.meta.com/llama3",
        "Meta",
        "Meta Llama 3 Community",
    ],
    "command-r-plus-FC": [
        "Command-R-Plus (FC) (Original)",
        "https://txt.cohere.com/command-r-plus-microsoft-azure",
        "Cohere For AI",
        "cc-by-nc-4.0",
    ],
    "command-r-plus": [
        "Command-R-Plus (Prompt) (Original)",
        "https://txt.cohere.com/command-r-plus-microsoft-azure",
        "Cohere For AI",
        "cc-by-nc-4.0",
    ],
    "command-r-plus-FC-optimized": [
        "Command-R-Plus (FC) (Optimized)",
        "https://txt.cohere.com/command-r-plus-microsoft-azure",
        "Cohere For AI",
        "cc-by-nc-4.0",
    ],
    "command-r-plus-optimized": [
        "Command-R-Plus (Prompt) (Optimized)",
        "https://txt.cohere.com/command-r-plus-microsoft-azure",
        "Cohere For AI",
        "cc-by-nc-4.0",
    ],
    "snowflake/arctic": [
        "Snowflake/snowflake-arctic-instruct (Prompt)",
        "https://huggingface.co/Snowflake/snowflake-arctic-instruct",
        "Snowflake",
        "apache-2.0",
    ],
    "nvidia/nemotron-4-340b-instruct": [
        "Nemotron-4-340b-instruct (Prompt)",
        "https://huggingface.co/nvidia/nemotron-4-340b-instruct",
        "NVIDIA",
        "nvidia-open-model-license",
    ],
    "ibm-granite/granite-20b-functioncalling": [
        "Granite-20b-FunctionCalling (FC)",
        "https://huggingface.co/ibm-granite/granite-20b-functioncalling",
        "IBM",
        "Apache-2.0",
    ],
    "THUDM/glm-4-9b-chat": [
        "GLM-4-9b-Chat (FC)",
        "https://huggingface.co/THUDM/glm-4-9b-chat",
        "THUDM",
        "glm-4",
    ],
    "yi-large-fc": [
        "yi-large (FC)",
        "https://platform.01.ai/",
        "01.AI",
        "Proprietary",
    ],
    "Salesforce/xLAM-1b-fc-r": [
        "xLAM-1b-fc-r (FC)",
        "https://huggingface.co/Salesforce/xLAM-1b-fc-r",
        "Salesforce",
        "cc-by-nc-4.0",
    ],
    "Salesforce/xLAM-7b-fc-r": [
        "xLAM-7b-fc-r (FC)",
        "https://huggingface.co/Salesforce/xLAM-7b-fc-r",
        "Salesforce",
        "cc-by-nc-4.0",
    ],
    "Salesforce/xLAM-7b-r": [
        "xLAM-7b-r (FC)",
        "https://huggingface.co/Salesforce/xLAM-7b-r",
        "Salesforce",
        "cc-by-nc-4.0",
    ],
    "Salesforce/xLAM-8x7b-r": [
        "xLAM-8x7b-r (FC)",
        "https://huggingface.co/Salesforce/xLAM-8x7b-r",
        "Salesforce",
        "cc-by-nc-4.0",
    ],
    "Salesforce/xLAM-8x22b-r": [
        "xLAM-8x22b-r (FC)",
        "https://huggingface.co/Salesforce/xLAM-8x22b-r",
        "Salesforce",
        "cc-by-nc-4.0",
    ],
    "MadeAgents/Hammer-7b": [
        "Hammer-7b (FC)",
        "https://huggingface.co/MadeAgents/Hammer-7b",
        "MadeAgents",
        "cc-by-nc-4.0",
    ],
    "microsoft/Phi-3-mini-4k-instruct": [
        "Phi-3-mini-4k-instruct (Prompt)",
        "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct",
        "Microsoft",
        "MIT",
    ],
    "microsoft/Phi-3-mini-128k-instruct": [
        "Phi-3-mini-128k-instruct (Prompt)",
        "https://huggingface.co/microsoft/Phi-3-mini-128k-instruct",
        "Microsoft",
        "MIT",
    ],
    "microsoft/Phi-3-small-8k-instruct": [
        "Phi-3-small-8k-instruct (Prompt)",
        "https://huggingface.co/microsoft/Phi-3-small-8k-instruct",
        "Microsoft",
        "MIT",
    ],
    "microsoft/Phi-3-small-128k-instruct": [
        "Phi-3-small-128k-instruct (Prompt)",
        "https://huggingface.co/microsoft/Phi-3-small-128k-instruct",
        "Microsoft",
        "MIT",
    ],
    "microsoft/Phi-3-medium-4k-instruct": [
        "Phi-3-medium-4k-instruct (Prompt)",
        "https://huggingface.co/microsoft/Phi-3-medium-4k-instruct",
        "Microsoft",
        "MIT",
    ],
    "microsoft/Phi-3-medium-128k-instruct": [
        "Phi-3-medium-128k-instruct (Prompt)",
        "https://huggingface.co/microsoft/Phi-3-medium-128k-instruct",
        "Microsoft",
        "MIT",
    ],
    "microsoft/Phi-3.5-mini-instruct": [
        "Phi-3.5-mini-instruct (Prompt)",
        "https://huggingface.co/microsoft/Phi-3.5-mini-instruct",
        "Microsoft",
        "MIT",
    ],
}

MODEL_METADATA_MAPPING_DICT = {
    k.replace("/", "_"): v[0] for k, v in MODEL_METADATA_MAPPING.items()
}  # mapping form original name to converted name
MODEL_METADATA_MAPPING_DICT_inverted = {
    value: key for key, value in MODEL_METADATA_MAPPING_DICT.items()
}  # inverted mapping


# Loop through the files and process them
for TYPE in ["executable", "all", "ast"]:
    rows = []
    for file in json_files:
        filename = os.path.basename(file)
        if TYPE == "all":
            pass
        elif TYPE == "executable":
            if not "executable" in filename and not "rest" in filename:
                continue
        elif TYPE == "ast":
            if "executable" in filename or "rest" in filename:
                continue

        with open(file, "r") as f:
            for idx, line in enumerate(f):
                if idx == 0:
                    continue
                data_json = json.loads(line)
                valid = data_json.get("valid")
                if valid == "True" or valid is True:
                    continue
                id = data_json.get("id")
                model_name = data_json.get("model_name")
                test_category = data_json.get("test_category")
                error = data_json.get("error")
                error_type = data_json.get("error_type")
                model_result = data_json.get(
                    "model_result_decoded", "model_result not in data_json"
                )
                model_result_raw = data_json.get("model_result_raw")
                possible_answer = data_json.get("possible_answer")

                # Append the data to the rows list
                rows.append(
                    {
                        "id": id,
                        "model_name": model_name,
                        "test_category": test_category,
                        "valid": valid,
                        "error": error,
                        "error_type": error_type,
                        "model_result": model_result,
                        "model_result_raw": model_result_raw,
                        "possible_answer": possible_answer,
                    }
                )
    # Create a dataframe from the accumulated rows
    df = pd.DataFrame(
        rows,
        columns=[
            "id",
            "model_name",
            "test_category",
            "valid",
            "error",
            "error_type",
            "model_result",
            "model_result_raw",
            "possible_answer",
        ],
    )
    # Save the dataframe to a csv file
    df.to_csv(f"./analysis/error_type_{TYPE}_{SUFFIX}.csv", index=False)


def map_error_type(error_type):
    if error_type in error_mapping:
        return error_mapping[error_type]
    else:
        raise KeyError(f"Error type '{error_type}' not found in error_mapping")


def map_test_category(test_category):
    if test_category in test_category_mapping:
        return test_category_mapping[test_category]
    else:
        raise KeyError(
            f"Test category '{test_category}' not found in test_category_mapping"
        )


def map_model_name(model_name):
    if model_name in MODEL_METADATA_MAPPING_DICT:
        return MODEL_METADATA_MAPPING_DICT[model_name]
    else:
        raise KeyError(f"Model '{model_name}' not found in MODEL_METADATA_MAPPING_DICT")


for TYPE in ["ast", "executable", "all"]:
    df = pd.read_csv(f"./analysis/error_type_{TYPE}_{SUFFIX}.csv")
    df["Mapped_Category"] = df["error_type"].apply(map_error_type)
    df["error_type"] = df["Mapped_Category"]
    df["Mapped_Category"] = (
        df["test_category"].map(test_category_mapping).fillna(df["test_category"])
    )
    # df['Mapped_Category'] = df['test_category'].apply(map_test_category)
    df["test_category"] = df["Mapped_Category"]
    df["original_model_name"] = df["model_name"]
    df["Mapped_Model"] = (
        df["model_name"].map(MODEL_METADATA_MAPPING_DICT).fillna(df["model_name"])
    )
    # df['Mapped_Model'] = df['model_name'].apply(map_model_name)
    df["model_name"] = df["Mapped_Model"]

    TEST_CATEGORY_MAPPING = {
        "ast_multiple": 200,
        "exec_multiple": 50,
        "ast_parallel_multiple": 200,
        "exec_parallel": 50,
        "relevance": 240,
        "exec_rest": 70,
        "exec_parallel_multiple": 40,
        "exec_simple": 100,
        "ast_simple": 400,
        "chatable": 200,
        "ast_js": 100,
        "sql": 100,
        "ast_java": 100,
        "ast_parallel": 200,
        "live_simple": 258,
        "live_multiple": 1037,
        "live_parallel": 16,
        "live_parallel_multiple": 24,
        "live_irrelevance": 875,
        "live_relevance": 41,
    }

    df["root_type"] = df["error_type"].str.split(":").str[0]
    df["sub_type"] = df["error_type"].str.split(":").str[1]

    # please create a dataframe that summarizes all groupby results for each model each test_category, counting of each root_type and sub_type error types

    df_summary = pd.DataFrame(
        columns=[
            "model_name",
            "test_category",
            "root_type",
            "sub_type",
            "count",
            "error_type",
        ]
    )
    for model_name in df["model_name"].unique():
        original_model_name = MODEL_METADATA_MAPPING_DICT_inverted[model_name]
        for test in df[df["model_name"] == model_name]["test_category"].unique():
            if LIVE_ONLY:
                if "live" not in test:
                    continue
            df_temp = (
                df[(df["model_name"] == model_name) & (df["test_category"] == test)]
                .groupby(["root_type", "sub_type"])
                .size()
                .reset_index()
            )
            df_temp.columns = ["root_type", "sub_type", "count"]
            df_temp["model_name"] = model_name
            df_temp["original_model_name"] = original_model_name
            df_temp["test_category"] = test
            df_temp["error_type"] = df_temp["root_type"] + ":" + df_temp["sub_type"]
            df_temp["test_category_count"] = df_temp["test_category"].map(
                TEST_CATEGORY_MAPPING
            )
            df_summary = pd.concat([df_summary, df_temp])
    df_summary = df_summary.reset_index(drop=True)
    path = f"./analysis/error_type_summary_{TYPE}_{SUFFIX}.csv"
    df_summary.to_csv(path, index=False)
    print(f"Saved summary for {TYPE} to {path}")


sns.color_palette()
TYPE = "all"
df_summary = pd.read_csv(f"./analysis/error_type_summary_{TYPE}_{SUFFIX}.csv")


# Create the treemap
fig = px.treemap(
    df_summary, path=["model_name", "root_type", "sub_type"], values="count"
)

# Adjusting text attributes globally
fig.update_traces(
    textfont_size=12, selector=dict(type="treemap")  # Adjust the font size as needed
)

fig.update_traces(root_color="#C7CCDB")
fig.update_layout(paper_bgcolor="#e5effc")

fig.show()
fig.write_html(f"./treemap_website_{SUFFIX}.html")


print("🦍 done")
