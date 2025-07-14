import textwrap
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import subprocess
import requests
import re
import json
import sys
import io
import os

# Import for hypothesis testing
from scipy import stats
from statsmodels.stats.weightstats import ztest
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy.stats import chi2_contingency
import statsmodels.api as sm

# IMPORTS FOR DATABASE CONNECTION
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
# IMPORTS FOR LANGCHAIN AGENT
try:
    from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.chat_models import ChatPerplexity
    from langchain_community.chat_models import ChatOllama
    from langchain_ollama import ChatOllama

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# --- Dual Output Helper Functions (Prints to Terminal and displays in Streamlit) ---

def st_info_dual(message, *args, **kwargs):
    """Wrapper for st.info that also prints to terminal."""
    print(f"\n[INFO] Terminal Output:\n{message}\n")
    st.info(message, *args, **kwargs)

def st_success_dual(message, *args, **kwargs):
    """Wrapper for st.success that also prints to terminal."""
    print(f"\n[SUCCESS] Terminal Output:\n{message}\n")
    st.success(message, *args, **kwargs)

def st_warning_dual(message, *args, **kwargs):
    """Wrapper for st.warning that also prints to terminal."""
    print(f"\n[WARNING] Terminal Output:\n{message}\n")
    st.warning(message, *args, **kwargs)
    
def st_error_dual(message, *args, **kwargs):
    """Wrapper for st.error that also prints to terminal."""
    print(f"\n[ERROR] Terminal Output:\n{message}\n")
    st.error(message, *args, **kwargs)

def st_write_dual(data, *args, **kwargs):
    """Wrapper for st.write that also prints to terminal."""
    print(f"\n[WRITE] Terminal Output:")
    print(data)
    print("-" * 20)
    st.write(data, *args, **kwargs)

def st_markdown_dual(message, *args, **kwargs):
    """Wrapper for st.markdown that also prints to terminal."""
    print(f"\n[MARKDOWN] Terminal Output:\n{message}\n") 
    st.markdown(message, *args, **kwargs)

def st_dataframe_dual(df, *args, **kwargs):
    """Wrapper for st.dataframe that also prints to terminal."""
    print("\n[DATAFRAME] Terminal Output:")
    print(df.to_string())
    print("-" * 20)
    st.dataframe(df, *args, **kwargs)
    
def st_code_dual(code, *args, **kwargs):
    """Wrapper for st.code that also prints to terminal."""
    print("\n[CODE] Terminal Output:")
    print(code)
    print("-" * 20)
    st.code(code, *args, **kwargs)
    
def st_text_dual(text, *args, **kwargs):
    """Wrapper for st.text that also prints to terminal."""
    print("\n[TEXT] Terminal Output:")
    print(text)
    print("-" * 20)
    st.text(text, *args, **kwargs)

def plotly_chart_dual(fig, *args, **kwargs):
    """Wrapper for st.plotly_chart that prints a summary to the terminal."""
    title = fig.layout.title.text if fig.layout.title and fig.layout.title.text else "Untitled Chart"
    print(f"\n[PLOTLY CHART] Displaying chart in web app: '{title}'\n")
    st.plotly_chart(fig, *args, **kwargs)


st.set_page_config(page_title="DataSense AI - Advanced Auto EDA", layout="wide")

# --- API KEY PLACEHOLDERS ---
# IMPORTANT: Replace these with your actual API keys
# MODIFIED: Changed the fake key to an empty string. The app will now correctly prompt you if the key is missing.
GEMINI_API_KEY = "AIzaSyD21icfIXo9M8QUhYhq8kDuTbnDMvhN0Zc"  # Replace with your Gemini API key
PPLX_API_KEY = "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" # Replace with your Perplexity API key

# Ollama specific constants
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MISTRAL_MODEL = "mistral:latest"


# --- Initialize Session State ---
def init_session_state():
    defaults = {
        'df_original': None,
        'data_source': None,
        'active_filters': {},
        'viz_filters': {},
        'uploaded_dfs': {},
        'column_mv_strategies': {},
        'llm_eda_suggestions': [],
        'llm_autoclean_log': [],
        'processed_df_for_custom_clean': None,
        'sandbox_conversation': [],
        'page': 'landing', # New state to manage navigation
        'dataset_summary': None, # To store AI-generated summary
        'data_metrics': None # To store rows, cols, missing values info
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

common_na_values = ["NA", "N/A", "", " ", "NULL", "null", "-", "?", "None", "nan", "NaN"]

# --- All Helper Functions ---
def get_data_summary(df):
    summary_data = {}
    summary_data['shape_info'] = f"{df.shape[0]} rows, {df.shape[1]} columns"
    column_types_data = [{"Column": col, "Data Type": str(dtype)} for col, dtype in df.dtypes.items()]
    summary_data['column_types_df'] = pd.DataFrame(column_types_data)
    missing_values_data = df.isnull().sum().reset_index()
    missing_values_data.columns = ['Column', 'Missing Values Count']
    missing_values_data = missing_values_data[missing_values_data['Missing Values Count'] > 0].sort_values(by='Missing Values Count', ascending=False)
    summary_data['missing_values_df'] = missing_values_data
    unique_values_data = []
    for col in df.columns:
        num_unique = df[col].nunique()
        sample_values = ""
        if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):
            unique_vals = df[col].dropna().unique()
            sample_values = ', '.join(map(str, unique_vals[:10])) + (", ..." if num_unique > 10 else "")
        elif pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            if not df[col].dropna().empty:
                sample_values = f"Min: {df[col].min()}, Max: {df[col].max()}"
            else:
                sample_values = "N/A (no non-null values)"
        unique_values_data.append({"Column": col, "Unique Count": num_unique, "Sample Unique Values": sample_values})
    summary_data['unique_values_df'] = pd.DataFrame(unique_values_data)
    return summary_data

def generate_ai_insights(df):
    insights = []
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if not missing_cols.empty:
        insights.append(f"⚠️ Columns with missing values: {', '.join(missing_cols.index)}")
        insights.append("💡 Consider imputing or dropping rows/columns with missing values.")
    else:
        insights.append("✅ No missing values found.")
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    insights.append(f"🔢 Numeric columns: {len(num_cols)} ({', '.join(num_cols)})" if len(num_cols) else "🔢 No numeric columns")
    insights.append(f"🔤 Categorical columns: {len(cat_cols)} ({', '.join(cat_cols)})" if len(cat_cols) else "🔤 No categorical columns")
    if len(num_cols) >= 2:
        corr = df[num_cols].corr().abs()
        high_corr_pairs = [(corr.columns[i], corr.columns[j], corr.iloc[i, j])
                           for i in range(len(corr.columns))
                           for j in range(i + 1, len(corr.columns))
                           if corr.iloc[i, j] > 0.75]
        if high_corr_pairs:
            pairs_str = ", ".join([f"{x[0]} & {x[1]} ({x[2]:.2f})" for x in high_corr_pairs])
            insights.append(f"🔗 High correlations (>0.75): {pairs_str}")
            insights.append("💡 Consider reducing multicollinearity.")
        else:
            insights.append("✅ No strong correlations found.")
    else:
        insights.append("ℹ️ Not enough numeric columns for correlation analysis.")
    if len(num_cols) > 0:
        skewed = df[num_cols].skew().abs()
        skewed_cols = skewed[skewed > 1].index.tolist()
        if skewed_cols:
            insights.append(f"📈 Skewed columns: {', '.join(skewed_cols)}")
            insights.append("💡 Apply log/sqrt transformations.")
        else:
            insights.append("✅ Distributions are fairly symmetric.")
    rows = df.shape[0]
    if rows < 100:
        insights.append("⚠️ Small dataset (<100 rows).")
    elif rows > 100000:
        insights.append("⚠️ Large dataset (>100k rows). Performance may vary.")
    else:
        insights.append("✅ Dataset size is moderate.")
    high_card_cols = [col for col in cat_cols if df[col].nunique() > 50]
    if high_card_cols:
        insights.append(f"⚠️ High cardinality: {', '.join(high_card_cols)}")
        insights.append("💡 Consider encoding or grouping.")
    return insights

def perplexity_chat(question, df, prompt_type="all_rounder", system_message=None):
    if not PPLX_API_KEY or PPLX_API_KEY == "YOUR_PPLX_API_KEY":
        return "❌ Perplexity API Key is not set. Please replace 'YOUR_PPLX_API_KEY' with your actual key in the code."
    API_URL = "https://api.perplexity.ai/chat/completions"
    PPLX_MODEL = "sonar-pro"
    DATA_THRESHOLD = 100000
    context = ""
    is_full_data = df.size <= DATA_THRESHOLD
    if is_full_data:
        context_preamble = "You are working with the **complete dataset**, provided below in CSV format. Your analysis should be based on this full data."
        with io.StringIO() as buffer:
            df.to_csv(buffer, index=False)
            data_csv = buffer.getvalue()
        context = f"{context_preamble}\n\n### Full Dataset (CSV Format)\n```csv\n{data_csv}\n```"
    else:
        context_preamble = "You are working with a **summary of a large dataset** because the full data is too large to send. Base your analysis on the provided summary."
        with io.StringIO() as buffer:
            df.info(buf=buffer, verbose=True, show_counts=True)
            info_str = buffer.getvalue()
        summary_parts = [context_preamble]
        summary_parts.append(f"\n### Dataset Information\nThis provides an overview of columns, data types, and non-null counts.\n```\n{info_str}\n```")
        summary_parts.append(f"\n### First 5 Rows\nThis gives a glimpse of the data structure and values.\n{df.head().to_markdown(index=False)}")
        numeric_summary = df.describe(include=np.number)
        if not numeric_summary.empty:
            summary_parts.append(f"\n### Numeric Data Summary\nStatistical details for all numeric columns.\n{numeric_summary.to_markdown()}")
        categorical_summary = df.describe(include=['object', 'category'])
        if not categorical_summary.empty:
            summary_parts.append(f"\n### Categorical Data Summary\nSummary for non-numeric columns, including counts of unique values (unique), the most frequent value (top), and its frequency (freq).\n{categorical_summary.to_markdown()}")
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            summary_parts.append(f"\n### Missing Values Count\nColumns with one or more missing values.\n{missing_values[missing_values > 0].to_frame('missing_count').to_markdown()}")
        context = "\n".join(summary_parts)

    default_prompt_templates = {
        "python_code_analysis": {
            "system": "You are a professional Python data analyst. Use pandas, numpy, seaborn, matplotlib, and sklearn as needed. Only output code, no explanations.",
            "user": "I have a dataset containing columns: 'Age', 'Gender', 'Income', 'Purchase_History'.\nWrite a Python script to:\n- Load the CSV\n- Perform basic EDA\n- Show outliers and missing values\n- Visualize distributions\n- Prepare the data for modeling"
        },
        "eda_analysis": {
            "system": """You are a senior data analyst who operates with complete access to the user's data.
Your primary principle is to perform a full analysis, not one based on incomplete samples or summaries. Your code and analysis assume that the entire dataset has been loaded (e.g., via `df = df  # DataFrame already loaded from uploaded file`) without limitations like `nrows`.
You have been provided with one of two things:
1.  The **complete dataset** in CSV format (for smaller files).
2.  A **comprehensive summary** of the dataset (for larger files that cannot be sent in full).
Based on the provided context (either full data or summary), your job is to provide deep, actionable insights. Answer the user's question with clear, business-friendly interpretations.
""",
            "user": "Here is the dataset context:\n{context}\n\nBased on this, please answer the following question:\n{question}"
        },
        "business_analysis": {
            "system": "You are a strategic business analyst. Explain patterns, trends, risks, and opportunities using the provided data context. Your output should be structured, insightful, and directly applicable for executives.",
            "user": "Based on this data context:\n{context}\n\nBusiness Question:\n{question}"
        },
        "ml_random_forest": {
            "system": "You are a Python machine learning expert. Use sklearn and standard libraries to build, evaluate, and explain models. Only output clean, commented Python code.",
            "user": "Create a random forest classifier using the data described in this context:\n{context}\n\nTask:\n{question}"
        },
        "all_rounder": {
            "system": "You are a full-stack data analyst. Provide code, explain findings, and give business insights based on the data context. Be structured and detailed.",
            "user": "Dataset Context:\n{context}\n\nRequest:\n{question}"
        }
    }
    
    current_system_message = system_message if system_message else default_prompt_templates.get(prompt_type, default_prompt_templates["all_rounder"])["system"]
    
    if system_message:
        current_user_message = f"Dataset Context:\n{context}\n\nRequest:\n{question}"
    else:
        current_user_message = default_prompt_templates.get(prompt_type, default_prompt_templates["all_rounder"])["user"].format(context=context, question=question)

    messages = [
        {"role": "system", "content": current_system_message},
        {"role": "user", "content": current_user_message}
    ]
    payload = {
        "model": PPLX_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 8192,
    }
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {PPLX_API_KEY}",
        "content-type": "application/json"
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        if result and isinstance(result, dict) and result.get("choices"):
            content = result["choices"][0]["message"].get("content")
            return content if content else "❌ No content in API response."
        else:
            return f"❌ Unexpected response format: {result}"
    except requests.exceptions.Timeout:
        return "❌ API request timed out after 120 seconds."
    except requests.exceptions.RequestException as e:
        return f"❌ API request failed: {e}"
    except Exception as e:
        return f"❌ An unexpected error occurred: {e}"

def mistral_chat(question, context):
    prompt = (
        f"You are an expert data analyst with a deep understanding of datasets. "
        f"You have been provided with a summary of a Pandas DataFrame. "
        f"Your task is to analyze the data based on the user's question and provide insightful observations, "
        f"trends, potential issues (like outliers, skewness, missing data), and relationships between columns. "
        f"Focus on explaining your findings clearly and concisely, just like a human data analyst would. "
        f"Do NOT generate any code, explain how you would analyze the data, or list steps. "
        f"Just provide a direct, descriptive answer based on the context.\n\n"
        f"Dataset context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    try:
        result = subprocess.run(
            ["ollama", "run", "mistral", prompt],
            capture_output=True, text=True, check=True, timeout=120
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error querying Mistral via Ollama: {e.stderr.strip()}. Please ensure 'mistral' model is downloaded (`ollama pull mistral`) and Ollama server is running."
    except FileNotFoundError:
        return "Error: 'ollama' command not found. Please ensure Ollama is installed and in your system's PATH."
    except subprocess.TimeoutExpired:
        return "Error: Ollama response timed out after 120 seconds. The model might be taking too long or is not running. Consider increasing the timeout or checking Ollama server status."
    except Exception as e:
        return f"An unexpected error occurred with Mistral: {e}"

def ollama_generic_chat(question, context, model, system_message=None):
    if not LANGCHAIN_AVAILABLE:
        return "❌ LangChain is not installed. Please install it to use Ollama AI features."
    try:
        llm = ChatOllama(model=model, base_url=OLLAMA_BASE_URL, temperature=0.1)
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        user_content = f"Here is the dataset context:\n{context}\n\nQuestion: {question}"
        messages.append({"role": "user", "content": user_content})
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"❌ Error querying '{model}' via Ollama: {e}. Please ensure the model is downloaded (`ollama pull {model}`) and Ollama server is running at {OLLAMA_BASE_URL}."

def llama3_chat(question, context):
    prompt = f"Dataset context:\n{context}\n\nQuestion: {question}\nAnswer:"
    try:
        result = subprocess.run(
            ["ollama", "run", "llama3.2:latest", prompt],
            capture_output=True, text=True, check=True, timeout=60
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error querying Llama 3.2 via Ollama: {e.stderr.strip()}. Please ensure 'llama3.2:latest' model is downloaded (`ollama pull llama3.2:latest`) and Ollama server is running."
    except FileNotFoundError:
        return "Error: 'ollama' command not found. Please ensure Ollama is installed and in your system's PATH."
    except subprocess.TimeoutExpired:
        return "Error: Ollama response timed out after 60 seconds. The model might be taking too long or is not running. Consider increasing the timeout or checking Ollama server status."
    except Exception as e:
        return f"An unexpected error occurred with Llama 3.2: {e}"


# MODIFIED: This function has been updated to correctly check for a missing API key.
def chat_bot(messages, generation_config=None):
    API_KEY = GEMINI_API_KEY
    if not API_KEY:
        return "❌ Google Gemini API Key is not set. Please replace 'YOUR_GOOGLE_CLOUD_API_KEY' with your actual key in the code."
    API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"
    gemini_chat_history = []
    system_instruction = ""
    for msg in messages:
        if msg["role"] == "system":
            system_instruction = msg["content"]
        elif msg["role"] == "user":
            gemini_chat_history.append({"role": "user", "parts": [{"text": msg["content"]}]})
        elif msg["role"] == "assistant":
            gemini_chat_history.append({"role": "model", "parts": [{"text": msg["content"]}]})
    if system_instruction and gemini_chat_history and gemini_chat_history[0]["role"] == "user":
        gemini_chat_history[0]["parts"][0]["text"] = system_instruction + "\n\n" + gemini_chat_history[0]["parts"][0]["text"]
    payload = {
        "contents": gemini_chat_history,
        "generationConfig": generation_config if generation_config else {
            "temperature": 0.7, "maxOutputTokens": 8192, "candidateCount": 1
        }
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        if result.get("candidates") and len(result["candidates"]) > 0 and \
           result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts") and \
           len(result["candidates"][0]["content"]["parts"]) > 0:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return f"❌ No text content found in API response. Full response: {result}"
    except requests.exceptions.RequestException as e:
        return f"❌ API request failed: {e}"
    except KeyError as e:
        return f"❌ Unexpected response format from API. Missing key: {e}. Full response: {result}"
    except Exception as e:
        return f"❌ An unexpected error occurred: {e}"

def _apply_cleaning_action(df_to_process, action_dict, log_entries):
    col_name = action_dict.get('column')
    action_type = action_dict.get('action')
    reason = action_dict.get('reason', 'No reason provided.')
    if col_name not in df_to_process.columns:
        log_entries.append(f"⚠️ Skipped '{col_name}': Column not found. Reason: {reason}")
        return df_to_process, False
    try:
        if action_type == "impute_missing":
            method = action_dict.get('method')
            value = action_dict.get('value')
            initial_nulls = df_to_process[col_name].isnull().sum()
            if initial_nulls > 0:
                if method == "mean":
                    if pd.api.types.is_numeric_dtype(df_to_process[col_name]):
                        df_to_process[col_name].fillna(df_to_process[col_name].mean(), inplace=True)
                        log_entries.append(f"✅ Imputed '{col_name}' with mean. ({initial_nulls} nulls filled. Reason: {reason})")
                        return df_to_process, True
                    else:
                        log_entries.append(f"❌ Could not impute '{col_name}' with mean: not numeric. Reason: {reason}")
                elif method == "median":
                    if pd.api.types.is_numeric_dtype(df_to_process[col_name]):
                        df_to_process[col_name].fillna(df_to_process[col_name].median(), inplace=True)
                        log_entries.append(f"✅ Imputed '{col_name}' with median. ({initial_nulls} nulls filled. Reason: {reason})")
                        return df_to_process, True
                    else:
                        log_entries.append(f"❌ Could not impute '{col_name}' with median: not numeric. Reason: {reason}")
                elif method == "mode":
                    if not df_to_process[col_name].empty:
                        mode_val = df_to_process[col_name].mode()[0]
                        df_to_process[col_name].fillna(mode_val, inplace=True)
                        log_entries.append(f"✅ Imputed '{col_name}' with mode. ({initial_nulls} nulls filled. Reason: {reason})")
                        return df_to_process, True
                    else:
                        log_entries.append(f"❌ Could not impute '{col_name}' with mode: column is empty. Reason: {reason}")
                elif method == "constant" and value is not None:
                    if pd.api.types.is_numeric_dtype(df_to_process[col_name]):
                        try:
                            converted_value = pd.to_numeric(value)
                        except ValueError:
                            converted_value = value
                    else:
                        converted_value = value
                    df_to_process[col_name].fillna(converted_value, inplace=True)
                    log_entries.append(f"✅ Imputed '{col_name}' with constant '{converted_value}'. ({initial_nulls} nulls filled. Reason: {reason})")
                    return df_to_process, True
                else:
                    log_entries.append(f"⚠️ Skipped imputation for '{col_name}': Invalid method or missing value. Reason: {reason}")
            else:
                log_entries.append(f"ℹ️ No missing values in '{col_name}', imputation skipped. Reason: {reason}")
        elif action_type == "drop_rows_on_missing":
            rows_before_drop = df_to_process.shape[0]
            df_to_process.dropna(subset=[col_name], inplace=True)
            rows_after_drop = df_to_process.shape[0]
            if rows_before_drop != rows_after_drop:
                log_entries.append(f"✅ Dropped {rows_before_drop - rows_after_drop} rows due to missing values in '{col_name}'. Reason: {reason}")
                return df_to_process, True
            else:
                log_entries.append(f"ℹ️ No rows dropped for '{col_name}': no missing values found. Reason: {reason}")
        elif action_type == "drop_column":
            if col_name in df_to_process.columns:
                df_to_process.drop(columns=[col_name], inplace=True)
                log_entries.append(f"✅ Dropped column '{col_name}'. Reason: {reason}")
                return df_to_process, True
            else:
                log_entries.append(f"⚠️ Skipped dropping '{col_name}': Column already dropped or not found. Reason: {reason}")
        elif action_type == "convert_type":
            to_type = action_dict.get('to_type')
            data_format = action_dict.get('format')
            original_dtype = str(df_to_process[col_name].dtype)
            if to_type == "float":
                df_to_process[col_name] = pd.to_numeric(df_to_process[col_name], errors='coerce')
                log_entries.append(f"✅ Converted '{col_name}' from '{original_dtype}' to 'float'. Reason: {reason}")
                return df_to_process, True
            elif to_type == "int":
                df_to_process[col_name] = pd.to_numeric(df_to_process[col_name], errors='coerce').astype('Int64')
                log_entries.append(f"✅ Converted '{col_name}' from '{original_dtype}' to 'integer'. Reason: {reason}")
                return df_to_process, True
            elif to_type == "datetime":
                if data_format:
                    df_to_process[col_name] = pd.to_datetime(df_to_process[col_name], format=data_format, errors='coerce')
                    log_entries.append(f"✅ Converted '{col_name}' from '{original_dtype}' to 'datetime' (format: '{data_format}'). Reason: {reason}")
                    return df_to_process, True
                else:
                    df_to_process[col_name] = pd.to_datetime(df_to_process[col_name], errors='coerce')
                    log_entries.append(f"✅ Converted '{col_name}' from '{original_dtype}' to 'datetime' (inferred format). Reason: {reason}")
                    return df_to_process, True
            elif to_type == "category":
                df_to_process[col_name] = df_to_process[col_name].astype('category')
                log_entries.append(f"✅ Converted '{col_name}' from '{original_dtype}' to 'category'. Reason: {reason}")
                return df_to_process, True
            else:
                log_entries.append(f"⚠️ Skipped type conversion for '{col_name}': Unsupported target type '{to_type}'. Reason: {reason}")
        elif action_type == "normalize":
            if pd.api.types.is_numeric_dtype(df_to_process[col_name]):
                scaler = StandardScaler()
                non_null_mask = df_to_process[col_name].notnull()
                df_to_process.loc[non_null_mask, col_name] = scaler.fit_transform(df_to_process.loc[non_null_mask, [col_name]])
                log_entries.append(f"✅ Normalized column '{col_name}'. Reason: {reason}")
                return df_to_process, True
            else:
                log_entries.append(f"❌ Could not normalize '{col_name}': not numeric. Reason: {reason}")
        else:
            log_entries.append(f"⚠️ Unsupported action type '{action_type}' for column '{col_name}'. Reason: {reason}")
    except Exception as e:
        log_entries.append(f"❌ Error applying action to '{col_name}': {action_type} - {e}. Reason: {reason}")
    return df_to_process, False

def generate_python_for_sandbox(query, df, correction_context=None):
    """Generates Python code using Google Gemini, with an optional self-correction mechanism."""
    if correction_context:
        code_gen_system_prompt = (
            "You are an expert Python debugger and data analyst. The user's previous code attempt failed. "
            "Your task is to analyze the user's original goal, the faulty code, and the resulting error message. "
            "You MUST rewrite the code to fix the error and achieve the original goal. "
            "Output ONLY the corrected, complete Python code. Do not include any explanations, introductory text, or markdown code fences like ```python."
        )
        code_gen_user_prompt = (
            f"The original goal was to answer this question: '{query}'\n\n"
            f"The following code failed:\n```python\n{correction_context['failed_code']}\n```\n\n"
            f"It produced this error:\n```\n{correction_context['error_message']}\n```\n\n"
            f"Please provide the corrected Python code:"
        )
    else:
        df_summary_for_code_gen = f"""
        You have access to a pandas DataFrame named `df`.
        DataFrame Info:
        - Shape: {df.shape}
        - Columns and Data Types:
{df.dtypes.to_string()}
        - First 5 rows:
{df.head().to_string()}
        """
        code_gen_system_prompt = (
            "You are an expert Python data analyst. Your task is to write a Python script to answer a user's question about a pandas DataFrame. "
            "The DataFrame is available in the variable `df`. "
            "The script must perform the necessary analysis and print the final result to standard output using the `print()` function. "
            "The output should be a clear, concise answer to the user's question (e.g., a number, a list of items, a small table, or a descriptive sentence). "
            "Do NOT include any example usage or explanations, just the pure Python code. "
            "Ensure the code is self-contained and relies only on standard libraries like pandas and numpy, which are already imported and available as `pd` and `np`. "
            "The DataFrame is available as `df`."
        )
        code_gen_user_prompt = f"DataFrame Summary:\n{df_summary_for_code_gen}\n\nUser Question: {query}\n\nPython Code:"

    messages_for_code_gen = [
        {"role": "system", "content": code_gen_system_prompt},
        {"role": "user", "content": code_gen_user_prompt}
    ]
    generated_code = chat_bot(messages_for_code_gen)
    if generated_code.strip().startswith("```python"):
        generated_code = generated_code.strip()[9:].strip()
        if generated_code.endswith("```"):
            generated_code = generated_code[:-3].strip()
    return generated_code

def generate_python_for_sandbox_pplx(query, df, correction_context=None):
    """Generates Python code using Perplexity AI, with an optional self-correction mechanism."""
    if correction_context:
        system_message = (
            "You are an expert Python debugger and data analyst. The user's previous code attempt failed. "
            "Your task is to analyze the user's original goal, the faulty code, and the resulting error message. "
            "You MUST rewrite the code to fix the error and achieve the original goal. "
            "Based on the dataset context provided, output ONLY the corrected, complete Python code. Do not include any explanations or markdown."
        )
        user_question_for_pplx = (
            f"My original goal was to answer this question: '{query}'\n\n"
            f"The following code failed:\n```python\n{correction_context['failed_code']}\n```\n\n"
            f"It produced this error:\n```\n{correction_context['error_message']}\n```\n\n"
            f"Please provide the corrected Python code:"
        )
        generated_code = perplexity_chat(question=user_question_for_pplx, df=df, system_message=system_message)
    else:
        system_message = (
            "You are an expert Python data analyst. Your task is to write a Python script to answer a user's question about a pandas DataFrame. "
            "The DataFrame is available in a variable named `df`. "
            "The script MUST perform the necessary analysis and print the final result to standard output using the `print()` function. "
            "Your output must ONLY be the Python code. Do not include any explanations, introductory text, or markdown code fences like ```python. "
            "The code must be self-contained and rely only on pandas and numpy, which are available as `pd` and `np`. "
            "The DataFrame is named `df`."
        )
        generated_code = perplexity_chat(question=query, df=df, system_message=system_message)

    if isinstance(generated_code, str) and generated_code.strip().startswith("```python"):
        generated_code = generated_code.strip()[9:].strip()
        if generated_code.endswith("```"):
            generated_code = generated_code[:-3].strip()
    if isinstance(generated_code, str) and generated_code.startswith("❌"):
        return f"❌ Perplexity API Error: {generated_code}"
    return generated_code



def generate_python_for_sandbox_mistrallatest(query, df, correction_context=None):
    """Generates Python code using Mistral, with an optional self-correction mechanism."""
    if correction_context:
        system_message = (
            "You are an expert Python debugger and data analyst. The user's previous code attempt failed. "
            "Your task is to analyze the user's original goal, the faulty code, and the resulting error message. "
            "You MUST rewrite the code to fix the error and achieve the original goal. "
            "Output ONLY the corrected, complete Python code. Do not include any explanations or markdown."
        )
        user_question = (
            f"My original goal was to answer this question: '{query}'\n\n"
            f"The following code failed:\n```python\n{correction_context['failed_code']}\n```\n\n"
            f"It produced this error:\n```\n{correction_context['error_message']}\n```\n\n"
            f"Please provide the corrected Python code:"
        )
    else:
        system_message = (
            "You are an expert Python data analyst. Your task is to write a Python script to answer a user's question about a pandas DataFrame. "
            "The DataFrame is available in a variable named `df`. "
            "The script MUST perform the necessary analysis and print the final result to standard output using the `print()` function. "
            "Your output must ONLY be the Python code. Do not include any explanations, introductory text, or markdown code fences like ```python. "
            "The code must be self-contained and rely only on pandas and numpy, which are available as `pd` and `np`. "
            "The DataFrame is named `df`."
        )
        user_question = query
    
    df_summary_for_code_gen = ""
    with io.StringIO() as buffer:
        df.info(buf=buffer, verbose=True, show_counts=True)
        df_summary_for_code_gen += buffer.getvalue() + "\n\n"
    df_summary_for_code_gen += f"First 5 rows:\n{df.head().to_markdown(index=False)}\n\n"
    numeric_desc_for_ai = df.describe(include=np.number)
    if not numeric_desc_for_ai.empty:
        df_summary_for_code_gen += f"Numeric Column Summary:\n{numeric_desc_for_ai.to_markdown()}\n\n"
    
    generated_code = ollama_generic_chat(
        question=user_question, context=df_summary_for_code_gen, model=OLLAMA_MISTRAL_MODEL, system_message=system_message
    )
    if '```python' in generated_code:
        generated_code = generated_code.split('```python')[1].split('```')[0]
    generated_code = textwrap.dedent(generated_code).strip()
    return generated_code

def execute_and_explain_sandbox(query, generated_code, df, model_choice="Google Gemini"):
    """
    Executes AI-generated Python code in a sandboxed environment.
    If the code fails, it asks the AI to correct it and retries once.
    Finally, it asks the AI to explain the result or the final error.
    """
    MAX_RETRIES = 1
    current_code = generated_code
    code_output = ""
    final_explanation = ""
    
    st_status_container = st.status(f"Executing AI-generated code for: '{query}'", expanded=True)

    with st_status_container:
        for attempt in range(MAX_RETRIES + 1):
            if attempt > 0:
                st.info(f"An error occurred. Attempting to self-correct (Attempt {attempt}/{MAX_RETRIES})...")

            old_stdout = sys.stdout
            redirected_output = io.StringIO()
            try:
                st.write("▶️ **Attempting to execute the following code:**")
                st.code(current_code, language="python")
                
                # Execute the code
                compile(current_code, '<string>', 'exec')
                sys.stdout = redirected_output
                local_scope = {'df': df.copy(), 'pd': pd, 'np': np} # Use a copy of df
                exec(current_code, globals(), local_scope)
                sys.stdout = old_stdout # Restore stdout
                
                code_output = redirected_output.getvalue()
                if not code_output:
                    code_output = "The script ran successfully but produced no printed output."
                
                st.success("✅ Code executed successfully!")
                break # Exit loop on success
            
            except Exception as e:
                sys.stdout = old_stdout # Restore stdout
                error_message = f"An error occurred during code execution:\n{e}\n\nCaptured output before error:\n{redirected_output.getvalue()}"
                
                st.error(f"Execution failed on attempt {attempt + 1}.")
                st.code(error_message, language="text")

                if attempt >= MAX_RETRIES:
                    code_output = f"The code failed after {attempt + 1} attempts. Last error:\n{error_message}"
                    st_status_container.update(label="Code execution failed after multiple attempts.", state="error")
                    break

                st.write(f"🤖 Asking AI ({model_choice}) to fix the code...")
                correction_context = {'failed_code': current_code, 'error_message': error_message}
                
                new_code = ""
                if model_choice == "Perplexity AI":
                    new_code = generate_python_for_sandbox_pplx(query, df, correction_context=correction_context)
                elif model_choice == "mistral:latest":
                    new_code = generate_python_for_sandbox_mistrallatest(query, df, correction_context=correction_context)
                else: # Google Gemini
                    new_code = generate_python_for_sandbox(query, df, correction_context=correction_context)

                if not new_code or new_code.strip() == "" or new_code.startswith("❌"):
                    code_output = f"AI failed to correct the code. Last error:\n{error_message}"
                    st_status_container.update(label="AI failed to correct the code.", state="error")
                    break

                # Clean the new code
                if new_code.strip().startswith("```python"):
                    new_code = new_code.strip()[9:].strip()
                if new_code.endswith("```"):
                    new_code = new_code[:-3].strip()
                new_code = new_code.strip().strip('"').strip("'")
                new_code = textwrap.dedent(new_code).strip()

                st.info("AI has provided a corrected version of the code.")
                current_code = new_code # Update code for the next loop iteration
            
            finally:
                sys.stdout = old_stdout

        # --- Explanation Generation ---
        st.write("💬 Generating final explanation...")
        explanation_system_prompt = ""
        explanation_user_prompt = ""
        
        # Check if the last execution resulted in an error
        if "An error occurred" in code_output or "A syntax error was found" in code_output:
            explanation_system_prompt = (
                "You are an expert Python debugger. You were asked to execute a script, but it failed even after a correction attempt. "
                "You will be given the user's original request, the final code that failed, and the final error message. "
                "Your task is to explain what went wrong in a clear, easy-to-understand way. Do not attempt to re-run the code. Just explain the error."
            )
            explanation_user_prompt = (
                f"Here is the debugging context:\n\n"
                f"**Original Question:** {query}\n\n"
                f"**Final Python Code that Failed:**\n```python\n{current_code}\n```\n\n"
                f"**Final Error Message:**\n```\n{code_output}\n```\n\n"
                f"**Your Explanation of the Error:**"
            )
            st_status_container.update(label="Execution failed. Generating error analysis.", state="error")
        else:
            explanation_system_prompt = (
                "You are an expert data analyst and a helpful assistant. "
                "Your role is to explain the results of a data analysis script to a user. "
                "You will be given the user's original question, the Python code that was successfully executed, and the output from that code. "
                "Your task is to provide a clear, easy-to-understand explanation of what the code did and what the output means in the context of the user's question. "
                "Structure your response well. Start by summarizing the finding, then explain the methodology and the result."
            )
            explanation_user_prompt = (
                f"Here is the analysis context:\n\n"
                f"**Original Question:** {query}\n\n"
                f"**Python Code Executed:**\n```python\n{current_code}\n```\n\n"
                f"**Output of the Code:**\n```\n{code_output}\n```\n\n"
                f"**Your Explanation:**"
            )
            st_status_container.update(label="Execution complete. Generating final analysis.", state="complete")

        # Call the appropriate LLM for the final explanation
        if model_choice == "Google Gemini":
            messages_for_explanation = [
                {"role": "system", "content": explanation_system_prompt},
                {"role": "user", "content": explanation_user_prompt}
            ]
            final_explanation = chat_bot(messages_for_explanation)
        elif  model_choice == "Perplexity AI":
            final_explanation = perplexity_chat(
                question=explanation_user_prompt, df=pd.DataFrame(), system_message=explanation_system_prompt
            )
        
            
    return code_output, final_explanation, current_code

# --- Data Analysis and Metrics Generation ---
def analyze_and_store_data_details(df):
    """Analyzes the dataframe and stores summary and metrics in session state."""
    # 1. Generate AI summary
    with st.spinner("Generating AI summary of the dataset..."):
        # Using a default prompt for general dataset description
        summary_prompt = (
            "Based on the provided data summary (column names, types, sample data), "
            "provide a brief, high-level description of what this dataset is likely about. "
            "Identify the potential domain (e.g., e-commerce, finance, healthcare) and the main entities represented. "
            "Keep the description to 2-3 sentences."
        )
        summary = perplexity_chat(
            question=summary_prompt,
            df=df,
            system_message="You are a data analyst who is skilled at quickly understanding datasets from their schema and a few sample rows. Your goal is to provide a concise summary."
        )
        st.session_state.dataset_summary = summary

    # 2. Calculate data metrics
    rows, cols = df.shape
    missing_values = df.isnull().sum()
    cols_with_missing = missing_values[missing_values > 0]
    missing_info = []
    if not cols_with_missing.empty:
        for col, count in cols_with_missing.items():
            percentage = (count / rows) * 100
            missing_info.append({
                "Column": col,
                "Missing Count": count,
                "Percentage": f"{percentage:.2f}%"
            })
    
    st.session_state.data_metrics = {
        "rows": rows,
        "columns": cols,
        "missing_info_df": pd.DataFrame(missing_info)
    }

# --- Theming ---
bg_color = "#FFFFFF"      
font_color = "#000000"    
primary_color = "#E40000" 

st.markdown(f"""
    <style>
    /* Base app colors */
    .reportview-container, .main {{
        background-color: {bg_color};
        color: {font_color};
    }}
    .sidebar .sidebar-content {{
        background-color: {bg_color};
    }}
    h1, h2, h3 {{
        color: {primary_color}; 
    }}
    .stButton > button {{
        color: #FFFFFF;
        background-color: {primary_color};
        border-color: {primary_color};
    }}
    .stButton > button:hover {{
        background-color: #c30000; 
        border-color: #c30000;
        color: #FFFFFF;
    }}
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
         border-bottom: 3px solid {primary_color};
         color: {primary_color};
    }}
    .stSlider [data-baseweb="slider"] > div:nth-of-type(3) {{
        background-color: {primary_color}; 
    }}
    .stSlider [data-baseweb="slider"] > div:nth-of-type(1) > div {{
        background-color: {primary_color}; 
    }}
    /* Custom styles for the navigation boxes */
    .nav-box {{
        padding: 20px;
        border-radius: 10px;
        background-color: #f8f9fa;
        text-align: center;
        border: 1px solid #dee2e6;
        transition: transform 0.2s, box-shadow 0.2s;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}
    .nav-box:hover {{
        transform: translateY(-5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }}
    .nav-box h3 {{
        margin-top: 0;
        color: {primary_color};
    }}
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Controls ---
st.sidebar.title("📊 Data Operations")

# 1. Load Data Expander
with st.sidebar.expander("📂 Load Data", expanded=True):
    data_source_choice = st.radio(
        "Choose Data Source:",
        ("Upload CSV", "Upload Excel", "Connect to Database"),
        key="data_source_choice"
    )

    uploaded_files = None
    if data_source_choice == "Upload CSV":
        uploaded_files = st.file_uploader(
            "Upload one or more CSV files",
            type="csv",
            accept_multiple_files=True,
            key="csv_uploader",
            help="Upload your datasets here"
        )
        if uploaded_files:
            for file in uploaded_files:
                file_name = file.name.split('.')[0]
                try:
                    df_temp = pd.read_csv(file, na_values=common_na_values)
                    st.session_state['uploaded_dfs'][file_name] = df_temp
                except Exception as e:
                    st_error_dual(f"Error loading {file.name}: {e}")

            if len(st.session_state['uploaded_dfs']) > 0:
                merge_all_uploaded_files = st.checkbox("Load all uploaded CSV files as a single merged DataFrame (if compatible)", value=False, key="merge_initial_csv_uploader")
                if merge_all_uploaded_files and len(st.session_state['uploaded_dfs']) > 1:
                    try:
                        df_list_to_concat = list(st.session_state['uploaded_dfs'].values())
                        st.session_state['df_original'] = pd.concat(df_list_to_concat, ignore_index=True)
                        st.session_state['data_source'] = "csv"
                        st_success_dual("All uploaded CSV files concatenated into one DataFrame.")
                    except Exception as e:
                        st_warning_dual(f"Could not concatenate all uploaded files (likely incompatible columns). Please use the 'Link DataFrames' feature for merging. Error: {e}")
                        df_keys = list(st.session_state['uploaded_dfs'].keys())
                        selected_initial_file_key = st.selectbox("Select one of the uploaded files to begin EDA:", df_keys, key="select_single_initial_csv")
                        st.session_state['df_original'] = st.session_state['uploaded_dfs'][selected_initial_file_key]
                        st.session_state['data_source'] = "csv"
                elif len(st.session_state['uploaded_dfs']) == 1:
                    st.session_state['df_original'] = list(st.session_state['uploaded_dfs'].values())[0]
                    st.session_state['data_source'] = "csv"
                else:
                    df_keys = list(st.session_state['uploaded_dfs'].keys())
                    selected_initial_file_key = st.selectbox("Select one of the uploaded files to begin EDA:", df_keys, key="select_single_initial_csv")
                    st.session_state['df_original'] = st.session_state['uploaded_dfs'][selected_initial_file_key]
                    st.session_state['data_source'] = "csv"
            else:
                st_info_dual("Upload one or more CSV files, or connect to a database to begin EDA.")
        else:
            st.session_state['df_original'] = None
            st.session_state['data_source'] = None
            st.session_state['uploaded_dfs'] = {}
            st_info_dual("Upload one or more CSV files, or connect to a database to begin EDA.")

    elif data_source_choice == "Upload Excel":
        uploaded_excel_files = st.file_uploader(
            "Upload one or more Excel files",
            type=["xlsx", "xls"],
            accept_multiple_files=True,
            key="excel_uploader",
            help="Upload your Excel datasets here"
        )
        if uploaded_excel_files:
            temp_uploaded_dfs = {}
            for i, file in enumerate(uploaded_excel_files):
                file_content = file.getvalue()
                file_name_base = file.name.split('.')[0]
                try:
                    excel_file = pd.ExcelFile(file_content)
                    sheet_names = excel_file.sheet_names
                    st_markdown_dual(f"**Options for '{file.name}'**")
                    if len(sheet_names) > 1:
                        load_method = st.radio(
                            f"How to load '{file_name_base}' sheets?",
                            ("Select a single sheet", "Load ALL sheets (merge)", "Select specific sheets (merge)"),
                            key=f"load_method_{file_name_base}_{i}"
                        )
                    else:
                        load_method = "Select a single sheet"
                        st_info_dual(f"'{file_name_base}' has only one sheet: '{sheet_names[0]}'. Loading it directly.")

                    if load_method == "Load ALL sheets (merge)":
                        all_sheets_dataframes = []
                        for sheet_name in sheet_names:
                            df_sheet = pd.read_excel(file_content, sheet_name=sheet_name)
                            df_sheet.replace(common_na_values, np.nan, inplace=True)
                            all_sheets_dataframes.append(df_sheet)
                        merged_df_all_sheets = pd.concat(all_sheets_dataframes, ignore_index=True)
                        temp_uploaded_dfs[f"{file_name_base}_all_sheets"] = merged_df_all_sheets
                        st_success_dual(f"Successfully loaded and merged all {len(sheet_names)} sheets from '{file_name_base}'.")
                    elif load_method == "Select specific sheets (merge)":
                        selected_sheets_for_merge = st.multiselect(
                            f"Select sheets from '{file_name_base}' to merge:",
                            sheet_names,
                            default=sheet_names,
                            key=f"multiselect_sheets_{file_name_base}_{i}"
                        )
                        if selected_sheets_for_merge:
                            specific_sheets_dataframes = []
                            for sheet_name_selected in selected_sheets_for_merge:
                                df_sheet = pd.read_excel(file_content, sheet_name=sheet_name_selected)
                                df_sheet.replace(common_na_values, np.nan, inplace=True)
                                specific_sheets_dataframes.append(df_sheet)
                            merged_specific_sheets = pd.concat(specific_sheets_dataframes, ignore_index=True)
                            temp_uploaded_dfs[f"{file_name_base}_{'_'.join(selected_sheets_for_merge)}_merged"] = merged_specific_sheets
                            st_success_dual(f"Successfully loaded and merged selected sheets from '{file_name_base}'.")
                        else:
                            st_warning_dual(f"No sheets selected for '{file_name_base}'. Skipping this file.")
                    else:
                        selected_sheet = sheet_names[0]
                        if len(sheet_names) > 1:
                            selected_sheet = st.selectbox(
                                f"Select sheet for '{file_name_base}':",
                                sheet_names,
                                key=f"sheet_selector_{file_name_base}_{i}"
                            )
                            st_info_dual(f"Loading sheet: '{selected_sheet}' from '{file_name_base}'.")
                        df_temp = pd.read_excel(file_content, sheet_name=selected_sheet)
                        df_temp.replace(common_na_values, np.nan, inplace=True)
                        temp_uploaded_dfs[f"{file_name_base}_{selected_sheet}"] = df_temp
                except Exception as e:
                    st_error_dual(f"Error loading {file.name}: {e}")

            st.session_state['uploaded_dfs'] = temp_uploaded_dfs

            if len(st.session_state['uploaded_dfs']) > 0:
                merge_all_uploaded_files = st.checkbox("Load all active uploaded DataFrames as a single merged DataFrame (if compatible)", value=False, key="merge_initial_excel_uploader")
                if merge_all_uploaded_files and len(st.session_state['uploaded_dfs']) > 1:
                    try:
                        df_list_to_concat = list(st.session_state['uploaded_dfs'].values())
                        st.session_state['df_original'] = pd.concat(df_list_to_concat, ignore_index=True)
                        st.session_state['data_source'] = "excel_merged_multi_file"
                        st_success_dual("All active uploaded Excel DataFrames concatenated into one DataFrame.")
                    except Exception as e:
                        st_warning_dual(f"Could not concatenate all active uploaded files (likely incompatible columns). Please use the 'Link DataFrames' feature for merging. Error: {e}")
                        df_keys = list(st.session_state['uploaded_dfs'].keys())
                        selected_initial_file_key = st.selectbox("Select one of the uploaded files/sheets to begin EDA:", df_keys, key="select_single_initial_excel")
                        st.session_state['df_original'] = st.session_state['uploaded_dfs'][selected_initial_file_key]
                        st.session_state['data_source'] = "excel_single_sheet"
                elif len(st.session_state['uploaded_dfs']) == 1:
                    st.session_state['df_original'] = list(st.session_state['uploaded_dfs'].values())[0]
                    st.session_state['data_source'] = "excel_single_sheet"
                else:
                    df_keys = list(st.session_state['uploaded_dfs'].keys())
                    selected_initial_file_key = st.selectbox("Select one of the uploaded files/sheets to begin EDA:", df_keys, key="select_single_initial_excel")
                    st.session_state['df_original'] = st.session_state['uploaded_dfs'][selected_initial_file_key]
                    st.session_state['data_source'] = "excel_single_sheet"
            else:
                st_info_dual("Upload one or more Excel files, or connect to a database to begin EDA.")
        else:
            st.session_state['df_original'] = None
            st.session_state['data_source'] = None
            st.session_state['uploaded_dfs'] = {}
            st_info_dual("Upload one or more Excel files, or connect to a database to begin EDA.")

    elif data_source_choice == "Connect to Database":
        st.markdown("---")
        st.markdown("### 🗄️ Connect to SQL Database")
        db_type = st.selectbox(
            "Database Type",
            ["PostgreSQL", "MySQL", "SQLite", "SQL Server", "Oracle", "Other (Provide URL)"],
            key="db_type"
        )

        if db_type == "SQLite":
            db_path = st.text_input("SQLite Database Path (e.g., my_database.db)", key="sqlite_path")
            query_text = st.text_area("SQL Query (e.g., SELECT * FROM my_table)", height=150, key="sql_query")
        else:
            db_host = st.text_input("Host", key="db_host")
            db_port = st.text_input("Port", key="db_port")
            db_name = st.text_input("Database Name", key="db_name")
            db_user = st.text_input("Username", key="db_user")
            db_password = st.text_input("Password", type="password", key="db_password")
            query_text = st.text_area("SQL Query (e.g., SELECT * FROM your_table LIMIT 1000)", height=150, key="sql_query")

        if st.button("Fetch Data from Database", key="fetch_db_data"):
            if query_text:
                try:
                    conn_string = None
                    if db_type == "SQLite":
                        if db_path:
                            conn_string = f"sqlite:///{db_path}"
                        else:
                            st_error_dual("Please provide a path for the SQLite database.")
                    elif db_type == "PostgreSQL":
                        conn_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
                    elif db_type == "MySQL":
                        conn_string = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
                    elif db_type == "SQL Server":
                        conn_string = f"mssql+pyodbc://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?driver=ODBC+Driver+17+for+SQL+Server"
                    elif db_type == "Oracle":
                        conn_string = f"oracle+cx_oracle://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
                    elif db_type == "Other (Provide URL)":
                        conn_string = st.text_input("Full SQLAlchemy Connection URL (e.g., 'postgresql://user:pass@host:port/dbname')", key="custom_db_url")

                    if conn_string:
                        with st.spinner("Connecting to database and fetching data..."):
                            engine = create_engine(conn_string)
                            fetched_df = pd.read_sql(query_text, engine)
                            st.session_state['df_original'] = fetched_df
                            st.session_state['data_source'] = "database"
                            st_success_dual("Data fetched successfully from database!")
                            st.session_state['active_filters'] = {}
                            st.session_state['viz_filters'] = {}
                    else:
                        st_warning_dual("Please provide all database connection details or a custom URL.")
                except SQLAlchemyError as e:
                    st_error_dual(f"Database connection or query failed: {e}")
                except Exception as e:
                    st_error_dual(f"An unexpected error occurred: {e}")
            else:
                st_warning_dual("Please enter an SQL query.")


# 2. Link DataFrames Expander
if len(st.session_state.get('uploaded_dfs', {})) > 1:
    with st.sidebar.expander("🔗 Link DataFrames", expanded=False):
        st.info("Functionality to link or merge multiple DataFrames is coming soon.")
        pass

# 4. Data Filtering Expander
with st.sidebar.expander("🔍 Filter Data", expanded=False):
    if st.session_state.get('df_original') is not None:
        df_for_filter = st.session_state['df_original']
        filter_cols = st.multiselect("Select columns to filter:", df_for_filter.columns.tolist(), key="filter_cols")
        
        for col in filter_cols:
            if pd.api.types.is_numeric_dtype(df_for_filter[col]):
                min_val, max_val = float(df_for_filter[col].min()), float(df_for_filter[col].max())
                slider_val = st.slider(f"Range for {col}", min_val, max_val, (min_val, max_val), key=f"filter_{col}")
                st.session_state['active_filters'][col] = slider_val
            elif pd.api.types.is_datetime64_any_dtype(df_for_filter[col]):
                min_date, max_date = df_for_filter[col].min(), df_for_filter[col].max()
                date_val = st.date_input(f"Date range for {col}", value=(min_date, max_date), key=f"filter_{col}")
                if len(date_val) == 2:
                    st.session_state['active_filters'][col] = date_val
            else:
                unique_vals = df_for_filter[col].unique().tolist()
                ms_val = st.multiselect(f"Values for {col}", unique_vals, default=unique_vals, key=f"filter_{col}")
                st.session_state['active_filters'][col] = ms_val

        if st.button("Apply Filters", key="apply_filters"):
            st.rerun()
        if st.button("Clear Filters", key="clear_filters"):
            st.session_state['active_filters'] = {}
            st.rerun()

# --- Global Dataframe Preparation ---
df = None
if st.session_state.get('df_original') is not None:
    df = st.session_state['df_original'].copy()
    # Apply active filters
    if st.session_state.get('active_filters'):
        for col, value in st.session_state['active_filters'].items():
            if col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df = df[(df[col] >= value[0]) & (df[col] <= value[1])]
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    start_date, end_date = pd.to_datetime(value[0]), pd.to_datetime(value[1])
                    df = df[(df[col] >= start_date) & (df[col] <= end_date)]
                else:
                    df = df[df[col].isin(value)]
if df is not None and st.session_state.get('dataset_summary') is None:
    analyze_and_store_data_details(df)
    # Rerun the script immediately after analysis to update the page
    st.rerun()                    

# --- Page Views ---

def landing_page_view():
    st.title("🚀 Welcome to DataSense AI")
    st.markdown("Your dataset has been loaded. Here's a quick overview:")

    if st.session_state.dataset_summary and st.session_state.data_metrics:
        st.subheader("🤖 AI-Generated Dataset Description")
        st.info(st.session_state.dataset_summary)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Data Dimensions")
            st.metric("Number of Rows", st.session_state.data_metrics['rows'])
            st.metric("Number of Columns", st.session_state.data_metrics['columns'])
        
        with col2:
            st.subheader("Missing Values")
            missing_df = st.session_state.data_metrics['missing_info_df']
            if not missing_df.empty:
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("✅ No missing values found in the dataset.")
        
        st.markdown("<hr>", unsafe_allow_html=True)
        st.header("Select a Feature to Get Started")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            with st.container():
                st.markdown("""<div class="nav-box"><h3>📄 Data Profile</h3><p>View the raw data, summary statistics, and column details.</p></div>""", unsafe_allow_html=True)
                if st.button("Explore Data Profile", key="nav_profile", use_container_width=True):
                    st.session_state.page = 'data_profile'
                    st.rerun()
        with col2:
            with st.container():
                st.markdown("""<div class="nav-box"><h3>📈 Visualisation Engine</h3><p>Create interactive charts and graphs to explore your data visually.</p></div>""", unsafe_allow_html=True)
                if st.button("Launch Visualisation", key="nav_viz", use_container_width=True):
                    st.session_state.page = 'visualisation'
                    st.rerun()
        with col3:
            with st.container():
                st.markdown("""<div class="nav-box"><h3>🧠 AI Analysis</h3><p>Ask questions, get cleaning suggestions, and use the AI sandbox.</p></div>""", unsafe_allow_html=True)
                if st.button("Get AI Insights", key="nav_ai", use_container_width=True):
                    st.session_state.page = 'ai_analysis'
                    st.rerun()
        with col4:
            with st.container():
                st.markdown("""<div class="nav-box"><h3>🛠️ Smart Features</h3><p>Use the Pivot Table and run Hypothesis Tests for advanced analysis.</p></div>""", unsafe_allow_html=True)
                if st.button("Use Smart Features", key="nav_smart", use_container_width=True):
                    st.session_state.page = 'smart_features'
                    st.rerun()
    else:
        st.info("Waiting for data analysis to complete...")

def data_profile_view():
    st.subheader("📄 Original Data Profile")
    if df is not None:
        st_dataframe_dual(df)
        st_markdown_dual("### Summary Statistics")
        summary_data = get_data_summary(df)
        with st.expander("Dataset Shape & Column Types", expanded=True):
            st_write_dual(summary_data['shape_info'])
            st_dataframe_dual(summary_data['column_types_df'], use_container_width=True)
        with st.expander("Missing Values Overview"):
            st_dataframe_dual(summary_data['missing_values_df'], use_container_width=True)
        with st.expander("Unique Values per Column"):
            st_dataframe_dual(summary_data['unique_values_df'], use_container_width=True)
        with st.expander("Numeric Summary"):
            st_dataframe_dual(df.describe().transpose(), use_container_width=True)
    else:
        st.warning("No data loaded.")

def ask_ai_about_chart(chart_description, chart_title, df_context, key_suffix):
    """
    A reusable component to ask an AI model about a generated chart.
    """
    st.markdown("---")
    st.markdown("#### 🤔 Ask AI about this Chart")

    # Use session state to store the AI's response for this specific chart query
    session_key = f"viz_ai_response_{key_suffix}"
    if session_key not in st.session_state:
        st.session_state[session_key] = ""

    ai_model = st.radio(
        "Choose AI Model:",
        ("Google Gemini", "Perplexity AI"),
        key=f"viz_ai_model_{key_suffix}",
        horizontal=True
    )
    question = st.text_area(
        "Ask a question about this chart:",
        key=f"viz_ai_question_{key_suffix}",
        height=80
    )

    if st.button("Get AI Insight", key=f"viz_ai_submit_{key_suffix}"):
        if question:
            with st.spinner(f"Asking {ai_model} to analyze the '{chart_title}' chart..."):
                # Construct a detailed prompt for the AI
                system_prompt = (
                    "You are an expert data visualization analyst. "
                    "A user has generated a chart and is asking a question about it. "
                    "Your task is to analyze the chart's description and the user's question, "
                    "then provide a clear, insightful interpretation based on the chart's context. "
                    "Do not just state the obvious; explain what the patterns or values *mean*."
                )
                user_prompt = (
                    f"I have created a '{chart_title}'.\n\n"
                    f"**Chart Description:** {chart_description}\n\n"
                    f"**My Question:** {question}\n\n"
                    f"Please provide your analysis of the chart based on my question."
                )

                response = ""
                if ai_model == "Google Gemini":
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                    response = chat_bot(messages)
                elif ai_model == "Perplexity AI":
                    # Perplexity chat function takes the full dataframe for context
                    response = perplexity_chat(question=user_prompt, df=df_context, system_message=system_prompt)
                
                st.session_state[session_key] = response
        else:
            st.warning("Please enter a question before submitting.")
    
    if st.session_state[session_key]:
        st.markdown("##### 🤖 AI Interpretation:")
        st.success(st.session_state[session_key])


def visualisation_view():
    st.subheader("📈 Data Visualization Engine")
    if df is not None and not df.empty:
        chart_options = [
            "Histogram", "Bar", "Line", "Scatter", "Boxplot", "Pie Chart", 
            "Cluster Chart", "Stacked Bar Chart", "Correlation Heatmap", "Countplot"
        ]
        selected_charts = st.multiselect(
            "Select chart types to display:", 
            chart_options, 
            key="chart_selector"
        )
        
        st.markdown("---")
        st.markdown("#### Apply Filters for Visualization (Visualization Specific Filters)")

        df_for_viz_filter = df.copy()
        current_viz_available_columns = df_for_viz_filter.columns.tolist()

        viz_filter_columns_selection = st.multiselect(
            "Select columns to apply visualization-specific filters:",
            current_viz_available_columns,
            default=list(st.session_state['viz_filters'].keys()),
            key="viz_filter_cols_selection_main"
        )

        for col_name in viz_filter_columns_selection:
            if col_name in df_for_viz_filter.columns:
                default_filter_value = st.session_state['viz_filters'].get(col_name)
                if pd.api.types.is_numeric_dtype(df_for_viz_filter[col_name]):
                    min_val_viz = float(df_for_viz_filter[col_name].min())
                    max_val_viz = float(df_for_viz_filter[col_name].max())
                    current_value_viz = default_filter_value if (default_filter_value and isinstance(default_filter_value, tuple) and len(default_filter_value) == 2) else ((min_val_viz, max_val_viz) if min_val_viz != max_val_viz else (min_val_viz, max_val_viz + 1.0))
                    if min_val_viz == max_val_viz:
                        range_val_viz = st.slider(f"Filter {col_name} (Value)", min_value=min_val_viz, max_value=max_val_viz + 1.0, value=current_value_viz, key=f"viz_filter_range_{col_name}")
                    else:
                        range_val_viz = st.slider(f"Filter {col_name} (Range)", min_value=min_val_viz, max_value=max_val_viz, value=current_value_viz, key=f"viz_filter_range_{col_name}")
                    st.session_state['viz_filters'][col_name] = range_val_viz
                    df_for_viz_filter = df_for_viz_filter[(df_for_viz_filter[col_name] >= range_val_viz[0]) & (df_for_viz_filter[col_name] <= range_val_viz[1])]
                elif pd.api.types.is_datetime64_any_dtype(df_for_viz_filter[col_name]):
                    temp_series_viz = pd.to_datetime(df_for_viz_filter[col_name], errors='coerce').dropna()
                    if not temp_series_viz.empty:
                        min_date_viz = temp_series_viz.min()
                        max_date_viz = temp_series_viz.max()
                        current_date_value_viz = default_filter_value if (default_filter_value and isinstance(default_filter_value, tuple) and len(default_filter_value) == 2) else (min_date_viz, max_date_viz)
                        date_range_viz = st.date_input(f"Filter {col_name} (Date Range)", value=current_date_value_viz, min_value=min_date_viz, max_value=max_date_viz, key=f"viz_filter_date_range_{col_name}")
                        if len(date_range_viz) == 2:
                            st.session_state['viz_filters'][col_name] = (pd.to_datetime(date_range_viz[0]), pd.to_datetime(date_range_viz[1]))
                            df_for_viz_filter = df_for_viz_filter[(df_for_viz_filter[col_name] >= pd.to_datetime(date_range_viz[0])) & (df_for_viz_filter[col_name] <= pd.to_datetime(date_range_viz[1]))]
                        else:
                            st_info_dual(f"Please select both start and end dates for {col_name}.")
                            st.session_state['viz_filters'].pop(col_name, None)
                    else:
                        st_info_dual(f"No valid date values for filtering in '{col_name}'.")
                        st.session_state['viz_filters'].pop(col_name, None)
                else:
                    unique_vals_viz = df_for_viz_filter[col_name].dropna().unique().tolist()
                    current_selected_vals_viz = default_filter_value if (default_filter_value and isinstance(default_filter_value, list)) else unique_vals_viz
                    selected_vals_viz = st.multiselect(f"Select values for {col_name}", unique_vals_viz, default=current_selected_vals_viz, key=f"viz_filter_multiselect_{col_name}")
                    st.session_state['viz_filters'][col_name] = selected_vals_viz
                    if selected_vals_viz:
                        df_for_viz_filter = df_for_viz_filter[df_for_viz_filter[col_name].isin(selected_vals_viz)]
                    else:
                        df_for_viz_filter = df_for_viz_filter[df_for_viz_filter[col_name].isin([])]
            else:
                st_warning_dual(f"Column '{col_name}' not found in the current dataset for visualization filtering.")

        if st.button("Clear Visualization Filters", key="clear_viz_filters_btn_tab3"):
            st.session_state['viz_filters'] = {}
            st.rerun()

        st.markdown("---")

        if df_for_viz_filter.empty:
            st_warning_dual("No data available for visualization after applying current filters.")
        else:
            
            viz_numeric_cols = df_for_viz_filter.select_dtypes(include=np.number).columns.tolist()
            viz_categorical_cols = df_for_viz_filter.select_dtypes(include=['object', 'category']).columns.tolist()

            if "Histogram" in selected_charts:
                st.markdown("### Histogram")
                st.info("A histogram shows the distribution of a single numeric variable. It helps you see the frequency of values, the central tendency, and the spread.")
                if viz_numeric_cols:
                    num_col_hist = st.selectbox("Select a numeric column for Histogram", ["None"] + viz_numeric_cols, key="hist_col_viz")
                    if num_col_hist != "None":
                        fig_hist = px.histogram(df_for_viz_filter, x=num_col_hist, title=f'Distribution of {num_col_hist}')
                        plotly_chart_dual(fig_hist, use_container_width=True)
                        chart_desc = f"This is a histogram showing the frequency distribution for the numeric column '{num_col_hist}'."
                        ask_ai_about_chart(chart_desc, f'Distribution of {num_col_hist}', df_for_viz_filter, f"hist_{num_col_hist}")
                else:
                    st_info_dual("No numeric columns available for histogram.")

            if "Bar" in selected_charts:
                st.markdown("### Bar Chart")
                st.info("A bar chart is used to compare values across different categories. It can show counts of categories or an aggregated numeric value for each category.")
                if viz_categorical_cols:
                    cat_col_bar = st.selectbox("Select a categorical column for Bar Chart", ["None"] + viz_categorical_cols, key="bar_cat_col_viz")
                    if cat_col_bar != "None":
                        num_col_bar_options = ["None"] + viz_numeric_cols
                        num_col_bar = st.selectbox("Select a numeric column for Bar Chart (optional, for aggregation)", num_col_bar_options, key="bar_num_col_viz")
                        chart_title = ""
                        chart_desc = ""
                        if num_col_bar != "None":
                            agg_func_bar = st.selectbox(
                                "Select Aggregation Function",
                                ["sum", "mean", "median", "min", "max"],
                                key="bar_agg_func_viz"
                            )
                            try:
                                bar_data = df_for_viz_filter.groupby(cat_col_bar, as_index=False).agg({num_col_bar: agg_func_bar})
                                chart_title = f'{agg_func_bar.capitalize()} of {num_col_bar} by {cat_col_bar}'
                                fig_bar = px.bar(bar_data, x=cat_col_bar, y=num_col_bar, title=chart_title)
                                plotly_chart_dual(fig_bar, use_container_width=True)
                                chart_desc = f"This bar chart displays the '{agg_func_bar}' of the numeric column '{num_col_bar}' for each category in '{cat_col_bar}'."
                                ask_ai_about_chart(chart_desc, chart_title, df_for_viz_filter, f"bar_{cat_col_bar}_{num_col_bar}")
                            except Exception as e:
                                st_error_dual(f"Could not generate aggregated bar chart. Error: {e}")
                        else:
                            bar_data = df_for_viz_filter[cat_col_bar].value_counts().reset_index()
                            bar_data.columns = [cat_col_bar, 'Count']
                            chart_title = f'Count of {cat_col_bar}'
                            fig_bar = px.bar(bar_data, x=cat_col_bar, y='Count', title=chart_title)
                            plotly_chart_dual(fig_bar, use_container_width=True)
                            chart_desc = f"This bar chart shows the count of occurrences for each category in the '{cat_col_bar}' column."
                            ask_ai_about_chart(chart_desc, chart_title, df_for_viz_filter, f"bar_count_{cat_col_bar}")
                else:
                    st_info_dual("No categorical columns available for bar chart.")

            if "Line" in selected_charts:
                st.markdown("### Line Chart")
                st.info("A line chart is ideal for showing trends over time or the relationship between two ordered numeric variables.")
                datetime_cols_viz = df_for_viz_filter.select_dtypes(include='datetime64[ns]').columns.tolist()
                numeric_cols_viz = df_for_viz_filter.select_dtypes(include=np.number).columns.tolist()
                potential_x_cols = list(dict.fromkeys(datetime_cols_viz + numeric_cols_viz))

                if potential_x_cols and viz_numeric_cols:
                    x_line_col = st.selectbox("Select a column for X-axis (Date or Numeric)", ["None"] + potential_x_cols, key="line_x_col_viz")
                    y_line_col = st.selectbox("Select a column for Y-axis (Numeric)", ["None"] + viz_numeric_cols, key="line_y_col_viz")
                    color_line_col = st.selectbox("Group by color (optional, categorical)", ["None"] + viz_categorical_cols, key="line_color_col_viz")
                    
                    agg_func_line = st.selectbox(
                        "Select Aggregation Function (optional)",
                        ["None", "mean", "sum", "median", "min", "max"],
                        key="line_agg_func_viz",
                        help="Select 'None' to plot raw data points. Otherwise, data will be grouped by X-axis (and color) and aggregated."
                    )

                    if x_line_col != "None" and y_line_col != "None":
                        try:
                            df_for_plot = df_for_viz_filter.copy()
                            fig_title = f'Line Chart of {y_line_col} over {x_line_col}'
                            chart_desc = ""

                            if agg_func_line != "None":
                                grouping_cols = [x_line_col]
                                if color_line_col != "None":
                                    grouping_cols.append(color_line_col)
                                
                                df_for_plot = df_for_plot.groupby(grouping_cols, as_index=False).agg({y_line_col: agg_func_line})
                                fig_title = f'{agg_func_line.capitalize()} of {y_line_col} over {x_line_col}'
                                chart_desc = f"This is an aggregated line chart showing the '{agg_func_line}' of '{y_line_col}' over '{x_line_col}'"
                            else:
                                chart_desc = f"This is a line chart plotting individual data points for '{y_line_col}' against '{x_line_col}'"

                            if color_line_col != "None":
                                chart_desc += f", with lines colored by the '{color_line_col}' category."
                            
                            df_for_plot = df_for_plot.sort_values(by=x_line_col)
                            fig_line = px.line(df_for_plot, x=x_line_col, y=y_line_col, color=color_line_col if color_line_col != "None" else None, title=fig_title)
                            
                            if pd.api.types.is_datetime64_any_dtype(df_for_plot[x_line_col]):
                                fig_line.update_xaxes(
                                    rangeslider_visible=True,
                                    rangeselector=dict(
                                        buttons=list([
                                            dict(count=1, label="1m", step="month", stepmode="backward"),
                                            dict(count=6, label="6m", step="month", stepmode="backward"),
                                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                                            dict(count=1, label="1y", step="year", stepmode="backward"),
                                            dict(step="all")
                                        ])
                                    )
                                )
                            plotly_chart_dual(fig_line, use_container_width=True)
                            ask_ai_about_chart(chart_desc, fig_title, df_for_viz_filter, f"line_{x_line_col}_{y_line_col}")

                        except Exception as e:
                            st_error_dual(f"Could not generate line chart. Error: {e}")
                else:
                    st_warning_dual("A line chart requires at least one numeric column for the Y-axis and one numeric or date column for the X-axis.")

            if "Scatter" in selected_charts:
                st.markdown("### Scatter Plot")
                st.info("A scatter plot is used to observe the relationship between two numeric variables. It's excellent for identifying correlations, trends, and outliers.")
                if viz_numeric_cols:
                    x_scatter_col = st.selectbox("Select X-axis column", ["None"] + viz_numeric_cols, key="scatter_x_col_viz")
                    y_scatter_col = st.selectbox("Select Y-axis column", ["None"] + viz_numeric_cols, key="scatter_y_col_viz")
                    color_scatter_col = st.selectbox("Select a column for Color (optional)", ["None"] + viz_categorical_cols, key="scatter_color_col_viz")
                    size_scatter_col = st.selectbox("Select a column for Size (optional)", ["None"] + viz_numeric_cols, key="scatter_size_col_viz")
                    if x_scatter_col != "None" and y_scatter_col != "None":
                        fig_title = f'{y_scatter_col} vs {x_scatter_col}'
                        fig_scatter = px.scatter(df_for_viz_filter, x=x_scatter_col, y=y_scatter_col,
                                                color=color_scatter_col if color_scatter_col != "None" else None,
                                                size=size_scatter_col if size_scatter_col != "None" else None,
                                                title=fig_title)
                        plotly_chart_dual(fig_scatter, use_container_width=True)
                        chart_desc = f"This scatter plot shows the relationship between '{x_scatter_col}' (X-axis) and '{y_scatter_col}' (Y-axis)."
                        if color_scatter_col != "None": chart_desc += f" Points are colored by the '{color_scatter_col}' category."
                        if size_scatter_col != "None": chart_desc += f" The size of each point is determined by the '{size_scatter_col}' value."
                        ask_ai_about_chart(chart_desc, fig_title, df_for_viz_filter, f"scatter_{x_scatter_col}_{y_scatter_col}")
                else:
                    st_info_dual("Requires numeric columns for a scatter plot.")

            if "Boxplot" in selected_charts:
                st.markdown("### Box Plot")
                st.info("A box plot displays the distribution of a numeric variable, showing the median, quartiles, and potential outliers. It's great for comparing distributions across categories.")
                if viz_numeric_cols:
                    y_boxplot_col = st.selectbox("Select a numeric column for Box Plot", ["None"] + viz_numeric_cols, key="boxplot_y_col_viz")
                    x_boxplot_col = st.selectbox("Select a categorical column for X-axis (optional)", ["None"] + viz_categorical_cols, key="boxplot_x_col_viz")
                    if y_boxplot_col != "None":
                        fig_title = f'Box Plot of {y_boxplot_col}'
                        chart_desc = f"This box plot shows the distribution of the numeric column '{y_boxplot_col}'."
                        if x_boxplot_col != "None":
                            fig_title += f' by {x_boxplot_col}'
                            chart_desc += f" The distributions are grouped by the categories in '{x_boxplot_col}'."
                        fig_box = px.box(df_for_viz_filter, y=y_boxplot_col, x=x_boxplot_col if x_boxplot_col != "None" else None, title=fig_title)
                        plotly_chart_dual(fig_box, use_container_width=True)
                        ask_ai_about_chart(chart_desc, fig_title, df_for_viz_filter, f"boxplot_{y_boxplot_col}_{x_boxplot_col}")
                else:
                    st_info_dual("Requires numeric columns for a box plot.")

            if "Pie Chart" in selected_charts:
                st.markdown("### Pie Chart")
                st.info("A pie chart shows the proportion of categories relative to the whole. It's best used for a small number of categories that sum to 100%.")
                if viz_categorical_cols:
                    names_pie_col = st.selectbox("Select a categorical column for Pie Chart slices", ["None"] + viz_categorical_cols, key="pie_names_col_viz")
                    if names_pie_col != "None":
                        values_pie_col = st.selectbox("Select a numeric column for Pie Chart values (optional)", ["None"] + viz_numeric_cols, key="pie_values_col_viz")
                        chart_title = ""
                        chart_desc = ""
                        if values_pie_col != "None":
                            agg_func_pie = st.selectbox(
                                "Select Aggregation Function for Values",
                                ["sum", "mean", "median"],
                                key="pie_agg_func_viz"
                            )
                            try:
                                pie_data = df_for_viz_filter.groupby(names_pie_col, as_index=False).agg({values_pie_col: agg_func_pie})
                                chart_title = f'{agg_func_pie.capitalize()} of {values_pie_col} by {names_pie_col}'
                                fig_pie = px.pie(pie_data, names=names_pie_col, values=values_pie_col, title=chart_title)
                                plotly_chart_dual(fig_pie, use_container_width=True)
                                chart_desc = f"This pie chart shows the proportion of '{names_pie_col}' based on the '{agg_func_pie}' of '{values_pie_col}'."
                                ask_ai_about_chart(chart_desc, chart_title, df_for_viz_filter, f"pie_agg_{names_pie_col}_{values_pie_col}")
                            except Exception as e:
                                st_error_dual(f"Could not generate aggregated pie chart. Error: {e}")
                        else:
                            chart_title = f'Proportion of {names_pie_col}'
                            fig_pie = px.pie(df_for_viz_filter, names=names_pie_col, title=chart_title)
                            plotly_chart_dual(fig_pie, use_container_width=True)
                            chart_desc = f"This pie chart shows the proportion of each category in the '{names_pie_col}' column based on their counts."
                            ask_ai_about_chart(chart_desc, chart_title, df_for_viz_filter, f"pie_count_{names_pie_col}")
                else:
                    st_info_dual("Requires categorical columns for a pie chart.")

            if "Cluster Chart" in selected_charts:
                st.markdown("### Cluster Chart (Scatter Plot with Color/Size)")
                st.info("A cluster chart is a type of scatter plot that uses color to group data points into clusters, helping to identify distinct groups within the data.")
                if len(viz_numeric_cols) >= 2:
                    x_cluster_col = st.selectbox("Select X-axis column", ["None"] + viz_numeric_cols, key="cluster_x_col_viz")
                    y_cluster_col = st.selectbox("Select Y-axis column", ["None"] + viz_numeric_cols, key="cluster_y_col_viz")
                    color_cluster_col = st.selectbox("Select a column for Color/Clusters (categorical)", ["None"] + viz_categorical_cols, key="cluster_color_col_viz")
                    size_cluster_col = st.selectbox("Select a column for Size (numeric, optional)", ["None"] + viz_numeric_cols, key="cluster_size_col_viz")
                    if x_cluster_col != "None" and y_cluster_col != "None" and color_cluster_col != "None":
                        fig_title = f'Cluster of {y_cluster_col} vs {x_cluster_col} by {color_cluster_col}'
                        fig_cluster = px.scatter(df_for_viz_filter, x=x_cluster_col, y=y_cluster_col, color=color_cluster_col,
                                                size=size_cluster_col if size_cluster_col != "None" else None,
                                                title=fig_title)
                        plotly_chart_dual(fig_cluster, use_container_width=True)
                        chart_desc = f"This is a cluster chart (a scatter plot) showing the relationship between '{x_cluster_col}' and '{y_cluster_col}', with points clustered by the color of the '{color_cluster_col}' category."
                        ask_ai_about_chart(chart_desc, fig_title, df_for_viz_filter, f"cluster_{x_cluster_col}_{y_cluster_col}")
                else:
                    st_info_dual("Requires at least two numeric and one categorical column for a cluster chart.")

            if "Stacked Bar Chart" in selected_charts:
                st.markdown("### Stacked Bar Chart")
                st.info("A stacked bar chart shows how a larger category is divided into smaller sub-categories and what the relationship of each part has on the total amount.")
                if len(viz_categorical_cols) >= 2 and viz_numeric_cols:
                    x_stack_col = st.selectbox("Select X-axis (Categorical)", ["None"] + viz_categorical_cols, key="stack_x_col_viz")
                    y_stack_col = st.selectbox("Select Y-axis (Numeric, for values)", ["None"] + viz_numeric_cols, key="stack_y_col_viz")
                    
                    available_color_stack_cols = [col for col in viz_categorical_cols if col != x_stack_col]
                    
                    if available_color_stack_cols:
                        color_stack_col = st.selectbox("Select Color/Stack (Categorical)", ["None"] + available_color_stack_cols, key="stack_color_col_viz")
                        if x_stack_col != "None" and y_stack_col != "None" and color_stack_col != "None":
                            agg_func_stack = st.selectbox(
                                "Select Aggregation Function for Y-axis",
                                ["sum", "mean", "median", "min", "max"],
                                key="stack_agg_func_viz"
                            )
                            try:
                                stack_data = df_for_viz_filter.groupby([x_stack_col, color_stack_col], as_index=False).agg({y_stack_col: agg_func_stack})
                                fig_title = f'{agg_func_stack.capitalize()} of {y_stack_col} by {x_stack_col} stacked by {color_stack_col}'
                                fig_stacked_bar = px.bar(stack_data, x=x_stack_col, y=y_stack_col, color=color_stack_col, title=fig_title)
                                plotly_chart_dual(fig_stacked_bar, use_container_width=True)
                                chart_desc = f"This stacked bar chart shows the '{agg_func_stack}' of '{y_stack_col}' for each category in '{x_stack_col}', with bars segmented by the '{color_stack_col}' category."
                                ask_ai_about_chart(chart_desc, fig_title, df_for_viz_filter, f"stacked_{x_stack_col}_{y_stack_col}")
                            except Exception as e:
                                st_error_dual(f"Could not generate aggregated stacked bar chart. Error: {e}")
                    else:
                        st_info_dual("A different categorical column is needed for the color/stack option.")

                else:
                    st_info_dual("Requires at least two categorical and one numeric column for a stacked bar chart.")

            if "Correlation Heatmap" in selected_charts:
                st.markdown("### Correlation Heatmap")
                st.info("A correlation heatmap visually represents the correlation matrix between all numeric columns. It helps to quickly identify which variables are related.")
                if len(viz_numeric_cols) >= 2:
                    corr_matrix = df_for_viz_filter[viz_numeric_cols].corr()
                    fig_title = "Correlation Heatmap of Numeric Columns"
                    fig_heatmap = px.imshow(corr_matrix, text_auto=True, aspect="auto", title=fig_title, color_continuous_scale=px.colors.sequential.Viridis)
                    plotly_chart_dual(fig_heatmap, use_container_width=True)
                    chart_desc = f"This is a correlation heatmap for the numeric columns in the dataset. It shows the Pearson correlation coefficient between each pair of variables. Values close to 1 or -1 indicate a strong linear relationship, while values close to 0 indicate a weak one."
                    ask_ai_about_chart(chart_desc, fig_title, df_for_viz_filter, "corr_heatmap")
                else:
                    st_info_dual("Need at least two numeric columns to display a correlation heatmap.")

            if "Countplot" in selected_charts:
                st.markdown("### Count Plot")
                st.info("A count plot is like a histogram but for a categorical variable. It shows the number of occurrences of each category.")
                if viz_categorical_cols:
                    cat_col_count = st.selectbox("Select a categorical column for Count Plot", ["None"] + viz_categorical_cols, key="count_cat_col_viz")
                    if cat_col_count != "None":
                        fig_title = f'Count of {cat_col_count}'
                        fig_count = px.histogram(df_for_viz_filter, x=cat_col_count, title=fig_title, color_discrete_sequence=px.colors.qualitative.Pastel)
                        plotly_chart_dual(fig_count, use_container_width=True)
                        chart_desc = f"This is a count plot showing the number of occurrences for each category within the '{cat_col_count}' column."
                        ask_ai_about_chart(chart_desc, fig_title, df_for_viz_filter, f"countplot_{cat_col_count}")
                else:
                    st_info_dual("No categorical columns available for count plot.")
    else:
        st_info_dual("Please upload data and ensure it's not empty after general filtering/cleaning to enable visualization.")


# MODIFIED: This function has been restructured with st.columns to add a column selector on the right.
def ai_analysis_view():
    st.subheader("🧠 AI Analysis")

    # Create two columns: a main content area and a right sidebar for column selection
    main_col, side_col = st.columns([3, 1])

    with side_col:
        st.markdown("#### Select Columns to View")
        if df is not None and not df.empty:
            # Use an expander for better layout
            with st.expander("Column Selector", expanded=True):
                selected_cols = st.multiselect(
                    "Choose columns to display for reference:",
                    options=df.columns.tolist(),
                    key="ai_view_column_selector"
                )
                if selected_cols:
                    st.markdown("---")
                    st.markdown("##### Data for Selected Columns")
                    st.dataframe(df[selected_cols], use_container_width=True)
                else:
                    st.info("Select one or more columns to see their data here.")
        else:
            st.warning("No data loaded.")

    # All AI tools will now be in the main (left) column
    with main_col:
        tab_titles = ["Q&A", "AI Cleaning Suggestions", "Custom AI Data Preparation", "AI Sandbox"]
        tab4_1, tab4_2, tab4_3, tab4_sandbox = st.tabs(tab_titles)

        with tab4_1:
            st.markdown("### ❓ Ask AI about your Data")
            st.info("This feature allows you to ask questions about your dataset in natural language. The AI will provide answers based on the data context.")

            # --- MODIFICATION 1: Replaced st.radio with st.select_slider ---
            model_options = ["Google Gemini (Internet Access)", "Perplexity AI (Internet Access)", "Mistral (Local llama)", "Llama 3.2 (Local llama)"]
            ai_model_choice = st.selectbox(
                "Slide to choose your AI Model:",
                options=model_options,
                key="ai_model_slider"
            )
            st.markdown(f"**Selected Model:** `{ai_model_choice}`") # To make the selection clear

            # --- MODIFICATION 2: Added a default prompt selector ---
            default_prompts = [
                "Write your own question...",
                "Generate a 1 page detailed summary about the dataset",
                "Which columns have missing values and how many?",
                "Identify potential outliers in the most important numeric column.",
                "Suggest a good visualization for this dataset."
            ]
            selected_prompt = st.selectbox(
                "Select a default prompt or write your own:",
                options=default_prompts,
                index=0
            )
            initial_question = "" if selected_prompt == "Write your own question..." else selected_prompt
            user_question = st.text_area(
                "Ask your question here:",
                value=initial_question,
                key="ai_question",
                height=100
            )

            if st.button("Get AI Answer", key="get_ai_answer_btn"):
                if user_question:
                    with st.spinner(f"Getting answer from {ai_model_choice}..."):
                        context_for_ai = ""
                        if df is not None and not df.empty:
                            summary_data_for_ai = get_data_summary(df)
                            context_for_ai = f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n"
                            context_for_ai += "Column Types:\n" + summary_data_for_ai['column_types_df'].to_markdown(index=False) + "\n\n"
                            context_for_ai += "Missing Values (Columns with >0 missing):\n"
                            if not summary_data_for_ai['missing_values_df'].empty:
                                context_for_ai += summary_data_for_ai['missing_values_df'].to_markdown(index=False) + "\n\n"
                            else:
                                context_for_ai += "No missing values.\n\n"
                            context_for_ai += "Unique Values per Column:\n" + summary_data_for_ai['unique_values_df'].to_markdown(index=False) + "\n\n"
                            numeric_desc_for_ai = df.select_dtypes(include=np.number).describe().transpose()
                            if not numeric_desc_for_ai.empty:
                                context_for_ai += "Numeric Summary:\n" + numeric_desc_for_ai.to_markdown() + "\n\n"
                            else:
                                context_for_ai += "No numeric columns to summarize.\n\n"
                        else:
                            st_warning_dual("No data loaded to ask questions about.")
                            
                        if ai_model_choice == "Google Gemini (Internet Access)":
                            messages = [
                                {"role": "system", "content": "You are an expert data analyst. Provide insightful observations and answers based on the provided dataset context. Do NOT generate code."},
                                {"role": "user", "content": f"Here is a summary of my dataset:\n{context_for_ai}\n\nMy question is: {user_question}"}
                            ]
                            gemini_response = chat_bot(messages)
                            st_write_dual(gemini_response)
                        elif ai_model_choice == "Perplexity AI (Internet Access)":
                            pplx_response = perplexity_chat(user_question, df)
                            st_write_dual(pplx_response)
                        elif ai_model_choice == "Mistral (Local llama)":
                            mistral_response = mistral_chat(user_question, context_for_ai)
                            st_write_dual(mistral_response)
                        elif ai_model_choice == "Llama 3.2 (Local llama)":
                            llama_response = llama3_chat(user_question, context_for_ai)
                            st_write_dual(llama_response)
                        
                else:
                    st_warning_dual("Please enter a question to get AI insights.")
            
        with tab4_2:
            st.markdown("### ✨ AI Cleaning Suggestions (Powered by Gemini)")
            st.info("This feature uses AI to analyze your dataset and suggest data cleaning operations, such as handling missing values or correcting data types.")
            if df is not None and not df.empty:
                auto_eda_context = f"Current Data Columns and Types:\n{df.dtypes.to_markdown()}\n\n"
                auto_eda_context += f"Missing Values:\n{df.isnull().sum()[df.isnull().sum() > 0].to_frame('Missing Count').to_markdown()}\n\n"
                auto_eda_context += f"Numerical Column Statistics:\n{df.select_dtypes(include=np.number).describe().to_markdown()}\n\n"
                auto_eda_context += f"Categorical Column Unique Counts & Top Values:\n"
                for col in df.select_dtypes(include=['object', 'category', 'bool']).columns:
                    auto_eda_context += f"- {col}: Unique Count = {df[col].nunique()}, Top 5 values = {df[col].value_counts().head(5).index.tolist()}\n"
                auto_eda_context += f"\nFirst 5 rows:\n{df.head().to_markdown(index=False)}"

                auto_eda_system_instruction = (
                    "You are an expert data cleaning and preprocessing specialist. "
                    "Analyze the provided DataFrame summary for data quality issues. "
                    "Suggest specific, actionable data cleaning steps for each identified issue. "
                    "Focus on common issues like missing values, incorrect data types, and potentially sparse columns. "
                    "Format your suggestions as a numbered list. For each suggestion, clearly state the column, the action, and any necessary details. "
                    "**Strictly follow this format for each suggestion:** "
                    "**[Column]: [Action] ([Details])**\n\n"
                    "Example:\n"
                    "1. Age: Impute missing values (with median)\n"
                    "2. Trip_ID: Drop rows (where Driver_ID is null)\n"
                    "3. Price: Convert data type (to float)\n"
                    "4. Notes: Drop column (due to high sparsity > 90% missing)\n"
                    "5. Created_At: Convert to datetime (format '%Y-%m-%d %H:%M:%S')"
                )
                if st.button("Get Cleaning Suggestions (Gemini)", key="get_cleaning_suggestions_btn"):
                    with st.spinner("Analyzing data and generating cleaning suggestions..."):
                        llm_prompt = f"Please analyze this dataset and suggest cleaning steps:\n\n{auto_eda_context}"
                        messages_for_cleaning = [
                            {"role": "system", "content": auto_eda_system_instruction},
                            {"role": "user", "content": llm_prompt}
                        ]
                        llm_response = chat_bot(messages_for_cleaning)
                        st.session_state['llm_eda_suggestions'] = []
                        parsed_suggestions = []
                        suggestion_pattern = re.compile(r"^\d+\.\s*(?P<column>[^:]+):\s*(?P<action>[^\(]+)\s*\((?P<details>[^\)]+)\)", re.MULTILINE)
                        for line in llm_response.split('\n'):
                            match = suggestion_pattern.match(line.strip())
                            if match:
                                suggestion_dict = match.groupdict()
                                parsed_suggestions.append(suggestion_dict)
                        if parsed_suggestions:
                            st.session_state['llm_eda_suggestions'] = parsed_suggestions
                            st_success_dual("Suggestions generated! Review and apply below.")
                        else:
                            st_warning_dual("Could not parse cleaning suggestions from AI. Please try again or refine your data.")
                            st.text_area("Raw AI Response (for debugging):", llm_response, height=200)

                if st.session_state['llm_eda_suggestions']:
                    st.markdown("---")
                    st.markdown("#### Review and Apply Suggested Cleaning Steps:")
                    selected_suggestions_indices = []
                    for i, suggestion in enumerate(st.session_state['llm_eda_suggestions']):
                        checkbox_label = f"{suggestion.get('column', 'N/A')}: {suggestion.get('action', 'N/A')} ({suggestion.get('details', 'N/A')})"
                        if st.checkbox(checkbox_label, key=f"llm_suggest_{i}"):
                            selected_suggestions_indices.append(i)
                    if st.button("Apply Selected Cleaning Steps", key="apply_llm_suggestions_btn"):
                        temp_df = df.copy()
                        applied_count = 0
                        st_info_dual("Applying selected cleaning steps...")
                        for idx in selected_suggestions_indices:
                            suggestion = st.session_state['llm_eda_suggestions'][idx]
                            _, action_applied = _apply_cleaning_action(temp_df, suggestion, st.session_state['llm_autoclean_log'])
                            if action_applied:
                                applied_count += 1
                        if applied_count > 0:
                            st.session_state['df_original'] = temp_df
                            st_success_dual(f"Successfully applied {applied_count} cleaning steps.")
                            st.session_state['llm_eda_suggestions'] = []
                            st.rerun()
                        else:
                            st_info_dual("No new cleaning steps were applied or selected.")
            else:
                st_info_dual("Please upload data to get AI cleaning suggestions.")
        
        with tab4_3:
            st.markdown("### 🤖 Custom AI Data Preparation (Powered by Perplexity AI)")
            st_info_dual("Provide instructions to the AI on how to clean and prepare your data. The AI will attempt to execute these instructions.")
            if st.session_state['df_original'] is None or st.session_state['df_original'].empty:
                st_info_dual("Please upload data to enable Custom AI Data Preparation.")
            else:
                if st.session_state['processed_df_for_custom_clean'] is None or st.button("Initialize Data for Custom Cleaning", key="init_custom_clean_btn"):
                    st.session_state['processed_df_for_custom_clean'] = st.session_state['df_original'].copy()
                    st.session_state['llm_autoclean_log'] = []
                    st_success_dual("Data initialized for custom cleaning. Provide your instructions below.")

                st.markdown("---")
                st.markdown("#### Current Data State for Custom Cleaning:")
                st_write_dual(f"Shape: {st.session_state['processed_df_for_custom_clean'].shape[0]} rows, {st.session_state['processed_df_for_custom_clean'].shape[1]} columns")
                st_dataframe_dual(st.session_state['processed_df_for_custom_clean'].head(5))
                st.markdown("---")

                user_cleaning_instruction_prompt = st.text_area(
                    "Enter your data cleaning instructions (e.g., 'Impute missing values in column Age with the median', 'Drop the column named Notes if more than 80% values are missing', 'Convert TransactionDate to datetime format %Y-%m-%d', 'Normalize Income and Salary columns'):",
                    height=150,
                    key="user_cleaning_instruction_prompt"
                )
                if st.button("Run Custom Cleaning (Perplexity)", key="run_custom_clean_btn"):
                    if user_cleaning_instruction_prompt:
                        with st.spinner("Perplexity AI is processing your cleaning instructions..."):
                            current_df = st.session_state['processed_df_for_custom_clean']
                            custom_clean_context = f"Current Data Columns and Types:\n{current_df.dtypes.to_markdown()}\n\n"
                            custom_clean_context += f"Missing Values:\n{current_df.isnull().sum()[current_df.isnull().sum() > 0].to_frame('Missing Count').to_markdown()}\n\n"
                            custom_clean_context += f"Numerical Column Statistics:\n{current_df.select_dtypes(include=np.number).describe().to_markdown()}\n\n"
                            custom_clean_context += f"Categorical Column Unique Counts & Top Values:\n"
                            for col in current_df.select_dtypes(include=['object', 'category', 'bool']).columns:
                                custom_clean_context += f"- {col}: Unique Count = {current_df[col].nunique()}, Top 5 values = {current_df[col].value_counts().head(5).index.tolist()}\n"
                            custom_clean_context += f"\nFirst 5 rows:\n{current_df.head().to_markdown(index=False)}"

                            custom_clean_system_instruction = (
                                "You are an autonomous and highly capable data cleaning and preprocessing agent. "
                                "Your goal is to thoroughly analyze the provided DataFrame summary and the user's cleaning instructions. "
                                "Based on these, output a series of precise, executable data cleaning actions in JSON format. "
                                "Prioritize actions that directly address the user's prompt, while also identifying and addressing other common data quality issues if relevant to the user's implied goal (e.g., preparing data for analysis/modeling). "
                                "Output ONLY a JSON array of objects. Each object MUST have `column`, `action`, `reason`, and action-specific parameters. "
                                "Valid actions are: `impute_missing`, `drop_rows_on_missing`, `drop_column`, `convert_type`, `normalize`.\n\n"
                                "**JSON Schema for each action object:**\n"
                                "{\n"
                                "  \"column\": \"[Column Name]\",\n"
                                "  \"action\": \"[impute_missing | drop_rows_on_missing | drop_column | convert_type | normalize]\",\n"
                                "  \"reason\": \"[Brief explanation of why this action is taken]\",\n"
                                "  \"method\": \"[mean | median | mode | constant]\", // Optional for impute_missing\n"
                                "  \"value\": \"[Constant value]\", // Required if method is 'constant'\n"
                                "  \"to_type\": \"[float | int | datetime | category]\", // Required for convert_type\n"
                                "  \"format\": \"[Datetime format string, e.g., '%Y-%m-%d %H:%M:%S']\" // Optional for convert_type to datetime\n"
                                "}\n\n"
                                "Ensure your output is valid JSON and contains only the array of cleaning actions. No other text or explanation."
                            )
                            pplx_user_message = f"Here is the current state of the dataset:\n{custom_clean_context}\n\nMy cleaning instructions are: {user_cleaning_instruction_prompt}"
                            pplx_raw_response = perplexity_chat(
                                question=pplx_user_message, df=current_df, system_message=custom_clean_system_instruction
                            )
                            if isinstance(pplx_raw_response, str) and pplx_raw_response.startswith("❌"):
                                st_error_dual(f"Perplexity AI cleaning failed: {pplx_raw_response}")
                                st_info_dual("Please check your API key and network connection. Sometimes the model might return an error.")
                                st.text_area("Raw AI Response (for debugging):", pplx_raw_response, height=200)
                            else:
                                try:
                                    if isinstance(pplx_raw_response, str) and pplx_raw_response.strip().startswith("```json"):
                                        pplx_raw_response = pplx_raw_response.strip()[7:-3].strip()
                                    cleaning_actions = json.loads(pplx_raw_response)
                                    if not isinstance(cleaning_actions, list):
                                        st_error_dual("Perplexity AI returned an invalid structure. Expected a list of cleaning actions.")
                                        st.text_area("Raw AI Response (for debugging):", pplx_raw_response, height=300)
                                        st.stop()
                                    actions_applied_count = 0
                                    new_log_entries = []
                                    for action in cleaning_actions:
                                        st.session_state['processed_df_for_custom_clean'], applied = _apply_cleaning_action(
                                            st.session_state['processed_df_for_custom_clean'].copy(), action, new_log_entries
                                        )
                                        if applied:
                                            actions_applied_count += 1
                                    st.session_state['llm_autoclean_log'].extend(new_log_entries)
                                    if actions_applied_count > 0:
                                        st.session_state['df_original'] = st.session_state['processed_df_for_custom_clean'].copy()
                                        st_success_dual(f"Perplexity AI Custom Clean completed! Applied {actions_applied_count} cleaning steps.")
                                    else:
                                        st_info_dual("Perplexity AI Custom Clean completed, but no new cleaning steps were applied based on your instructions.")
                                    st.rerun()
                                except json.JSONDecodeError as e:
                                    st_error_dual(f"Perplexity AI returned invalid JSON: {e}")
                                    st.text_area("Raw AI Response (for debugging):", pplx_raw_response, height=300)
                                except Exception as e:
                                    st_error_dual(f"An error occurred during custom cleaning: {e}")
                                    st.text_area("Raw AI Response (for debugging):", pplx_raw_response, height=300)
                    else:
                        st_warning_dual("Please enter your cleaning instructions to proceed.")

                st.markdown("---")
                st.markdown("#### Cleaning Log:")
                if st.session_state['llm_autoclean_log']:
                    for entry in reversed(st.session_state['llm_autoclean_log']):
                        st_write_dual(entry)
                else:
                    st_info_dual("No cleaning actions applied yet in this session.")

                st.markdown("---")
                st.markdown("#### Download Cleaned Data:")
                if st.session_state['processed_df_for_custom_clean'] is not None and not st.session_state['processed_df_for_custom_clean'].empty:
                    csv_output = io.StringIO()
                    st.session_state['processed_df_for_custom_clean'].to_csv(csv_output, index=False)
                    st.download_button(
                        label="Download Cleaned Data as CSV",
                        data=csv_output.getvalue(),
                        file_name="cleaned_data.csv",
                        mime="text/csv",
                        key="download_cleaned_csv"
                    )
                    excel_output = io.BytesIO()
                    st.session_state['processed_df_for_custom_clean'].to_excel(excel_output, index=False, engine='xlsxwriter')
                    excel_output.seek(0)
                    st.download_button(
                        label="Download Cleaned Data as Excel",
                        data=excel_output.getvalue(),
                        file_name="cleaned_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_cleaned_excel"
                    )
                else:
                    st_warning_dual("No data to download yet. Please load and clean some data.")

        with tab4_sandbox:
            st.markdown("### 🧪 AI Sandbox Model")
            st.info("""
            **How it works:**
            1.  **Ask a Question:** Pose a question or a task for the AI to perform on the data.
            2.  **Code Generation:** The AI will generate Python code to answer your question.
            3.  **Review & Execute:** You can review the code. When you execute, the AI will automatically attempt to fix any errors it encounters.
            4.  **Analysis & Follow-up:** The code's final output is sent back to the AI for an explanation. You can then ask follow-up questions.
            """)
            st.warning("""
            **⚠️ Security Warning:** The AI-generated code will be executed on the server running this app.
            While it operates within a constrained environment with a copy of your data, executing arbitrary code always carries risks.
            **Please review the code carefully before execution.** Never run code from an untrusted source.
            """)

            if df is not None and not df.empty:
                sandbox_model_choice = st.radio(
                    "Choose AI Model for the Sandbox:",
                    ("Google Gemini", "Perplexity AI", "mistral:latest"),
                    key="sandbox_model_choice",
                    horizontal=True
                )

                with st.form(key="sandbox_initial_query_form"):
                    sandbox_query = st.text_area(
                        f"Enter your initial query or task for {sandbox_model_choice}:",
                        height=100,
                        key="sandbox_query",
                        help="e.g., 'What is the average age?', 'List the top 5 most frequent categories in the product_type column'"
                    )
                    generate_code_button = st.form_submit_button("1. Generate Python Code")

                if generate_code_button and sandbox_query:
                    st.session_state.sandbox_conversation = []
                    with st.spinner(f"AI ({sandbox_model_choice}) is generating Python code..."):
                        generated_code = ""
                        if sandbox_model_choice == "Perplexity AI":
                            generated_code = generate_python_for_sandbox_pplx(sandbox_query, df)
                        elif sandbox_model_choice == "mistral:latest":
                            generated_code = generate_python_for_sandbox_mistrallatest(sandbox_query, df)
                        else: # Google Gemini
                            generated_code = generate_python_for_sandbox(sandbox_query, df)

                        if generated_code:
                            if generated_code.strip().startswith("```python"):
                                generated_code = generated_code[9:].strip()
                            if generated_code.endswith("```"):
                                generated_code = generated_code[:-3].strip()
                            generated_code = textwrap.dedent(generated_code).strip()

                        st.session_state.sandbox_conversation.append({
                            "query": sandbox_query,
                            "code": generated_code,
                            "output": None,
                            "explanation": None
                        })
                        st.rerun()

                if st.session_state.sandbox_conversation:
                    for i, turn in enumerate(st.session_state.sandbox_conversation):
                        st.markdown("---")
                        st.markdown(f"#### Your Question {i+1}:")
                        st_info_dual(turn["query"])

                        if turn["code"] and turn["output"] is None:
                            st.markdown("#### Generated Python Code (Review Carefully)")
                            if turn["code"].startswith("❌"):
                                st_error_dual(turn["code"])
                            else:
                                st_code_dual(turn["code"], language="python")
                                if st.button("2. Execute Code and Get Explanation", key=f"execute_{i}"):
                                    # The execute function now handles its own status UI
                                    code_output, final_explanation, corrected_code = execute_and_explain_sandbox(
                                        turn["query"], turn["code"], df, sandbox_model_choice
                                    )
                                    st.session_state.sandbox_conversation[i]["output"] = code_output
                                    st.session_state.sandbox_conversation[i]["explanation"] = final_explanation
                                    st.session_state.sandbox_conversation[i]["code"] = corrected_code
                                    st.rerun()

                        if turn["output"]:
                            st.markdown("#### Code Execution Output")
                            # The execute function now creates its own status box, so we just show the final result here
                            if "An error occurred" in turn["output"] or "A syntax error was found" in turn["output"]:
                                st_error_dual(turn["output"])
                            else:
                                st_text_dual(turn["output"])

                        if turn["explanation"]:
                            st.markdown("#### Final AI Explanation")
                            if turn["explanation"].startswith("❌"):
                                st_error_dual(turn["explanation"])
                            else:
                                st_success_dual(f"Analysis for Question {i+1} Complete!")
                                st_markdown_dual(turn["explanation"])

                    last_turn_complete = st.session_state.sandbox_conversation and st.session_state.sandbox_conversation[-1]["explanation"]
                    if last_turn_complete:
                        st.markdown("---")
                        st.markdown("### Ask a Follow-up Question")

                        with st.form(key="sandbox_followup_query_form"):
                            follow_up_query = st.text_area(
                                "Enter your follow-up question based on the analysis above:",
                                height=100,
                                key="follow_up_query"
                            )
                            ask_followup_button = st.form_submit_button("Ask Follow-up")

                        if ask_followup_button and follow_up_query:
                            with st.spinner(f"AI ({sandbox_model_choice}) is answering your follow-up..."):
                                conversation_context = []
                                for t in st.session_state.sandbox_conversation:
                                    conversation_context.append(f"User asked: {t['query']}")
                                    if t['code']:
                                        conversation_context.append(f"You generated and ran this code:\n```python\n{t['code']}\n```")
                                    if t['output']:
                                        conversation_context.append(f"The code produced this output:\n```\n{t['output']}\n```")
                                    if t['explanation']:
                                        conversation_context.append(f"You provided this explanation: {t['explanation']}")

                                context_string = "\n\n".join(conversation_context)
                                followup_system_prompt = (
                                    "You are an expert data analyst having a conversation. "
                                    "You previously performed an analysis based on a user's request. "
                                    "Below is the history of our conversation. Now, answer the user's follow-up question based on all the prior context. "
                                    "Provide a direct and concise answer."
                                )
                                followup_user_prompt = f"Here is our conversation so far:\n\n{context_string}\n\nMy follow-up question is: {follow_up_query}"

                                final_explanation = ""
                                if sandbox_model_choice == "Google Gemini":
                                    messages = [
                                        {"role": "system", "content": followup_system_prompt},
                                        {"role": "user", "content": followup_user_prompt}
                                    ]
                                    final_explanation = chat_bot(messages)
                                elif sandbox_model_choice == "Perplexity AI":
                                    final_explanation = perplexity_chat(
                                        question=followup_user_prompt, df=pd.DataFrame(), system_message=followup_system_prompt
                                    )
                                
                                st.session_state.sandbox_conversation.append({
                                    "query": follow_up_query,
                                    "code": None,
                                    "output": None,
                                    "explanation": final_explanation
                                })
                                st.rerun()
            else:
                st_info_dual("Please upload data to use the AI Sandbox.")


def smart_features_view():
    st.subheader("🛠️ Smart Features")
    pivot_tab, hypothesis_tab = st.tabs(["Pivot Table", "Hypothesis Testing"])

    with pivot_tab:
        if df is not None and not df.empty:
            st_markdown_dual("Create a custom pivot table for deeper analysis.")
            pivot_cols = df.columns.tolist()
            index_cols = st.multiselect("Select Row(s) (Index)", pivot_cols, key="pivot_index_cols")
            column_cols = st.multiselect("Select Column(s)", pivot_cols, key="pivot_column_cols")
            value_cols = st.multiselect("Select Value(s)", df.select_dtypes(include=np.number).columns.tolist(), key="pivot_value_cols")
            agg_func_options = ["sum", "mean", "median", "min", "max", "count", "nunique", "std"]
            agg_func = st.selectbox("Select Aggregation Function", agg_func_options, key="pivot_agg_func")

            if index_cols and value_cols:
                try:
                    pivot_table = pd.pivot_table(
                        df,
                        values=value_cols,
                        index=index_cols,
                        columns=column_cols if column_cols else None,
                        aggfunc=agg_func
                    )
                    st_dataframe_dual(pivot_table)
                    # MODIFIED: Removed the redundant pivot table summary block below.
                except Exception as e:
                    st_error_dual(f"Error creating pivot table: {e}. Ensure selected columns are appropriate for the chosen aggregation function.")
            else:
                st_info_dual("Please select at least one Row (Index) and one Value column to create a pivot table.")
        else:
            st_info_dual("Please upload data to create a pivot table.")
            
    with hypothesis_tab:
        st.subheader("🔬 Hypothesis Testing with Full Explanation")
        if df is not None and not df.empty:
            test_type = st.selectbox(
                "Select Hypothesis Test",
                ["Select a Test", "One-Sample T-Test", "Two-Sample T-Test", "ANOVA", "Chi-Squared Test", "Correlation Test", "Linear Regression"],
                key="hypothesis_test_type"
            )
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

            def show_explanation(title, description, null_hypothesis_text, alt_hypothesis_text):
                st_markdown_dual(f"### ❓ What is {title}?")
                st_info_dual(description)
                st_markdown_dual("### 📜 Hypotheses")
                st_markdown_dual(f"- **Null Hypothesis (H₀):** {null_hypothesis_text}\n- **Alternative Hypothesis (H₁):** {alt_hypothesis_text}")

            def display_result_table(result_dict):
                result_df = pd.DataFrame(list(result_dict.items()), columns=["Metric", "Value"])
                st_dataframe_dual(result_df, hide_index=True)

            if test_type == "One-Sample T-Test":
                if not numeric_cols:
                    st.warning("No numeric columns available for this test.")
                else:
                    col = st.selectbox("Select Numeric Column", numeric_cols, key="one_sample_col")
                    pop_mean = st.number_input("Hypothesized Population Mean (μ₀)", value=0.0, key="one_sample_pop_mean")
                    alpha = st.slider("Significance Level (α)", 0.01, 0.1, 0.05, step=0.01, key="one_sample_alpha")
                    null_h = f"The sample mean of **{col}** is equal to the hypothesized population mean (μ₀ = {pop_mean})."
                    alt_h = f"The sample mean of **{col}** is different from the hypothesized population mean (μ ≠ {pop_mean})."
                    show_explanation("One-Sample T-Test", "Used to determine whether the sample mean is significantly different from a known or hypothesized population mean.", null_h, alt_h)
                    if st.button("Run One-Sample T-Test", key="run_one_sample_ttest"):
                        data = df[col].dropna()
                        if not data.empty:
                            t_stat, p_value = stats.ttest_1samp(data.values, pop_mean)
                            display_result_table({
                                "T-Statistic": f"{t_stat:.4f}", "P-Value": f"{p_value:.4f}",
                                "Significance Level (α)": alpha, "Sample Mean": round(data.mean(), 4),
                                "Hypothesized Mean": pop_mean
                            })
                            if p_value < alpha:
                                st_success_dual(f"*Conclusion:* Based on the p-value ({p_value:.4f}) being less than the significance level ({alpha}), we *reject the null hypothesis*. This suggests that the sample mean of `{col}` is significantly different from the hypothesized population mean of `{pop_mean}`.")
                            else:
                                st_info_dual(f"*Conclusion:* Based on the p-value ({p_value:.4f}) being greater than or equal to the significance level ({alpha}), we *fail to reject the null hypothesis*. This suggests there is no statistically significant difference between the sample mean of `{col}` and the hypothesized population mean of `{pop_mean}`.")
                        else:
                            st_warning_dual(f"No valid data in column '{col}' for the test.")

            elif test_type == "Two-Sample T-Test":
                if not numeric_cols or not categorical_cols:
                    st.warning("This test requires at least one numeric and one categorical column.")
                else:
                    num_col = st.selectbox("Select Numeric Column", numeric_cols, key="two_sample_num_col")
                    group_col = st.selectbox("Select Categorical Grouping Column", categorical_cols, key="two_sample_group_col")
                    alpha = st.slider("Significance Level (α)", 0.01, 0.1, 0.05, key="two_sample_alpha")
                    groups = df[group_col].dropna().unique()
                    group1_name, group2_name = None, None
                    if len(groups) < 2:
                        st_warning_dual(f"The selected grouping column '{group_col}' must have at least two unique groups for a Two-Sample T-Test. Found: {len(groups)}.")
                    elif len(groups) > 2:
                        st_info_dual(f"The grouping column '{group_col}' has more than two groups. Please select two specific groups to compare.")
                        g1, g2 = st.columns(2)
                        group1_name = g1.selectbox("Select Group 1", groups, key="group1_select")
                        group2_name = g2.selectbox("Select Group 2", [g for g in groups if g != group1_name], key="group2_select")
                    else:
                        group1_name, group2_name = groups[0], groups[1]
                        st_write_dual(f"Comparing: **{group1_name}** vs. **{group2_name}**")

                    if group1_name is not None and group2_name is not None:
                        null_h = f"The mean of **{num_col}** for **{group1_name}** is equal to the mean of **{num_col}** for **{group2_name}**."
                        alt_h = f"The mean of **{num_col}** for **{group1_name}** is different from the mean of **{num_col}** for **{group2_name}**."
                        show_explanation("Two-Sample T-Test", "Compares the means of two independent groups to determine if they are significantly different.", null_h, alt_h)
                        if st.button("Run Two-Sample T-Test", key="run_two_sample_ttest"):
                            data1 = df[df[group_col] == group1_name][num_col].dropna()
                            data2 = df[df[group_col] == group2_name][num_col].dropna()
                            if not data1.empty and not data2.empty:
                                t_stat, p_value = stats.ttest_ind(data1.values, data2.values, equal_var=False)
                                display_result_table({
                                    f"Mean of {group1_name}": round(data1.mean(), 4),
                                    f"Mean of {group2_name}": round(data2.mean(), 4),
                                    "T-Statistic": f"{t_stat:.4f}", "P-Value": f"{p_value:.4f}",
                                    "Significance Level (α)": alpha
                                })
                                if p_value < alpha:
                                    st_success_dual(f"*Conclusion:* Based on the p-value ({p_value:.4f}) being less than the significance level ({alpha}), we *reject the null hypothesis*. This suggests a statistically significant difference in the means of `{num_col}` between `{group1_name}` and `{group2_name}`.")
                                else:
                                    st_info_dual(f"*Conclusion:* Based on the p-value ({p_value:.4f}) being greater than or equal to the significance level ({alpha}), we *fail to reject the null hypothesis*. This suggests no statistically significant difference in the means of `{num_col}` between `{group1_name}` and `{group2_name}`.")
                            else:
                                st_warning_dual(f"Not enough valid data in one or both groups for column '{num_col}'.")

            elif test_type == "ANOVA":
                if not numeric_cols or not categorical_cols:
                    st.warning("This test requires at least one numeric and one categorical column.")
                else:
                    y_col = st.selectbox("Select Numeric Response Variable", numeric_cols, key="anova_y_col")
                    x_col = st.selectbox("Select Categorical Factor", categorical_cols, key="anova_x_col")
                    alpha = st.slider("Significance Level (α)", 0.01, 0.1, 0.05, key="anova_alpha")
                    null_h = f"The mean of **{y_col}** is equal across all groups of **{x_col}**."
                    alt_h = f"At least one group mean of **{y_col}** is different across the groups of **{x_col}**."
                    show_explanation("ANOVA (Analysis of Variance)", "Tests whether there are statistically significant differences between the means of three or more groups.", null_h, alt_h)
                    if st.button("Run ANOVA", key="run_anova"):
                        anova_df = df[[y_col, x_col]].dropna()
                        if anova_df[x_col].nunique() < 2:
                            st_warning_dual(f"The categorical factor '{x_col}' must have at least two unique groups for ANOVA. Found: {anova_df[x_col].nunique()}.")
                        else:
                            try:
                                model = ols(f'Q("{y_col}") ~ C(Q("{x_col}"))', data=anova_df).fit()
                                result = anova_lm(model)
                                st_markdown_dual("### ANOVA Table")
                                st_dataframe_dual(result)
                                p_value = result.iloc[0]['PR(>F)']
                                display_result_table({
                                    "F-Statistic": f"{result.iloc[0]['F']:.4f}", "P-Value": f"{p_value:.4f}",
                                    "Significance Level (α)": alpha
                                })
                                if p_value < alpha:
                                    st_success_dual(f"*Conclusion:* Based on the p-value ({p_value:.4f}) being less than the significance level ({alpha}), we *reject the null hypothesis*. This suggests there is a statistically significant difference in means of `{y_col}` across the groups of `{x_col}`.")
                                else:
                                    st_info_dual(f"*Conclusion:* Based on the p-value ({p_value:.4f}) being greater than or equal to the significance level ({alpha}), we *fail to reject the null hypothesis*. This suggests no statistically significant difference in means of `{y_col}` across the groups of `{x_col}`.")
                            except Exception as e:
                                st_error_dual(f"Error running ANOVA: {e}. Ensure data is suitable for ANOVA.")

            elif test_type == "Chi-Squared Test":
                if len(categorical_cols) < 2:
                    st.warning("This test requires at least two categorical columns.")
                else:
                    cat1 = st.selectbox("Select Categorical Variable 1", categorical_cols, key="chi_sq_cat1")
                    available_cat2 = [col for col in categorical_cols if col != cat1]
                    cat2 = st.selectbox("Select Categorical Variable 2", available_cat2, key="chi_sq_cat2")
                    alpha = st.slider("Significance Level (α)", 0.01, 0.1, 0.05, key="chi_sq_alpha")
                    null_h = f"There is no association (independence) between **{cat1}** and **{cat2}**."
                    alt_h = f"There is a significant association (dependence) between **{cat1}** and **{cat2}**."
                    show_explanation("Chi-Squared Test of Independence", "Tests if there is a significant association between two categorical variables.", null_h, alt_h)
                    if st.button("Run Chi-Squared Test", key="run_chi_squared"):
                        if not cat1 or not cat2 or cat1 == cat2:
                            st_warning_dual("Please select two different categorical variables.")
                        else:
                            contingency = pd.crosstab(df[cat1], df[cat2])
                            st_markdown_dual("### Contingency Table")
                            st_dataframe_dual(contingency)
                            if contingency.empty:
                                st_warning_dual("Contingency table is empty. Check if there are common values or enough data in selected columns.")
                            else:
                                try:
                                    chi2, p, dof, ex = chi2_contingency(contingency)
                                    display_result_table({
                                        "Chi-Squared Statistic": f"{chi2:.4f}", "P-Value": f"{p:.4f}",
                                        "Degrees of Freedom": dof, "Significance Level (α)": alpha
                                    })
                                    if p < alpha:
                                        st_success_dual(f"*Conclusion:* Based on the p-value ({p:.4f}) being less than the significance level ({alpha}), we *reject the null hypothesis*. This suggests a statistically significant association (dependence) between `{cat1}` and `{cat2}`.")
                                    else:
                                        st_info_dual(f"*Conclusion:* Based on the p-value ({p:.4f}) being greater than or equal to the significance level ({alpha}), we *fail to reject the null hypothesis*. This suggests no statistically significant association (independence) between `{cat1}` and `{cat2}`.")
                                except ValueError as e:
                                    st_error_dual(f"Could not perform Chi-Squared Test. Error: {e}. This often happens if the contingency table has cells with zero expected frequencies. Consider if your data has enough variation or try combining categories.")

            elif test_type == "Correlation Test":
                if len(numeric_cols) < 2:
                    st.warning("This test requires at least two numeric columns.")
                else:
                    col1 = st.selectbox("Select Variable 1", numeric_cols, key="corr_col1")
                    col2_options = [col for col in numeric_cols if col != col1]
                    col2 = st.selectbox("Select Variable 2", col2_options, key="corr_col2")
                    alpha = st.slider("Significance Level (α)", 0.01, 0.1, 0.05, key="corr_alpha")
                    null_h = f"There is no linear correlation between **{col1}** and **{col2}** (ρ = 0)."
                    alt_h = f"There is a linear correlation between **{col1}** and **{col2}** (ρ ≠ 0)."
                    show_explanation("Pearson Correlation Test", "Measures the strength and direction of linear relationship between two continuous variables.", null_h, alt_h)
                    if st.button("Run Correlation Test", key="run_correlation"):
                        corr_df = df[[col1, col2]].dropna()
                        if len(corr_df) < 2:
                            st_warning_dual("Not enough valid data points to calculate correlation. Need at least 2 complete pairs.")
                        else:
                            r, p = stats.pearsonr(corr_df[col1].values, corr_df[col2].values)
                            display_result_table({
                                "Correlation Coefficient (r)": f"{r:.4f}", "P-Value": f"{p:.4f}",
                                "Significance Level (α)": alpha
                            })
                            if p < alpha:
                                st_success_dual(f"*Conclusion:* Based on the p-value ({p:.4f}) being less than the significance level ({alpha}), we *reject the null hypothesis*. This suggests a statistically significant linear relationship between `{col1}` and `{col2}`.")
                            else:
                                st_info_dual(f"*Conclusion:* Based on the p-value ({p:.4f}) being greater than or equal to the significance level ({alpha}), we *fail to reject the null hypothesis*. This suggests no statistically significant linear relationship between `{col1}` and `{col2}`.")

            elif test_type == "Linear Regression":
                if not numeric_cols:
                    st.warning("This test requires at least one numeric column.")
                else:
                    y = st.selectbox("Select Dependent Variable (Y)", numeric_cols, key="lr_y")
                    X_options = [col for col in numeric_cols if col != y]
                    X = st.multiselect("Select Independent Variable(s) (X)", X_options, key="lr_X")
                    independent_vars_str = " and ".join([f"**{var}**" for var in X]) if X else "the independent variable(s)"
                    null_h = f"There is no linear relationship between **{y}** and {independent_vars_str} (all regression coefficients are zero)."
                    alt_h = f"There is a linear relationship between **{y}** and {independent_vars_str} (at least one regression coefficient is not zero)."
                    show_explanation("Linear Regression", "Explores the linear relationship between a dependent variable and one or more independent variables.", null_h, alt_h)
                    if st.button("Run Linear Regression", key="run_linear_regression"):
                        if not X:
                            st_warning_dual("Please select at least one independent variable.")
                        else:
                            reg_df = df[[y] + X].dropna()
                            if reg_df.empty:
                                st_warning_dual("No complete cases after dropping NaNs for the selected variables. Please check your data.")
                            else:
                                try:
                                    X_with_const = sm.add_constant(reg_df[X])
                                    model = sm.OLS(reg_df[y], X_with_const).fit()
                                    st.markdown("### Regression Results Summary")
                                    st_text_dual(model.summary())
                                    st.markdown("---")
                                    st.markdown("### Overall Model Significance (F-test)")
                                    overall_p_value = model.f_pvalue
                                    if overall_p_value < 0.05:
                                        st_success_dual(f"*Conclusion:* The overall p-value for the F-statistic is `{overall_p_value:.4f}`. Since this is less than 0.05, we *reject the null hypothesis*. This suggests that the model is statistically significant and at least one independent variable contributes to explaining the variance in the dependent variable.")
                                    else:
                                        st_info_dual(f"*Conclusion:* The overall p-value for the F-statistic is `{overall_p_value:.4f}`. Since this is greater than or equal to 0.05, we *fail to reject the null hypothesis*. This suggests that the model is not statistically significant, and the independent variables do not collectively explain a significant amount of variance in the dependent variable.")
                                except Exception as e:
                                    st_error_dual(f"Error running Linear Regression: {e}. Ensure data is suitable for regression.")
        else:
            st_warning_dual("Please upload a dataset to begin hypothesis testing.")

# --- Main Application Router ---
if st.session_state.get('df_original') is not None:
    # Navigation Bar
    cols = st.columns((2, 1, 1, 1, 1, 1, 2))
    with cols[1]:
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.page = 'landing'
            st.rerun()
    with cols[2]:
        if st.button("📄 Profile", use_container_width=True):
            st.session_state.page = 'data_profile'
            st.rerun()
    with cols[3]:
        if st.button("📈 Visualise", use_container_width=True):
            st.session_state.page = 'visualisation'
            st.rerun()
    with cols[4]:
        if st.button("🧠 AI Tools", use_container_width=True):
            st.session_state.page = 'ai_analysis'
            st.rerun()
    with cols[5]:
        if st.button("🛠️ Features", use_container_width=True):
            st.session_state.page = 'smart_features'
            st.rerun()
    st.markdown("<hr>", unsafe_allow_html=True)

    # Page Router
    page = st.session_state.get('page', 'landing')
    if page == 'landing':
        landing_page_view()
    elif page == 'data_profile':
        data_profile_view()
    elif page == 'visualisation':
        visualisation_view()
    elif page == 'ai_analysis':
        ai_analysis_view()
    elif page == 'smart_features':
        smart_features_view()
    else:
        landing_page_view()
else:
    # Initial screen before data is loaded
    st.title("🚀 Welcome to DataSense AI")
    st.markdown("An advanced tool for Exploratory Data Analysis, AI-powered insights, and statistical testing.")
    st.info("Please use the 'Load Data' section in the sidebar to upload a CSV/Excel file or connect to a database to begin.")