# Grimoire AI - AGENTIC CHATBOT
developer - BHAVISHYA SHARMA

[Video Demonstration of Project](working_demo.mp4)<br>
[Flow Chart of Project](flowchart.svg)

if you need further assistance with project setup or don't understand this documentation, email me at bhavishyasharma675@gmail.com
## Note
Because this project recieved updates from different timelines, i had to keep some irregularities to save time and maintain stability like :<br>
- for vectordb : Chromadb and Qdrant both ae used (should only be Qdrant)
- for embedding : SentenceTransformer and Ollama both are used (should only be Ollama)
- im converting some variables of ui elements (like size, position) to readable form in js itself (at the time of scraping) and color through python function (before indexing)
- im not using a reranker for UI RAG pipeline, but including one would make it scalable to many webpages




## Overview
This is a simple chatbot which has five tools : 1) web_search 2) query_ui 3) search_products 4) get_product 5) get_order:<br>
- web_search : searches web and gets the results in structured string format<br>
- query_ui : searches cached ui elements from vector_db given a short query string
- search_products: searches related products information given a particular category, supplier, price, name or a combination of these
- get_product: gets product details given a product_id
- get_order: get order details given a order_id

and rest is upto the LLM and System Prompt

## Specs
LLM API : groq (very generous people)<br>
Summarizing and Embedding : Locally via Ollama (and SentenceTransformers)<br>
Summarizing Model : ibm/granite4.1:3b |  Embedding Model : all-minilm:l6-v2<br>
Chat API : OpenAI chat completions<br>
Main LLM Model used : GPT-OSS-20B<br>
Web Search API : ddgs-python (again, very generous)<br>
VectorDB : Chromadb & Qdrant

## Local Setup
1. Download Ollama
2. Download Anaconda
3. Download Laragon Database (instead of xampp because it keeps crashing everytime)
   
- 1. Download the models (embedding and summarizing) listed in `specs` 
- 2. check your port number for ollama (im using it like this : ```client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="",
)``` )
1. Make your anaconda virtual environment using `environment.yml` file
2. Setup Qdrant on docker (if using windows)
3. Setup MySQL database importing `database.sql` file, use dbname = "online_store" only
4. Get your API key from groq

## Usage
To use the project, you'll have to do some preprocessing (if you want to use the query_ui feature otherwise skip these Phase1 and Phase2) : <br>
## Step 1.
### Phase 1
[for Query UI Tool]
1. Find **scraper.py** in root dir, run it like `python -m python.scraper https://url`
2. It would make data/website_snapshots/your_webpage.json

3. Find **flattenNindex.py**
4. Run it like :  `flattenNindex.py [-h] [--page PAGE] [--desc DESC] [--db DB] [--query QUERY] snapshot_file`<br>
  `--page` is the string html pagename that is shown to user <br>
  `--desc` is the short page description, currently useless, but useful for giving the agent particular webpage context<br>
  `--db` is string db name (optional)<br>
  `--query` is the string for a particular feature (optional)<br>
  `snapshot_file` is the path to the json file we just created<br><br>
  **Remember**
  - **you provide --query only when you want to test what the database retrieves otherwise leave**<br>
  - **Remember that you can provide collection name in this file by line `COLLECTION_NAME = "ui_elements"`** 
-  **If you provide collection name (default ui_elements), then change this line: `                        "enum": ["ui_elements"]` in root/python/agent.py**<br>
  
### Phase 2
[for Search Products Tool]
1. Fill your Database `online_store`
2. Run Qdrant [use defualt ports] in Docker and your SQL Database
3. Find fill_db.py and run it like `python -m python.fill_db full_url`
4. It will Cache Product related details, for Fuzzy name matching.

   
## Step 2.
1. Now start the agent server by command `uvicorn python.agent:app --host 127.0.0.1 --port 8001`(do not change port, otherwise reset it in backend.php)
2. Then Start php server by command `php -S php -S 127.0.0.1:8000`(i use it, or you can just copy paste the project into htdocs-like folder (in xampp))
3. Now just go localhost:8000 and start the thing
4. **remember** : this chatbot doesn't have Long-Term chat history context (i wanted to keep it simple), you can use your own RAG pipeline for that.
   <br> <br>
## Adding Tools
### Phase 1 - Make The Function Implementation
1. Go to tools.py
2. Make your function which logs to the variable `logger` currently it logs to `logs/tools.log` but you can change it by searching `logger.FileHandler` 

### Phase 2 - Add The Tool to the Agent
1. Search `Tool Definitions` in root/python/agent.py, youll see `TOOLS` variable, add your tool definition in Json format following that given structure strictly. **remember:** it is mandatory to give "required" = [_all parameters_], and "strict" = True, otherwise it raises error
2. Now Find `Tool Registry` and in it find the variable `AVAILABLE_TOOLS`, just import the tool and put it there.
  
## Logs
All the logging is done in `logs`<br>
For agent.py related it is `ai_backend.log`<br>
and for php related it is `php_logs.log`<br>
`test.log` is only for testing purposes<br>
`tools.log` is for tool calls




