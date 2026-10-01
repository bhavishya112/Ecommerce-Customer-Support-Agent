## <p>1. Establish a baseline before touching anything</p>

Make a fixed evaluation dataset for your current implementation:

eval/<br>
├── questions.json<br>
├── expected.json<br>
└── run_eval.py

Each test case should contain something like:

```
{
  "id": "product_001",
  "query": "Show me black Nike running shoes under ₹5000",
  "category": "product_search",
  "expected_tools": ["search_products"],
  "expected_behavior": "return_matching_products"
}
```
<br>

## 2. Don't use one giant "agent accuracy" metric
```
                 Agent
                   │
       ┌───────────┼───────────┐
       ↓           ↓           ↓
   Routing       Tools         RAG
       │           │           │
       ↓           ↓           ↓
   selection    execution    retrieval
                   │           │
                   └─────┬─────┘
                         ↓
                     Response
```

<br>

## 3. Agent/tool-selection metrics

Measure:<br>
1. Tool Selection Accuracy = 
   ```
   correct tool selections
   ────────────────────────
   total tool-selection cases
   ```

2. Wrong Tool Rate = 
    ```
    wrong tool calls / total tool calls
    ```

3. Average Unnecessary Tool Calls = 
   ```
   unnecessary tool calls / requests
   ```
<br>

## 4. Tool execution metrics
For each tool, **record**:
```
tool_name
success/failure
p95 latency
input
output
error
```

<br>

## 5. RAG
**separate:**

```
Retrieval quality
        ↓
Generation quality
```

Recall@K<br>
```
relevant retrieved
──────────────────
total relevant
```

Precision@K<br>
```
relevant retrieved
──────────────────
K
```

MRR

Useful when the first relevant document matters.

<br>

## 6. Latency
```
                3.2 sec
                   │
       ┌───────────┼────────────┐
       ↓           ↓            ↓
     LLM         RAG          Tool
    1.4s         0.6s         1.0s
```

```
request_start
    ↓
LLM decision
    ↓
tool execution
    ↓
retrieval
    ↓
LLM response
    ↓
request_end
```
<br> 

## 7. Token usage

If your LLM provider exposes usage, record:
```
input_tokens
output_tokens
total_tokens
```

## 8. Your tools create some really good edge cases

I'd build an explicit edge-case suite.

**Product Search**
```
"nike"
"NIKE"
"Nike running shoes"
"nike runing shoes"        ← typo
"black shoes"
"black nike shoes"
"shoes under 5000"
"shoes between 3000 and 5000"
"cheapest Nike shoes"
"show me something similar"
```
And importantly:
```
"No products match this."
```
The agent shouldn't hallucinate products.

**Get Product**

Test:
```
valid product ID
invalid product ID
nonexistent product
missing product ID
ambiguous product name
```

**Orders**

```
valid order
nonexistent order
invalid order ID
order belonging to another user
missing order number
```

**UI Query**
```
element exists
element doesn't exist
empty search results
```

**Web Search**

Test:
```
normal query
empty query
ambiguous query
no results
search failure
timeout
irrelevant results
```
And especially:
```
question answerable from your own data
```
The agent shouldn't unnecessarily hit web search.

<br>

## 9. Failure handling
Test:

```
LLM failure
↓
tool failure
↓
database failure
↓
vector DB failure
↓
web search failure
↓
timeout
↓
invalid tool arguments
```

## 10. A very simple evaluation record

Just create your own JSONL:

```
{
  "id": "rag_001",
  "query": "What is your return policy?",
  "category": "rag",
  "expected_route": "rag",
  "expected_facts": [
    "30 days",
    "unused condition"
  ]
}
```
And your runner produces:
```
{
  "id": "rag_001",
  "route_correct": true,
  "retrieval_recall_at_5": 1.0,
  "latency_ms": 1832,
  "input_tokens": 742,
  "output_tokens": 186,
  "answer_correct": true,
  "tool_calls": 1,
  "errors": []
}
```
Then aggregate everything with Pandas.


## 11. Your final benchmark should look roughly like this

I'd compare:

```
Metric	Original	LangChain	LangGraph
Tool selection accuracy			
Unnecessary tool calls			
Tool success rate			
RAG Recall@5			
RAG Precision@5			
MRR			
Answer correctness			
Median latency			
P95 latency			
Token usage			
Memory accuracy	—	—	
Failure recovery	
```

## 12. And there's one metric I'd add specifically for your project
Regression rate

For every test that worked before:

```
Original: PASS
New:      FAIL
```
That's a regression.

Calculate:

```
regressions
───────────
total tests
```

<br>

## FINAL FORMAT

```
{
  "id": "product_001",
  "category": "product_search",

  "input": {
    "query": "Show me black Nike running shoes under ₹5000",
    "conversation": []
  },

  "expected": {
    "route": "product_search",
    "tools": [
      "search_products"
    ],
    "retrieved_doc_ids": [],
    "answer_facts": [
      "Nike",
      "black",
      "running shoes",
      "price <= 5000"
    ]
  },

  "observed": {

    "agent": {
      "route": "product_search",
      "tool_calls": [
        "search_products"
      ],
      "final_answer": "Here are some black Nike running shoes under ₹5000..."
    },

    "tools": [
      {
        "name": "search_products",
        "success": true,
        "latency_ms": 421,
        "input": {
          "query": "black Nike running shoes under ₹5000"
        },
        "output": {
          "product_ids": [
            "p123",
            "p456"
          ]
        }
      }
    ],

    "rag": {
      "used": false,
      "query": null,
      "retrieved_documents": []
    },

    "llm": {
      "calls": 2,
      "input_tokens": 1840,
      "output_tokens": 312,
      "latency_ms": 1730
    },

    "performance": {
      "total_latency_ms": 2380
    },

    "errors": []
  }
}
```