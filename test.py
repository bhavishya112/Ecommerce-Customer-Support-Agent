# from tools import summarize

# text = """
# Result 1
#                             Title: Future of the Indian Air Force - Wikipedia
#                             URL: https://en.wikipedia.org/wiki/Future_of_the_Indian_Air_Force
#                             Snippet: The Indian Air Force has been undergoing a modernization program to replace and upgrade outdated equipment since the late 1990s to meet modern standards.

# Result 2
#                             Title: Indian Army Reveals Bold Equipment Modernization Roadmap
#                             URL: https://thedefensepost.com/2025/07/11/indian-army-modernization-roadmap/
#                             Snippet: Jul 11, 2025 ... The Indian Army has outlined a modernization plan focused on hypersonic weapons, cyber capabilities, and other advanced systems, ...

# Result 3
#                             Title: How AI and Drones are transforming the Indian Army - YouTube
#                             URL: https://www.youtube.com/watch?v=e52FOLVELbU
#                             Snippet: Mar 24, 2026 ... ... Indian Army is recalibrating its doctrine, capabilities and organisational design for multi-domain conflict. New technologies and advancing ...

# Result 4
#                             Title: Developments in the Indian Military - 2025 - Delhi Policy Group
#                             URL: https://www.delhipolicygroup.org/publication/detail/developments-in-the-indian-military-2025
#                             Snippet: Jan 2, 2026 ... Drawing lessons from Operation Sindoor, the Indian Army's procurements in 2026 were focused on acquiring unmanned systems. It is also raising ...

# Result 5
#                             Title: After assuming charge as the 31st Chief of Army Staff, General ...
#                             URL: https://www.facebook.com/airnewsalerts/posts/after-assuming-charge-as-the-31st-chief-of-army-staff-general-dhiraj-seth-said-h/1395837269406637/
#                             Snippet: Jul 1, 2026 ... New-generation Munitions include 155mm Terminally Guided Munition and Canister Launched Munition," he added. He said that Indian Army's ...

# Result 6
#                             Title: Tech-Infused Warfare: Quantum Leap for Indian Army
#                             URL: https://raksha-anirveda.com/tech-infused-warfare-quantum-leap-for-indian-army/?srsltid=AfmBOopeJ6wyRnbyEt6l-hiB6NDjBBMo_qswPC4lcaHonAGh-kgCQF2W
#                             Snippet: Jan 14, 2025 ... Under the ambitious 'Modernise to Indianise' initiative, the Indian Army is not merely adopting the latest technologies — it is investing in ...

# Result 7
#                             Title: INDIAN ARMY UNVEILS TECHNOLOGY ROADMAP FOR DRONES ...
#                             URL: https://www.instagram.com/p/DW09kVJEzri/
#                             Snippet: Apr 7, 2026 ... The Indian armed forces are executing a monumental shift in modern warfighting doctrine. In its largest-ever unmanned systems initiative, India ...

# Result 8
#                             Title: The Remaking of the Indian Army Since Operation Sindoor
#                             URL: https://thediplomat.com/2025/10/the-remaking-of-the-indian-army-since-operation-sindoor/
#                             Snippet: Oct 30, 2025 ... This is evident from the establishment of five Bhairav battalions, two Rudra brigades, and the raising of Ashni drone platoons. The newly ...

# Result 9
#                             Title: Innovations | DDPMoD
#                             URL: https://www.ddpmod.gov.in/offerings/innovations
#                             Snippet: 1, Conversion of Tank T-72 into Autonomous Armoured Fighting Veh (AFV) Platform, ADITI 4, Indian Army, 17/03/2026 ; 2, 155mm Terminally Guided Munition, ADITI 4 ...

# Result 10
#                             Title: 72 Hours That Changed Bharat's Military Power - YouTube
#                             URL: https://www.youtube.com/watch?v=m3M7FPaZE2k
#                             Snippet: May 12, 2026 ... ... latest updates from PM Modi, government initiatives, and national ... The Moment India Stops Buying Israeli Weapons and Starts Making Them: Indian ...

#  """
# print(summarize(text,"what are the latest advancements in indian army"))

# import logging
# import pprint

# logger = logging.getLogger(__name__)
# logging.basicConfig(
#     filename=__name__,
#     filemode="a",  # 'a' appends new logs; 'w' overwrites the file each run
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     datefmt="%H:%M %d %B",
#     encoding="utf-8",
# )
# logger.info("test : %s", pprint.pformat("\033[93m"))

# from tools import query_ui
# result = query_ui(question="leave",view="mobile")
# # print(query_ui(question="batches sidebar",view="mobile"))
# for data in result:
#     for x in data:
#         print(f"{x} : {data[x]}")


# from openai import OpenAI

# client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# response = client.embeddings.create(input = ["hi","bye","sex","fex"],model="all-minilm:l6-v2")
# print(response.data[0].embedding)
# for emb in response.data:
#     print("emb ---- ",emb.embedding)
# Step 1: Find all items with page="schedule"

# from python.tools import search_products
# import chromadb

# client = chromadb.PersistentClient(path="data/vectordb/ui_vector_db2")
# collection = client.list_collections()
# print(collection)

# collection.delete(where={"page": "detailed_pschedule"})


# from sentence_transformers import CrossEncoder

# reranker = CrossEncoder(
#     "BAAI/bge-reranker-base",
#     device="cuda",
#     trust_remote_code=True
# )

# query = "Where is the completion rate shown?"

# docs = [
#     "Dashboard contains Recent Activity...",
#     "Completion Rate card is on the right side...",
#     "Certificates page...",
# ]

# pairs = [[query, doc] for doc in docs]

# scores = reranker.predict(pairs)

# ranked = sorted(
#     zip(scores, docs),
#     reverse=True
# )

# for score, doc in ranked:
#     print(score, doc)

# from tools import query_ui


# print(query_ui("dashboard",20,'desktop','ui_elements'))

# """
# 'function': {'name': 'search_products',
#     'arguments': '{"category": "electronics", "name": "",
#     "price": {"operator": "lt", "value": 100}, "supplier": ""}
# """


# print(search_products({"category": "electronics", "name": "", "price": {
#       "operator": "lt", "value": 100}, "supplier": ""}))

# import logging

# logger = logging.getLogger("test")
# logger.setLevel(logging.INFO)

# handler = logging.FileHandler("logs/test.log", encoding="utf-8")
# handler.setFormatter(logging.Formatter(
#     "%(asctime)s %(levelname)s %(message)s",
#     datefmt="%d %B %I:%M %p"
# ))


# logger.addHandler(handler)
# logger.propagate = False
# logger.info("chhhhhooooooooooooooooooot")

# from python.agent import ChatRequest, QueueEmitter, MODEL
# from threading import Thread
# import json
# from fastapi.responses import StreamingResponse
# from fastapi import FastAPI
# import time

# app = FastAPI()


# @app.post("/chat")
# async def chat(req: ChatRequest):
#     # add_or_update_conv(req.query, req.user_id, req.conv_id)
#     emitter = QueueEmitter()

#     def run_agent():
#         Agent(MODEL, req, emitter.emit)

#     Thread(target=run_agent).start()

#     async def stream():

#         while True:

#             event = await emitter.queue.get()

#             yield (
#                 f"event: {event['event']}\n"
#                 f"data: {json.dumps(event['data'], ensure_ascii=False)}\n\n"
#             )

#             if event["event"] == "done":
#                 break

#     return StreamingResponse(stream(), media_type="text/event-stream")


# def Agent(model: str, req: ChatRequest, emit: QueueEmitter):
#     emit("thinking", {"token": "fuck around you son of a bitch you bloodyy mother fuckeerrrrr"})
#     time.sleep(0.5)
#     emit("thinking", {"token": "you mother fucker pussy sucker gutter mutterr"})
#     time.sleep(0.5)
#     emit("thinking", {"token": "fucker"})

from python.tools import search_products, _vector_matches_for_field
import json
from python.fill_db import fetch_product_field_values
from python.db_connection import create_qdrant_connection, create_connection
mariadbclient = create_connection()
vectordbclient = create_qdrant_connection()


# print(search_products( {'category': 'gpu','price': [{'operator': 'gt', 'value': 300},{'operator': 'lt', 'value': 700}], 'supplier': ''}))
print(search_products({'category': 'graphics card',
      'name': 'nvidia', 'price': [], 'supplier': ''}))

# print(_vector_matches_for_field(vectordbclient,"supplier","Intel"))
# from python.fill_db import fetch_product_field_values

# x = fetch_product_field_values(mariadbclient)
# from IPython import embed
# embed()

# from python.tools import query_ui

# print(query_ui("purchase",5,"desktop"))
