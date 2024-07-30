from vdb import PineConeRetriver
from embedders import AllMiniLML6V2
from llm import ReplicateLLM
from rag import Prag
from gradio_ui import gradio_run

system_prompt = """
You are a dermetology expert. Your job is to anwer questions only from the context. If there is no context, Tell you are not sure about the information you are telling. 
"""



gradio_run()

emb = AllMiniLML6V2()
ret = PineConeRetriver(emb)
llm  = ReplicateLLM("meta/llama-2-13b-chat",system_prompt )
# rag = Prag(llm,ret,emb)

# a = rag.respond("can you tell me about pimples?")
# for event in a:
#     print(str(event))