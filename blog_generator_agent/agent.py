import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

def get_llm(model_name:str = "openai/gpt-oss-120b", temperature:float = 0.5):
    api_key = api_key=os.environ.get("GROQ_API_KEY")
    llm = ChatGroq(model = model_name, temperature=temperature, api_key=api_key)
    return llm 


RESEARCHER_PROMPT = ChatPromptTemplate.from_message([
    {"role":"system", "content":"""
        "you are a Research agen. given a blog tool and target audience, produce a clear," structure reserach outline. include:\n"
        "1. 5-7 keys point the blog should cover\n"
        "2. Important facts, stats, or example for each point\n"
        "3. suggeest angle or hook\n"
        "Be concise and clear. do not write the blog, only provide a research outline.
    
    
"""
     },
     {"role":"user", "content":"Topic: {topic}, Audience: {audience},{revision_hint}, Write the research outline for the blog."
}])

def researcher_invoke(llm, topic:str, audience:str, feedback:str):
    research_prompt = RESEARCHER_PROMPT.format_messages(
        topic=topic,
        audience=audience,
        feedback=feedback,
        revision_hint=f"this is revision number for the feedback on your previous research- please check it.: {feedback}" if feedback else "this is the first draft"
    )
    research_response = llm.invoke(research_prompt)
    return research_response.content

writer_prompt = ChatPromptTemplate.from_message([
    {"role":"system", "content":"""
        "you are a blog writer. given a research outline, write a clear, structured blog post. include:\n"
        "1. An engaging introduction that hooks the reader\n"
        "2. Well-developed body paragraphs that cover each key point from the research outline\n"
        "3. A conclusion that summarizes the main points and provides a call to action or final thought\n"
        "4. Use a tone and style appropriate for the target audience\n"
        "5. Ensure the blog post is well-organized, coherent, and free of grammatical errors\n"
        "6. length: 500-800 words\n"
        "Be concise and clear. do not include any information that is not in the research outline\n"
    """},
    {"role":"user", "content":"Topic: {topic}, Audience: {audience}, Research: {research}, Revision: {revision_hint}, Write the blog post based on the research outline."}
])

def writer_agent(llm, topic:str, audience:str, feedback:str = "", research:str = ""):
    writer_prompt = writer_prompt.format_messages(
        topic=topic,
        audience=audience,
        research=research,
        feedback=feedback,
        revision_hint=f"this is revision number for the feedback on your previous research- please check it.: {feedback}" if feedback else "this is the first draft. please change the tone and style to be more engaging and appropriate for the target audience."
    )
    research_response = llm.invoke(writer_prompt)
    return research_response.content

##########editor prompt

editor_prompt = ChatPromptTemplate.from_message([
    {"role":"system", "content":"""
        "you are a blog editor. given a blog post, review and improve it for clarity, coherence, and engagement. include:\n"
        "1. Check for grammatical and spelling errors\n"
        "2. Ensure the blog post is well-structured and easy to follow\n"
        "3. Improve the introduction to better hook the reader\n"
        "4. Enhance the body paragraphs for better flow and readability\n"
        "5. Strengthen the conclusion to leave a lasting impression\n"
        "6. Adjust the tone and style to be more engaging and appropriate for the target audience\n"
        "7. Blog should be look like human written, not AI generated\n"
        "output only the final polished blog post, do not include any comments or explanations. Be concise and clear. do not include any information that is not in the blog post\n"
    """},
    {"role":"user", "content":"Topic: {topic}, Audience: {audience}, Draft: {draft}, return the published blog post."}
])

def editor_agent(llm, topic:str, audience:str, draft:str):
    editor_prompt = editor_prompt.format_messages(
        topic=topic,
        audience=audience,
        draft=draft
    )
    research_response = llm.invoke(editor_prompt)
    return research_response.content
      