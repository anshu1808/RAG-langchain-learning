from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

from agent import get_llm, researcher_agent, writer_agent, editor_agent
from state import BlogState
from typing import Literal
max_revisions = 3

#### Nde - edges
### DEfining the Nodes

def researcher_node(state: BlogState):
    """ Researcher node - takes the topic and audience from the state and generates a research outline. """
    llm = get_llm()
    research = researcher_agent(llm, state.topic, state.audience, state.research_feedback)
    
    state.research = research
    state.research_feedback = ""
    return state

def human_research_feedback_node(state: BlogState):
    """check with human to approve the research or send the feedback"""
    decision = interrupt({
        "stage": "researcher_review",
        "research": state.research,
        "instructions":(
            "reply with 'approve' to approve the research outline or provide feedback for revision. ",
            "or describe the changes you want to see in the research outline. "
        )
    })
    if isinstance(decision, dict):
        feedback = decision.get("feedback", "")
        if feedback.lower() == "approve":
            state.research_feedback = ""
            return state
        else:
            state.research_feedback = feedback
            return state

def writer_node(state: BlogState):
    """ Writer node - takes the research outline from the state and generates a draft blog post. """
    llm = get_llm()
    draft = writer_agent(
        llm,
        state.topic,
        state.audience,
        feedback=state.draft_feedback,
        research=state.research,
    )
    state.draft = draft
    state.draft_feedback = ""
    return state

def human_writer_feedback_node(state: BlogState):
    """check with human to approve the draft or send the feedback"""
    decision = interrupt({
        "stage": "writer_review",
        "draft": state.draft,
        "instructions":(
            "reply with 'approve' to approve the draft blog post or provide feedback for revision. ",
            "or describe the changes you want to see in the draft blog post. "
        )
    })
    if isinstance(decision, dict):
        feedback = decision.get("feedback", "")
        if feedback.lower() == "approve":
            state.draft_feedback = ""
            return state
        else:
            state.draft_feedback = feedback
            return state

def editor_node(state: BlogState):
    """ Editor node - takes the draft blog post from the state and generates a final blog post. """
    llm = get_llm()
    final_blog = editor_agent(llm, state.topic, state.audience, state.draft)
    state.final_blog = final_blog
    return state

#########conditional edges

def route_after_research_feedback(state: BlogState) -> Literal["researcher", "writer"]:
    if state.research_feedback:
        return "researcher"
    return "writer"

def route_after_writer_feedback(state: BlogState) -> Literal["writer", "editor"]:
    if state.draft_feedback and state.revision_count < max_revisions:
        return "writer"
    return "editor"

###build and define the graph
def build_graph():
    graph = StateGraph(BlogState)
    graph.add_node("researcher", researcher_node)
    graph.add_node("human_research_feedback", human_research_feedback_node)
    graph.add_node("writer", writer_node)
    graph.add_node("human_writer_feedback", human_writer_feedback_node)
    graph.add_node("editor", editor_node)

    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "human_research_feedback")
    graph.add_conditional_edges("human_research_feedback", route_after_research_feedback,{"researcher": "researcher", "writer": "writer"})
    #graph.add_edge("human_research_feedback", "writer")
    graph.add_edge("writer", "human_writer_feedback")
    graph.add_conditional_edges("human_writer_feedback", route_after_writer_feedback, {"writer": "writer", "editor": "editor"})
    graph.add_edge("human_writer_feedback", "editor")
    graph.add_edge("editor", END)

    graph_final = graph.compile(checkpointer=InMemorySaver())
    return graph_final


#adding condition edges
    
    #graph_final.add_conditional_edge("human_writer_feedback", "writer", route_after_writer_feedback)