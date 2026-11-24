from pydantic import BaseModel

class BlogState(BaseModel):
    #USER INPUT
    topic: str = ""
    audience: str = ""

    #researcher output
    research: str = ""
    research_feedback: str = ""

    #writer output
    draft: str = ""
    draft_feedback: str = ""

    #editor output
    final_blog: str = ""

    #metedata
    revision_count: int = 0
