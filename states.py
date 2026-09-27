#so now we are creating a graph
#and the first thing you create is a state

import os

# 1) typed dict(most common approach)

from typing import TypedDict

class State(TypedDict):
    topic : str
    summary : str
    score : int


# 2) Pydantic approach

# it is good at data validation and type cheching at runtime

from pydantic import BaseModel,field_validator
class State(BaseModel):
    topic : str
    score : int
    summary : str = ""

    @field_validator
    def score_positive(cla,v):
        if v<0:
            raise ValueError("score must be positive")


#3) python data classes
#it is a standard python data class but it is used rarely

from dataclasses import dataclass,field

@dataclass
class State:
    topic : str = ""
    summary : str = ""
    messages : list = field(default_factory=list)


#4)

from langgraph.graph import MessagesState

class State(MessagesState):
    #message field is already included with add_messages reducer
    #just add your extra fields
    user_name:str
    language:str

