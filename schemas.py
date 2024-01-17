"""
 /*************************************************************************
 * 
 * CONFIDENTIAL
 * __________________
 * 
 *  Copyright (2023-2024) AI Labs, IronOne Technologies, LLC
 *  All Rights Reserved
 *
 *  Author  : Theekshana Samaradiwakara
 *  Description :Python Backend API to chat with research paper data  
 *  CreatedDate : 10/01/2024
 *  LastModifiedDate : 10/01/2024
 *************************************************************************/
"""

from typing import Optional, List, Any, Dict
from pydantic import BaseModel


class Document(BaseModel):
    name: Optional[str]
    page_content: str
    metadata: Dict[str, Any]


class UserQuery(BaseModel):
    question: str

class UserQuery2(BaseModel):
    question: str
    chat_history: list = None

class ResponseModel(BaseModel):
    question: str
    answer: str
    source_documents: List[Document] = None

class ResponseModel2(BaseModel):
    success: str = None
    error: str = None
    documents: List[Document] # = None

   