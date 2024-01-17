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
 *  Description :Python Backend API to chat with private data  
 *  CreatedDate : 14/11/2023
 *  LastModifiedDate : 18/11/2020
 *************************************************************************/
 """

from abc import ABC
from typing import Any, Dict, Optional, Tuple
# import json

from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.memory.utils import get_prompt_input_key
from langchain.pydantic_v1 import Field
from langchain.schema import BaseChatMessageHistory, BaseMemory

from typing import List, Union

# from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema.messages import BaseMessage, get_buffer_string


class BaseChatMemory(BaseMemory, ABC):
    """Abstract base class for chat memory."""

    chat_memory: BaseChatMessageHistory = Field(default_factory=ChatMessageHistory)
    output_key: Optional[str] = None
    input_key: Optional[str] = None
    return_messages: bool = False

    def _get_input_output(
        self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> Tuple[str, str]:
        

        if self.input_key is None:
            prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key

        if self.output_key is None:
            """
            output for agent with LLM chain tool                     = {answer} 
            output for agent with ConversationalRetrievalChain tool  = {'question', 'chat_history', 'answer','source_documents'}
            """
        
            LLM_key = 'output'
            Retrieval_key = 'answer'
            if isinstance(outputs[LLM_key], dict):
                Retrieval_dict = outputs[LLM_key]
                if Retrieval_key in Retrieval_dict.keys():
                    #output keys are 'answer' , 'source_documents'
                    output = Retrieval_dict[Retrieval_key]
                else:
                    raise ValueError(f"output key: {LLM_key} not a valid dictionary")

            else:
                #otherwise output key will be 'output'
                output_key = list(outputs.keys())[0]
                output = outputs[output_key]

            # if len(outputs) != 1:
            #     raise ValueError(f"One output key expected, got {outputs.keys()}")
                
            
        else:
            output_key = self.output_key
            output = outputs[output_key]

        return inputs[prompt_input_key], output

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        input_str, output_str = self._get_input_output(inputs, outputs)
        self.chat_memory.add_user_message(input_str)
        self.chat_memory.add_ai_message(output_str)

    def clear(self) -> None:
        """Clear memory contents."""
        self.chat_memory.clear()





class ConversationBufferWindowMemory(BaseChatMemory):
    """Buffer for storing conversation memory inside a limited size window."""

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "history"  #: :meta private:
    k: int = 5
    """Number of messages to store in buffer."""

    @property
    def buffer(self) -> Union[str, List[BaseMessage]]:
        """String buffer of memory."""
        return self.buffer_as_messages if self.return_messages else self.buffer_as_str

    @property
    def buffer_as_str(self) -> str:
        """Exposes the buffer as a string in case return_messages is True."""
        messages = self.chat_memory.messages[-self.k * 2 :] if self.k > 0 else []
        return get_buffer_string(
            messages,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

    @property
    def buffer_as_messages(self) -> List[BaseMessage]:
        """Exposes the buffer as a list of messages in case return_messages is False."""
        return self.chat_memory.messages[-self.k * 2 :] if self.k > 0 else []

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""
        return {self.memory_key: self.buffer}