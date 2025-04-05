from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph.graph import CompiledGraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


def getChatGraph(llm_model_name, db) -> CompiledGraph:
    graph_builder = StateGraph(State)
    llm = ChatOllama(model=llm_model_name)

    def chatbot(state: State):
        return {"messages": [llm.invoke(state["messages"])]}

    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    return graph_builder.compile()

condense_question = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:"""
# CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question)

answer = """
### Instruction:
You're a helpful research assistant, who answers questions based on provided research in a clear way and easy-to-understand way.
If there is no research, or the research is irrelevant to answering the question, simply reply that you can't answer.
Please reply with just the detailed answer and your sources. If you're unable to answer the question, do not list sources

## Research:
{context}

## Question:
{question}
"""
# ANSWER_PROMPT = ChatPromptTemplate.from_template(answer)

# DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(
#     template="Source Document: {source}, Page {page}:\n{page_content}"
# )


# def _combine_documents(
#     docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
# ):
#     doc_strings = [format_document(doc, document_prompt) for doc in docs]
#     return document_separator.join(doc_strings)


# memory = ConversationBufferMemory(return_messages=True, output_key="answer", input_key="question")


# def getStreamingChain(question: str, memory, llm, db):
#     retriever = db.as_retriever(search_kwargs={"k": 10})
#     loaded_memory = RunnablePassthrough.assign(
#         chat_history=RunnableLambda(
#             lambda x: "\n".join(
#                 [f"{item['role']}: {item['content']}" for item in x["memory"]]
#             )
#         ),
#     )

#     standalone_question = {
#         "standalone_question": {
#             "question": lambda x: x["question"],
#             "chat_history": lambda x: x["chat_history"],
#         }
#         | CONDENSE_QUESTION_PROMPT
#         | llm
#         | (lambda x: x.content if hasattr(x, "content") else x)
#     }

#     retrieved_documents = {
#         "docs": itemgetter("standalone_question") | retriever,
#         "question": lambda x: x["standalone_question"],
#     }

#     final_inputs = {
#         "context": lambda x: _combine_documents(x["docs"]),
#         "question": itemgetter("question"),
#     }

#     answer = final_inputs | ANSWER_PROMPT | llm

#     final_chain = loaded_memory | standalone_question | retrieved_documents | answer

#     return final_chain.stream({"question": question, "memory": memory})


# def getChatChain(llm, db):
#     retriever = db.as_retriever(search_kwargs={"k": 10})

#     loaded_memory = RunnablePassthrough.assign(
#         chat_history=RunnableLambda(memory.load_memory_variables)
#         | itemgetter("history"),
#     )

#     standalone_question = {
#         "standalone_question": {
#             "question": lambda x: x["question"],
#             "chat_history": lambda x: get_buffer_string(x["chat_history"]),
#         }
#         | CONDENSE_QUESTION_PROMPT
#         | llm
#         | (lambda x: x.content if hasattr(x, "content") else x)
#     }

#     # Now we retrieve the documents
#     retrieved_documents = {
#         "docs": itemgetter("standalone_question") | retriever,
#         "question": lambda x: x["standalone_question"],
#     }

#     # Now we construct the inputs for the final prompt
#     final_inputs = {
#         "context": lambda x: _combine_documents(x["docs"]),
#         "question": itemgetter("question"),
#     }

#     # And finally, we do the part that returns the answers
#     answer = {
#         "answer": final_inputs
#         | ANSWER_PROMPT
#         | llm.with_config(callbacks=[StreamingStdOutCallbackHandler()]),
#         "docs": itemgetter("docs"),
#     }

#     final_chain = loaded_memory | standalone_question | retrieved_documents | answer

#     def chat(question: str):
#         inputs = {"question": question}
#         result = final_chain.invoke(inputs)
#         memory.save_context(inputs, {"answer": result["answer"].content if hasattr(result["answer"], "content") else result["answer"]})

#     return chat
