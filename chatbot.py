from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import DeepLake
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate

import streamlit as st

# load api keys
load_dotenv()

def llm_init():

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini")

    # set activeloop account
    my_activeloop_org_id = "sundiu"
    my_activeloop_dataset_name = "canadapost_guides"
    dataset_path = f"hub://sundiu/canadapost_guides"

    # connect Deeplake vector database
    embeddings = OpenAIEmbeddings()
    db = DeepLake(dataset_path=dataset_path, 
                embedding=embeddings, 
                overwrite=False)

    # config restriver; retrieve the closest 5 chucks of documents by cosine similarity
    retriever = db.as_retriever()
    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs['k'] = 5

    # Set up chat history prompt for retriever
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # main prompt
    system_prompt = (
        "You are a manager in Canada Post for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use five sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # rag chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    

    return rag_chain

def rag(rag_chain, input, chat_history):
    return rag_chain.invoke({"input": input, 
                             'chat_history':chat_history})

# Parse and display the context metadata in a tidy format
def display_context_metadata(context_metadata):
    # Use a set to store unique document titles
    unique_documents = set()
    
    for entry in context_metadata:
        # Extract the filename from the metadata source and remove the extension
        filename = entry.metadata['source'].split('/')[-1].replace('.pdf', '')
        
        # Convert filename to a more readable title format
        readable_title = filename.replace('-', ' ').title()
        
        # Add the formatted title to the set of unique documents
        unique_documents.add(readable_title)
    
    # Format the unique document titles as a numbered list
    formatted_references = "Reference:\n" + "\n".join(
        [f"{i+1}) {title}" for i, title in enumerate(unique_documents)]
    )
    
    return formatted_references

def main():
    rag_chain = llm_init()
    
    st.set_page_config(page_title="Ask a Canada Post Manager")
    st.title("Ask a Canada Post Manager")
    
    # check if there is chat history. If not, initialize a blank list
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history if any
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    if prompt := st.chat_input("Ask me anything about Canada Post."):
        
        # Append user's input to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user input
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response from rag()
        response = rag(rag_chain, 
                       input=prompt, 
                       chat_history = st.session_state.messages)

        # Display and store assistant response
        with st.chat_message("assistant"):
            
            # Display rag response
            st.markdown(response['answer'])
            
            # Display rag reference
            references_text = display_context_metadata(response['context'])
            st.markdown(references_text)
            
        st.session_state.messages.append({"role": "assistant", "content": response['answer']})
        
    
    
if __name__ == "__main__":
    main()