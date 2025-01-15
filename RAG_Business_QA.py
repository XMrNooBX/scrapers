from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq
from pinecone import Pinecone
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from IPython.display import display_markdown

llmx=ChatMistralAI(model="mistral-large-latest",temperature=0.3,api_key="r1u9jBlZye7QrH3ymxkJjAMVd4VLoSEA")
llm = ChatGroq(temperature=0.2, model="llama-3.3-70b-versatile", api_key='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')

def get_llm_data(query, llm):
    system = """
You are a knowledgeable and approachable Business Studies professor with expertise in a wide range of topics.
Your role is to provide clear, easy, and engaging explanations to help students understand complex business concepts.
When answering:
- Provide any necessary calculations or numerical data related to the solution.
- Start with a high-level overview of the concept or solution, then dive into details as needed.
- Use examples, analogies, or step-by-step explanations to clarify ideas.
- Ensure your answers are accurate, well-structured, and easy to follow.
- If you don’t know the answer, acknowledge it and suggest ways to explore or research further.
"""

    user = """{query}
    """

    filtering_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("user", user)
        ]
    )

    filtering_chain = filtering_prompt | llm | StrOutputParser()

    response = filtering_chain.invoke({"query": query})

    return response

def clean_rag_data(query, context, llm):
    system = """
        You are a Highly capable Proffesor of understanding the value and context of both user queries and given data.
        Your Task for Documents Data is to analyze the list of document's content and properties and find the most important information regarding user's query.
        Your Task for Web Data is to analyze the web scraped data then summarize only useful data regarding user's query.
        You Must adhere to User's query before answering.

        Output:
            For Document Data
                Conclusion:
                    ...
            For Web Data
                Web Scarped Data:
                ...
    """

    user = """{context}
            User's query is given below:
            {question}
    """

    filtering_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("user", user)
        ]
    )

    filtering_chain = filtering_prompt | llm | StrOutputParser()

    response = filtering_chain.invoke({"context": context, "question": query})

    return response

def get_context(query):

    context = ""
    try:
        pc = Pinecone(api_key="pcsk_7T5yD_5dUhax1xCeRv6Tm1MdiDpBkGTpv41tcUzsH2VG671YW2gFaQYcGqF57QY3BFZWn")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="AIzaSyARa0MF9xC5YvKWnGCEVI4Rgp0LByvYpHw")
        index = pc.Index('business-docs')
        vector_store = PineconeVectorStore(index=index, embedding=embeddings)
        result = "\n\n".join([_.page_content for _ in vector_store.similarity_search(query, k=4)])
        clean_data = clean_rag_data(query, f"Documents Data \n\n{result}", llmx)
        context += f"Documents Data: \n\n{clean_data}"
    except:
        pass

    try:
        search = DuckDuckGoSearchRun()
        clean_data = clean_rag_data(query, search.invoke(query), llm)
        context += f"\n\nWeb Data:\n{clean_data}"
    except:
        pass

    context += f"\n\n LLM Data {get_llm_data(query, llm)}"

    return context

def respond_to_user(query, context, llm):
    system_prompt = """
You are a specialized professor in Business Studies. Your job is to answer the given question based on information from multiple sources, including web data, document data, and insights from the language model, providing a unified response.
When answering:
- Include all important information, key concepts, and takeaways from the context.
- Ensure the response is comprehensive, combining all insights seamlessly without explicitly separating the data sources.
- If applicable, provide any necessary calculations or numerical data related to the solution.
- Start with a high-level overview of the concept or solution, then dive into details as needed.
- Use examples, analogies, or step-by-step explanations to clarify ideas.
- Ensure your answer is clear, easy to understand, and approachable, even for someone with no prior knowledge of the subject.
"""
    user_prompt = """Question: {question}
    Context: {context} """

    rag_chain_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", user_prompt)
        ]
    )

    rag_chain = rag_chain_prompt | llm | StrOutputParser()

    response = rag_chain.invoke({"question": query, "context": context})

    return response

query = 'give me some tips for improving my startup'
context = get_context(query)
response = respond_to_user(query, context, llm)
display_markdown(response, raw=True)
