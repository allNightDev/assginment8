import dotenv
from langchain_community.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import streamlit as st

dotenv.load_dotenv(override=True)

llm = None

answers_prompt = ChatPromptTemplate.from_template("""
Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make up an answer.
                                                  
Then, give a score to the answer between 0 and 5. 0 being not helpful to the user and 5 being helpful to the user.
                                                  
Example:
Question: How far away is the moon?
Answer: The moon is about 238,855 miles away from the Earth.
Score: 5

Question: How far away is the sun?
Answer: I don't know.
Score: 0
                                                  
Your turn!
                    
Context: {context}
Question: {question}
""")

choose_prompt = ChatPromptTemplate.from_messages([
    ("system", """
        Use ONLY the following pre-existing answers to answer the user's question.
        
        Use the answer that have the highest score (more helpful to the user) and favor the more recent ones.
        
        Cite sources. Do not modify the source, keep it as a link.
        
        Answers: {answers}
    """),
    ("human", "{question}"),
])

def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm

    return {
        "question" : question,
        "answers" : [
            {
                "answer" : answers_chain.invoke({
                    "question": question,
                    "context": doc.page_content
                }).content,
                "source" : doc.metadata["source"],
                "date" : doc.metadata["lastmod"],
            }
            for doc in docs
        ]
    }

def choose_answer(inputs):
    question = inputs["question"]
    answers = inputs["answers"]

    choose_chain = choose_prompt | llm

    condensed_answers = "\n\n".join(
        f"Answer: {answer['answer']}\nSource: {answer['source']}\nDate: {answer['date']}\n"
        for answer in answers
    )

    return choose_chain.invoke({
        "question" : question,
        "answers" : condensed_answers
    })

def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    
    return str(
        soup.get_text()
        .replace("\n", " ")
        .replace("\t", " ")
        .replace("\r", " ")
        .replace("\f", " ")
        .replace("\v", " ")
        .replace("\b", " ")
        .replace("\a", " ")
    )

@st.cache_resource(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200
    )

    loader = SitemapLoader(
        web_path=url,
        requests_per_second=2,
        continue_on_failure=True,
        filter_urls=[
            r"^(.*\/ai-gateway\/).*",
            r"^(.*\/vectorize\/).*",
            r"^(.*\/workers-ai\/).*",
        ],
        parsing_function=parse_page
    )

    docs = loader.load_and_split(text_splitter=splitter)
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()

with st.sidebar:
    url = st.text_input("URL을 입력하세요", placeholder="https://example.com/sitemap.xml")
    openai_api_key = st.text_input("OpenAI API Key를 입력하세요")

if url and openai_api_key:
    if ".xml" not in url:
        with st.sidebar:
            st.error("사이트맵 파일을 주세요")
    else:
        llm = ChatOpenAI(temperature=0.1, api_key=openai_api_key)

        retriever = load_website(url)
        query = st.text_input("질문을 입력하세요")

        if query:
            chain = {
                "docs": retriever,
                "question": RunnablePassthrough(),
            } | RunnableLambda(get_answers) | RunnableLambda(choose_answer)

            result = chain.invoke(query)
            st.write(result.content)