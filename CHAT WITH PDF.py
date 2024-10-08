import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
import langchain_google_genai
import langchain_google_genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAI
#from langchain.embeddings import OpenAIEmbeddings  # Use a biblioteca de embeddings do LangChain
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


# Configura a API da Google com a chave da Streamlit
#AIzaSyD1jCU0lRN4u7wN2FBEzXYihRxIzXox77o #GOOGLE_API_KEY
#genai.configure(api_key=st.secrets["AIzaSyD1jCU0lRN4u7wN2FBEzXYihRxIzXox77o"])
genai.configure(api_key=st.secrets["api_keys"]["google_genai"])

#genai.configure(api_key=st.secrets["api_keys"]["google_genai"])



def get_pdf_text(pdf_docs):                         # Define função para extrair texto de arquivos PDF
    text = ""                                       # Inicializa uma string vazia para armazenar o texto
    for pdf in pdf_docs:                            # Itera sobre cada arquivo PDF
        pdf_reader = PdfReader(pdf)                 # Lê o arquivo PDF
        for page in pdf_reader.pages:               # Itera sobre cada página do PDF
            text += page.extract_text()             # Extrai e adiciona texto da página
    return text                                     # Retorna o texto extraído


def get_text_chunks(text):                          # Define função para dividir texto em pedaços
    splitter = RecursiveCharacterTextSplitter(      # Cria um divisor de texto
        chunk_size=10000, chunk_overlap=1000)       # Define tamanho do pedaço e sobreposição
    chunks = splitter.split_text(text)              # Divide o texto em pedaços
    return chunks  # list of strings                # Retorna lista de strings


def get_vector_store(chunks):                       # Define função para criar um armazenamento vetorial
    embeddings = GoogleGenerativeAIEmbeddings(      # Cria embeddings usando a AI do Google
        model="models/embedding-001")               # Especifica o modelo a ser usado    
    vector_store = FAISS.from_texts(chunks, embedding=embeddings) # Cria um vetor a partir dos pedaços de texto
    vector_store.save_local("faiss_index")          # Salva o armazenamento vetorial localmente


def get_conversational_chain():                     # Define função para criar uma cadeia de conversa #Define o template de prompt para perguntas e respostas
    prompt_template = """                           
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest",       # Cria um modelo de chat usando Gemini Pro
                                   client=genai,                        # Passa o cliente genai
                                   temperature=0.3,                     # Define a temperatura para a geração de texto
                                   )
    prompt = PromptTemplate(template=prompt_template,                   # Cria um template de prompt
                            input_variables=["context", "question"])    # Define variáveis de entrada
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt) # Carrega a cadeia de perguntas e respostas
    return chain # Retorna a cadeia


def clear_chat_history():                         # Define função para limpar o histórico do chat
    
    if os.path.isfile("faiss_index/index.faiss"): # Verifica se o arquivo do índice existe
        os.remove("faiss_index/index.faiss")      # Remove o arquivo do índice

    st.session_state.messages = [                 # Reseta as mensagens do estado da sessão
        {"role": "assistant", "content": "upload some pdfs and ask me a question"}]# Mensagem inicial do assistente
 

def user_input(user_question):                    # Define função para processar a pergunta do usuário
    embeddings = GoogleGenerativeAIEmbeddings(    # Cria embeddings usando a AI do Google
        model="models/embedding-001")             # Especifica o modelo a ser usado

    
    if not os.path.isfile("faiss_index/index.faiss"):                    # Verifica se o arquivo do índice existe
        st.error("No index file found. Please upload a document first.") # Exibe erro se não encontrar o índice
        return None                                                      # Retorna None

    new_db = FAISS.load_local("faiss_index", embeddings) # Carrega o armazenamento vetorial local
    if new_db is not None:                               # Verifica se o armazenamento foi carregado corretamente
        docs = new_db.similarity_search(user_question)   # Realiza busca de similaridade usando a pergunta do usuário
    else:
        st.error("Failed to load the database.")         # Exibe erro se falhar ao carregar o banco de dados
        return None                                      # Retorna None


    chain = get_conversational_chain() # Obtém a cadeia de conversa

    response = chain( # Chama a cadeia de perguntas e respostas
        {"input_documents": docs, "question": user_question}, return_only_outputs=True, ) # Passa documentos e pergunta


    print(response) # Imprime a resposta no console
    return response # Retorna a resposta


def main():  # Define a função principal do aplicativo
    st.set_page_config( # Configura a página do Streamlit
        page_title="Chat EMERSON", # Título da página
        page_icon=":gem:",  # Ícone da página
    )

    with st.sidebar:                       # Cria uma barra lateral
        st.title("Chat setting")           # Título da seção de configurações do chat
        pdf_docs = st.file_uploader(       # Cria um uploader de arquivos PDF
            "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)  # Permite múltiplos arquivos
        if st.button("Submit & Process"):  # Verifica se o botão foi clicado
            if not pdf_docs:               # Verifica se não há PDFs carregados
                st.error(                  # Exibe erro
                    "No PDF files uploaded. Please upload a file and try again.")
            else:
                with st.spinner("Processing..."):           # Mostra um carregando enquanto processa
                    raw_text = get_pdf_text(pdf_docs)       # Obtém texto bruto dos PDFs
                    text_chunks = get_text_chunks(raw_text) # Divide o texto em pedaços
                    get_vector_store(text_chunks)           # Cria o armazenamento vetorial a partir dos pedaços
                    st.success("Done") # Exibe mensagem de sucesso

    st.title("Chat with Emerson ❄🌳")  # Título do chat
    st.write(""" 
        Here you can chat with the AI model and upload PDFs to ask questions based on the PDFs.     
             """
             )                    # Escreve uma descrição
    st.sidebar.button('Clear & Reset Chat', on_click=clear_chat_history)                    # Botão para limpar e reiniciar o chat

    if "messages" not in st.session_state.keys():                                           # Verifica se não há mensagens no estado da sessão
        st.session_state.messages = [                                                       # Inicializa o histórico de mensagens
            {"role": "assistant", "content": "upload some pdfs and ask me a question"}]     # Mensagem inicial do assistente

    for message in st.session_state.messages:   # Itera sobre as mensagens no histórico
        with st.chat_message(message["role"]):  # Cria mensagem do chat
            if "content" in message:            # Verifica se a mensagem contém conteúdo
                st.write(message["content"])    # Escreve o conteúdo da mensagem
            else:
                st.error("Message content not found.") # Exibe erro se não encontrar conteúdo

    if prompt := st.chat_input():    # Espera pela entrada do usuário
        st.session_state.messages.append({"role": "user", "content": prompt})  # Adiciona a mensagem do usuário ao histórico
        with st.chat_message("user"): # Cria mensagem do usuário no chat
            st.write(prompt)          # Escreve a mensagem do usuário

    if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant": # Verifica se a última mensagem não é do assistente
        with st.chat_message("assistant"):                # Cria mensagem do assistente no chat
            with st.spinner("Thinking..."):               # Mostra um carregando enquanto processa
                response = user_input(prompt)             # Chama a função de entrada do usuário
                placeholder = st.empty()                  # Cria um espaço vazio para atualizar a resposta
                full_response = ''                        # Inicializa a resposta completa
                for item in response['output_text']:      # Itera sobre os itens da resposta
                    full_response += item                 # Concatena a resposta
                    placeholder.markdown(full_response)   # Atualiza a visualização da resposta
                placeholder.markdown(full_response)       # Mostra a resposta completa na interface
        if response is not None:                          # Verifica se a resposta não é None
            message = {"role": "assistant", "content": full_response} # Cria uma mensagem com a resposta do assistente
            st.session_state.messages.append(message)     # Adiciona a mensagem do assistente ao histórico


if __name__ == "__main__":   # Verifica se o script está sendo executado diretamente
    main()                   # Executa a função principal
