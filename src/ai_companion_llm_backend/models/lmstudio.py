import lmstudio as lms

def get_lmstudio_models():
    llm = []
    try:
        downloaded_llm = lms.list_downloaded_models("llm")
        
        for m in downloaded_llm:
            llm.append(m.model_key)

        return llm
    except:
        return ["LM Studio를 설치하고 서버를 실행해주세요."]

def get_lmstudio_embedding_models():
    embedding = []
    try:
        downloaded_embedding = lms.list_downloaded_models("embedding")
        for m in downloaded_embedding:
            embedding.append(m.model_key)

        return embedding
    except:
        return ["LM Studio를 설치하고 서버를 실행해주세요."]