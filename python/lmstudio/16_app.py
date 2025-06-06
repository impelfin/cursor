import lmstudio as lms

downloaded = lms.list_downloaded_models()
llm_only = lms.list_downloaded_models("llm")
embedding_only = lms.list_downloaded_models("embedding")

for model in downloaded:
    print(model)
print("-" * 50)

for model in llm_only:
    print(model)
print("-" * 50)

for model in embedding_only:
    print(model)
print("-" * 50)
