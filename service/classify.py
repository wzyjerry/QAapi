import service_impl.classify_impl

loader = service_impl.classify_impl.ClassifyLoader(
    classify_stat='data/stat/classify_stat.pkl',
    pre_emb='data/cleaned_zh_vec')
model, d_word_index = loader.load_stat('data/model/classification')


def classify(text):
    return service_impl.classify_impl.classify(text, model, d_word_index)
