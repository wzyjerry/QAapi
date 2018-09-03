import service_impl.ner_impl

loader = service_impl.ner_impl.NERLoader(
    ner_stat='data/stat/ner_stat.pkl',
    ner_train='data/aminer_train.dat',
    pre_emb='data/cleaned_zh_vec')
model, word_to_id, char_to_id, tag_to_id, id_to_tag = loader.load_stat('data/model/ner')


def ner(text):
    return service_impl.ner_impl.ner(text, model, word_to_id, char_to_id, tag_to_id, id_to_tag)
