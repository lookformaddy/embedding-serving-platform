import math

from embedserve.embedders import HashingEmbedder
from embedserve.index import ExactIndex, QuantizedPartitionedIndex
from embedserve.models import EmbeddingRecord


def test_embed_determinism_and_normalization(model):
    emb = HashingEmbedder()
    a = emb.embed(["reset my password"], model)[0]
    b = emb.embed(["reset my password"], model)[0]
    assert a == b
    assert math.isclose(math.sqrt(sum(x * x for x in a)), 1.0, rel_tol=1e-6)
    assert emb.embed(["reset my password"], model)[0] != emb.embed(["track my package"], model)[0]


def test_embed_dimension_and_empty_text(model):
    emb = HashingEmbedder()
    assert len(emb.embed(["hello world"], model)[0]) == 32
    assert emb.embed([""], model)[0] == [0.0] * 32


def test_exact_search_ranks_identical_text_first(model):
    emb, idx = HashingEmbedder(), ExactIndex()
    texts = ["reset my password", "track my package", "refund my order"]
    for i, text in enumerate(texts):
        idx.upsert(EmbeddingRecord(item_id=f"doc-{i}", vector=emb.embed([text], model)[0],
                                   model_key=model.key, text=text))
    hits = idx.search(emb.embed(["reset my password"], model)[0], k=2)
    assert hits[0].item_id == "doc-0"
    assert hits[0].score > hits[1].score


def test_ann_recall_reasonable(model):
    emb = HashingEmbedder()
    exact, ann = ExactIndex(), QuantizedPartitionedIndex(dimension=32, n_partitions=8)
    corpus = [f"topic {i} about billing shipping returns account item {i}" for i in range(60)]
    for i, text in enumerate(corpus):
        rec = EmbeddingRecord(item_id=f"doc-{i}", vector=emb.embed([text], model)[0],
                              model_key=model.key, text=text)
        exact.upsert(rec)
        ann.upsert(rec)
    recalls = []
    for query_text in corpus[::10]:
        q = emb.embed([query_text], model)[0]
        exact_ids = {h.item_id for h in exact.search(q, k=5)}
        ann_ids = {h.item_id for h in ann.search(q, k=5, n_probe=8)}
        recalls.append(len(exact_ids & ann_ids) / len(exact_ids))
    assert sum(recalls) / len(recalls) >= 0.6
    # Full-probe ANN must contain the exact top hit for an identical query.
    q = emb.embed([corpus[0]], model)[0]
    assert ann.search(q, k=1, n_probe=8)[0].item_id == "doc-0"


def test_ann_delete_removes_item(model):
    emb = HashingEmbedder()
    ann = QuantizedPartitionedIndex(dimension=32)
    ann.upsert(EmbeddingRecord(item_id="x", vector=emb.embed(["hello"], model)[0],
                               model_key=model.key))
    assert len(ann) == 1
    assert ann.delete("x") is True
    assert ann.delete("x") is False
    assert len(ann) == 0
