"""Stage 1b: several memory layers sharing one value table (B-1M) and the Wikipedia data preparation."""
import io

import pytest
import torch

from smlm.model import ModelConfig, Transformer
from smlm.optim import build_optimizer
from smlm.textprep import assign_split, normalize_title, unit_hash, wikitext_article_titles
from smlm.train import MODELS

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def shared_model(share=True, seed=0):
    cfg = ModelConfig(vocab_size=128, d_model=32, n_layers=4, n_heads=4, ffn_hidden=64, max_seq_len=64,
                      mem_layers=[0, 2, 3], mem_n_keys=16, mem_heads=2, mem_knn=4, mem_k_dim=16,
                      mem_share_values=share)
    torch.manual_seed(seed)
    return Transformer(cfg).double()


def test_shared_table_is_one_parameter():
    m = shared_model()
    mems = m.memory_layers()
    assert all(x.values is mems[0].values for x in mems)
    assert len({id(x.keys) for x in mems}) == 3                         # keys / query stay per layer
    table = mems[0].values.weight
    assert sum(p is table for p in m.parameters()) == 1
    pc = m.param_counts()
    assert pc["memory_values"] == 16 ** 2 * 32                          # counted once
    assert pc["total"] == sum(p.numel() for p in m.parameters())
    assert pc["active_non_embedding_per_token"] == pc["dense_body"] + 3 * 2 * 4 * 32
    opt = build_optimizer(m, 1e-3, 2.4e-3, 0.1)
    vg = next(g for g in opt.param_groups if g["name"] == "memory_values")
    assert len(vg["params"]) == 1 and vg["params"][0] is table and vg["base_lr"] == 2.4e-3


def test_unshared_layers_have_separate_tables():
    m = shared_model(share=False)
    mems = m.memory_layers()
    assert len({id(x.values.weight) for x in mems}) == 3
    assert m.param_counts()["memory_values"] == 3 * 16 ** 2 * 32


@pytest.mark.parametrize("device", DEVICES)
def test_shared_table_gradient_only_on_rows_read_by_any_layer(device):
    m = shared_model().to(device)
    mems = m.memory_layers()
    for x in mems:
        x.record = True
    idx = torch.randint(0, 128, (2, 12), device=device)
    _, loss = m(idx[:, :-1], idx[:, 1:])
    loss.backward()
    read = torch.unique(torch.cat([x.last_indices.reshape(-1) for x in mems]))
    g = mems[0].values.weight.grad
    rows = torch.nonzero(g.abs().sum(-1) > 0).squeeze(-1)
    assert torch.equal(rows, read)
    assert read.numel() < mems[0].size
    # each layer contributes: rows read only by one layer still get gradient
    only0 = set(mems[0].last_indices.reshape(-1).tolist()) - set(
        torch.cat([mems[1].last_indices.reshape(-1), mems[2].last_indices.reshape(-1)]).tolist())
    assert all(g[i].abs().sum() > 0 for i in only0)


def test_shared_table_save_load_roundtrip_keeps_sharing():
    m = shared_model()
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    buf.seek(0)
    m2 = shared_model(seed=1)
    m2.load_state_dict(torch.load(buf))
    mems2 = m2.memory_layers()
    assert all(x.values is mems2[0].values for x in mems2)
    x = torch.randint(0, 128, (2, 10))
    m.eval(); m2.eval()
    with torch.no_grad():
        torch.testing.assert_close(m(x), m2(x))


def test_shared_model_causal_and_kv_cache():
    m = shared_model().eval()
    x = torch.randint(0, 128, (1, 30))
    y = x.clone()
    y[:, 20:] = torch.randint(0, 128, (1, 10))
    with torch.no_grad():
        full = m(x)
        torch.testing.assert_close(full[:, :20], m(y)[:, :20])
        caches = [dict() for _ in range(m.cfg.n_layers)]
        steps = [m(x[:, :10], kv_caches=caches, pos0=0)] + \
                [m(x[:, t:t + 1], kv_caches=caches, pos0=t) for t in range(10, 30)]
    torch.testing.assert_close(torch.cat(steps, 1), full)


def test_b1m_preset_matches_the_estimate():
    with torch.device("meta"):
        m = Transformer(ModelConfig(**MODELS["B-1M"]))
    pc = m.param_counts()
    assert MODELS["B-1M"]["mem_layers"] == [2, 6, 10]
    assert pc["memory_values"] == 1024 ** 2 * 384 == 402_653_184
    assert round(pc["total"] / 1e6, 1) == 444.9
    assert round(m.macs_per_token()["total"] / 1e6, 1) == 47.1


# ---- Wikipedia preparation helpers ----------------------------------------------------------------

def test_titles_match_across_wikitext_escaping():
    assert normalize_title("M @-@ 82 ( Michigan highway )") == normalize_title("M-82 (Michigan highway)")
    assert normalize_title("Meridian , Mississippi") == normalize_title("Meridian, Mississippi")
    assert normalize_title("2 @.@ 5 Something") == normalize_title("2.5 Something")
    assert normalize_title("Frank Headlam") != normalize_title("Frank Heady")


def test_wikitext_title_extraction_skips_sections():
    lines = ["", " = Homarus gammarus = \n", "", " = = Description = = \n", " text = with = signs \n",
             " = = = Sub = = = \n", " = Frank Headlam = \n"]
    assert wikitext_article_titles(lines) == ["Homarus gammarus", "Frank Headlam"]


def test_split_assignment_is_deterministic_and_disjoint():
    us = [unit_hash(i, 7) for i in range(20000)]
    assert us == [unit_hash(i, 7) for i in range(20000)]
    assert all(0.0 <= u < 1.0 for u in us)
    assert unit_hash(5, 7) != unit_hash(5, 8)
    splits = [assign_split(u, 0.0005, 0.1) for u in us]
    assert splits.count("validation") > 0 and splits.count("train") > 0
    for u, s in zip(us, splits):
        if s == "validation":
            assert u < 0.0005
        elif s == "train":
            assert 0.001 <= u < 0.101
        else:
            assert 0.0005 <= u < 0.001 or u >= 0.101
    with pytest.raises(AssertionError):
        assign_split(0.5, 0.01, 0.1)                                   # overlapping bands are refused
    # with the real fractions the bands cannot overlap: validation < 0.0003 <= 0.001 <= train
    assert all(assign_split(u, 0.0003, 0.125) != "train" for u in us if u < 0.0003)
