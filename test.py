from lib import get_feature_vectors
import torch



def test_running_decoder_manually():
    sae_hidden_dim = 100
    llm_hidden_dim = 768
    seq_len = 314
    decoder = torch.nn.Linear(sae_hidden_dim, llm_hidden_dim)
    sae_activations = torch.randn(1,seq_len, sae_hidden_dim, 1)
    feat_vecs = get_feature_vectors(
        sae_hidden_dim, sae_activations, decoder.weight.transpose(0, 1)
    )
    reconstructed = torch.sum(feat_vecs, 2) + decoder.bias
    assert torch.allclose(
        decoder(sae_activations.squeeze(3)), reconstructed, atol=2e-5
    )
