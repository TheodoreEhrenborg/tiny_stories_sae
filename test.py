from lib import get_feature_vectors, get_feature_magnitudes
import torch



def test_running_decoder_manually():
    sae_hidden_dim = 100
    llm_hidden_dim = 768
    seq_len = 314
    decoder = torch.nn.Linear(sae_hidden_dim, llm_hidden_dim)
    sae_activations = torch.randn(1,seq_len, sae_hidden_dim, 1).abs()
    feat_vecs = get_feature_vectors(
        sae_hidden_dim, sae_activations, decoder.weight.transpose(0, 1)
    )
    reconstructed = torch.sum(feat_vecs, 2) + decoder.bias
    assert torch.allclose(
        decoder(sae_activations.squeeze(3)), reconstructed, atol=2e-5
    )

def test_same_loss():
    with torch.no_grad():
        sae_hidden_dim = 100
        llm_hidden_dim = 768
        seq_len = 314
        decoder = torch.nn.Linear(sae_hidden_dim, llm_hidden_dim)
        sae_activations = torch.randn(1,seq_len, sae_hidden_dim, 1).abs()
        feat_vecs = get_feature_vectors(
            sae_hidden_dim, sae_activations, decoder.weight.transpose(0, 1)
        )
        magnitudes1 = torch.linalg.vector_norm(feat_vecs, dim=3)
        l1_1 = torch.linalg.vector_norm(magnitudes1, ord=1)
        magnitudes2 = get_feature_magnitudes(
            sae_hidden_dim, sae_activations, decoder.weight.transpose(0, 1)
        )
        l1_2 = torch.linalg.vector_norm(magnitudes2, ord=1)
        assert magnitudes1.shape == magnitudes2.shape
        assert torch.allclose(magnitudes1, magnitudes2)
        assert torch.allclose(l1_1, l1_2)
