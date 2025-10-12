import torch
import pytest
from Bio import pairwise2 as pw
from Bio.Align import substitution_matrices

from forge.inference.esm_encoder import ESMEncoder
from forge.inference.sequence_decoder import SequenceDecoder
from raygun.pretrained import raygun_4_4mil_800M


@pytest.fixture(scope="session")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def blosum62():
    return substitution_matrices.load("BLOSUM62")


@pytest.fixture(scope="session")
def raygun_model(device):
    model = raygun_4_4mil_800M().eval().to(device)
    return model


@pytest.fixture(scope="session")
def esm_encoder(device):
    return ESMEncoder(device)


@pytest.fixture(scope="session")
def sequence_decoder(raygun_model):
    return SequenceDecoder(raygun_model)


def get_sequence_identity(
    gen_seq: str, ref_seq: str, blosum62, method="default"
) -> float:
    """Calculate sequence identity between two sequences."""
    align = pw.align.globaldx(ref_seq, gen_seq, blosum62)[0]
    aligned_seq1, aligned_seq2 = align[0], align[1]

    if method == "ignore-dash":
        zipped = [
            (r1, r2)
            for r1, r2 in zip(aligned_seq1, aligned_seq2)
            if (r1 != "-" and r2 != "-")
        ]
        aligned_seq1, aligned_seq2 = zip(*zipped)
        aligned_seq1 = "".join(aligned_seq1)
        aligned_seq2 = "".join(aligned_seq2)

    matches = sum(res1 == res2 for res1, res2 in zip(aligned_seq1, aligned_seq2))
    alignment_length = len(aligned_seq1)
    return matches / alignment_length


@pytest.mark.parametrize(
    "seq_id,ref_seq",
    [
        ("seq1", "MSSHKTFRIKRFLAKKQKQNRPIPQWIRMKTGNKIRYNSKRRHWRRTKLGL"),
        ("seq2", "MFRIEGLAPKLDPEEMKRKMREDVISSIRNFLIYVALLRVTPFILKKLDSI"),
        ("seq3", "MVTRFLGPRYRELVKNWVPTAYTWGAVGAVGLVWATDWRLILDWVPYINGKFKKDN"),
    ],
)
def test_decoder_identity(
    seq_id, ref_seq, esm_encoder, sequence_decoder, raygun_model, blosum62
):
    length = len(ref_seq)

    esm_embedding = esm_encoder.encode(ref_seq)
    raygun_embedding = raygun_model.encoder(esm_embedding)
    decoded_seq = sequence_decoder.decode(raygun_embedding, [length])[0]

    identity = get_sequence_identity(decoded_seq, ref_seq, blosum62)

    assert identity > 0.95, f"{seq_id} identity too low: {identity:.4f}"
