from transformers import AutoModel
from fastembed import SparseTextEmbedding

class Embedding:
    """
    Handles the creation and usage of dense and sparse embeddings.

    Attributes:
        dense: An instance of a dense embedding model loaded from transformers.
        sparse: An instance of a sparse embedding model from fastembed.
    """
    def __init__(self, dense_name: str, dense_revision: str, sparse_name: str):
        """
        Initializes the Embedding class with specified dense and sparse models.

        Args:
            dense_name: The name of the dense embedding model to load.
            dense_revision: The specific revision of the dense model to use.
            sparse_name: The name of the sparse embedding model to load.
        """
        self.dense = AutoModel.from_pretrained(dense_name, 
                                trust_remote_code=True, 
                                revision=dense_revision)
        self.sparse = SparseTextEmbedding(sparse_name)
    def encode_dense(self, text: str) -> list[float]:
        """
        Encodes the input text using the dense embedding model.

        Args:
            text: The input text string to encode.

        Returns:
            A list of floats representing the dense embedding of the text.
        """
        return self.dense.encode(text)
    def encode_sparse(self, text: str) -> list:
        """
        Encodes the input text using the sparse embedding model.

        Args:
            text: The input text string to encode.

        Returns:
            A dictionary containing the indices and corresponding values that define the sparse embedding. 
        """
        embed_list = map(
            lambda x: {"indices": x.indices.tolist(), "values": x.values.tolist()}, 
            self.sparse.embed(text)
        )
        return list(embed_list)[0]
    def encode_denses(self, texts: list[str]) -> list[float]:
        """
        Encodes list of the input text using the dense embedding model.

        Args:
            text: The input text string to encode.

        Returns:
            A list of floats representing the dense embedding of the text.
        """
        return self.dense.encode(texts)
    def encode_sparses(self, texts: list[str]) -> list:
        """
        Encodes list of the input text using the sparse embedding model.

        Args:
            text: The input text string to encode.

        Returns:
            A list of dictionary containing the indices and corresponding values that define the sparse embedding.
        """
        embed_list = map(
            lambda x: {"indices": x.indices.tolist(), "values": x.values.tolist()}, 
            self.sparse.embed(texts)
        )
        return list(embed_list)


if __name__ == "__main__":
    embedding = Embedding(
        dense_name="jinaai/jina-embeddings-v3",
        dense_revision="f1944de8402dcd5f2b03f822a4bc22a7f2de2eb9",
        sparse_name="Qdrant/bm42-all-minilm-l6-v2-attentions",
    )

    example_single_string = "Evaluating the performance of large language models."
    print(embedding.encode_dense(example_single_string))
    print(embedding.encode_sparse(example_single_string))
    example_list_of_strings = [
        "Techniques for generating technical documentation.",
        "Implementing efficient vector search algorithms."
    ]
    print(embedding.encode_denses(example_list_of_strings))
    print(embedding.encode_sparses(example_list_of_strings))